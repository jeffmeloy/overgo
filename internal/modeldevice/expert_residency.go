package modeldevice

import (
	"container/list"
	"context"
	"errors"
	"fmt"
	"math/bits"
	"slices"
	"sync"
	"unsafe"

	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

type expertKey struct {
	bank  string
	index uint32
}

const (
	// expertBankRank is the GGUF layout with input, output and expert axes.
	expertBankRank = 3
	// expertBankAxis is the expert axis of the GGUF expert-bank layout.
	expertBankAxis = expertBankRank - 1
	// expertAllocationFailure is CUDA_ERROR_OUT_OF_MEMORY from the CUDA Driver API.
	// Keep it aligned with the driver result contract used by allocation retry.
	expertAllocationFailure = 2
	// expertSlotAlignment: CUDA aligns an allocation to 256 bytes, so a slot
	// on that boundary gives a kernel the alignment the expert's own
	// allocation would.
	expertSlotAlignment = 256
	// routedWordBits: the experts one word of a routed-expert bitmap marks.
	routedWordBits = 32
	// expertHostChunkSlots: the host slots one pinned allocation adds once a
	// stride has none free. A promotion frees the slot of the expert it
	// brings to the card, so slots are added only as the card shrinks; a few
	// at a time keep the pinning calls few.
	expertHostChunkSlots = 16
	// expertFillHeadroomBytes: the card memory a fill without an explicit
	// limit leaves free, for CUDA library workspaces and module images, which
	// allocate outside the reservation hook, and a short request's scratch
	// (Flash-Next's short chat holds under 1 GiB beyond weights and experts).
	expertFillHeadroomBytes = 2 << 30
	// expertFillReaders: the file readers of a fill. Four read Flash-Next's
	// 59.5 GB of experts at 6.2 GB/s on the development PC, eight no faster,
	// one at 4.9 GB/s.
	expertFillReaders = 4
	// expertFillStagingBytes bounds the pinned buffer a reader fills with
	// experts bound for the card, so one copy moves many experts.
	expertFillStagingBytes = 64 << 20
)

// residentExpert is an expert on the card and the last advance whose routes named it.
type residentExpert struct {
	key  expertKey
	step uint64
}

// expertBank keeps where each of its experts lives: pointers is the host
// copy of its device table, and hosted holds the pinned bytes of each expert
// not on the card. A routed node's first bank holds the node's banks in group
// and the bitmap its routes mark at word routed.
type expertBank struct {
	info          gguf.TensorInfo
	bytes, stride uint64
	table         driver.DevicePtr
	pointers      []driver.DevicePtr
	hosted        [][]byte
	routed        uint64
	group         []*expertBank
	changed       bool
}

// expertClass keeps the experts of one stride: those on the card, least
// recently routed first, and the pinned host slots that hold none.
type expertClass struct {
	stride uint64
	order  list.List
	free   []expertSlot
}

// expertSlot is pinned host memory that holds one expert, at device on the card.
type expertSlot struct {
	device driver.DevicePtr
	host   []byte
}

// expertChunk is one card allocation of slots of one stride.
type expertChunk struct {
	pointer       driver.DevicePtr
	bytes, stride uint64
}

// expertMemory is the device surface a residency needs beyond a load.
type expertMemory interface {
	deviceMemory
	read(ctx context.Context, destination []byte, source driver.DevicePtr) error
	hostAlloc(ctx context.Context, size uint64) ([]byte, driver.DevicePtr, error)
	hostFree(data []byte) error
}

// expertResidency keeps every expert of the file-backed banks readable by
// the card: a card copy, or pinned host memory the card maps. The card's
// share is filled once under the limit; between runs the experts the last
// run routed to host memory take the card slots of experts it did not route.
type expertResidency struct {
	file              *gguf.File
	state             *device.State
	mem               expertMemory
	banks             map[string]*expertBank
	ordered           []*expertBank
	entries           map[expertKey]*list.Element
	classes           map[uint64]*expertClass
	cards             []expertChunk
	pinned            [][]byte
	tables, routedMap driver.DevicePtr
	routed            []uint32
	cleared           []byte
	bytes, tableBytes uint64
	step              uint64
}

// EnableExpertResidency binds file-backed banks after the ordinary weight load succeeds.
// The limit includes pointer tables and card experts but excludes other weight stores;
// zero takes the free device memory less expertFillHeadroomBytes.
func (w *DeviceWeights) EnableExpertResidency(ctx context.Context, file *gguf.File, infos []gguf.TensorInfo, limit uint64) error {
	if w == nil || w.worker == nil || file == nil || len(infos) == 0 {
		return errors.New("expert residency requires a worker, file and bank descriptors")
	}
	return w.worker.Do(ctx, func(state *device.State) error {
		if w.experts != nil {
			return errors.New("expert residency is already configured")
		}
		if limit == 0 {
			free, _, err := state.Driver.MemInfo()
			if err != nil {
				return err
			}
			limit = free - min(free, expertFillHeadroomBytes)
		}
		cache, err := newExpertResidency(ctx, file, infos, limit, stateMemory{state})
		if err != nil {
			return err
		}
		cache.state = state
		w.experts = cache
		return nil
	})
}

func newExpertResidency(ctx context.Context, file *gguf.File, infos []gguf.TensorInfo, limit uint64, mem expertMemory) (*expertResidency, error) {
	c := &expertResidency{file: file, mem: mem, banks: make(map[string]*expertBank),
		entries: make(map[expertKey]*list.Element), classes: make(map[uint64]*expertClass)}
	// Each table entry is a CUDA device address, whose ABI is uint64.
	const pointerBytes, wordBytes = uint64(unsafe.Sizeof(driver.DevicePtr(0))), uint64(unsafe.Sizeof(uint32(0)))
	var words uint64
	for _, info := range infos {
		listed, found := file.Tensor(info.Name)
		experts := info.Shape[expertBankAxis]
		if !found || listed != info || info.Dimensions != expertBankRank || experts == 0 || experts > uint64(^uint32(0)) || info.Size%experts != 0 || c.banks[info.Name] != nil {
			return nil, fmt.Errorf("expert bank %q requires one unique rank-three GGUF descriptor", info.Name)
		}
		stride, ok := checked.Align(info.Size/experts, expertSlotAlignment)
		if !ok {
			return nil, fmt.Errorf("expert bank %q slot size overflows", info.Name)
		}
		bank := &expertBank{info: info, bytes: info.Size / experts, stride: stride, routed: words,
			pointers: make([]driver.DevicePtr, experts), hosted: make([][]byte, experts)}
		if c.classes[stride] == nil {
			c.classes[stride] = &expertClass{stride: stride}
		}
		c.banks[info.Name] = bank
		c.ordered = append(c.ordered, bank)
		c.tableBytes += experts * pointerBytes
		words += (experts + routedWordBits - 1) / routedWordBits
	}
	c.tableBytes += words * wordBytes
	if c.tableBytes > limit {
		return nil, errors.New("expert pointer tables exceed the residency limit")
	}
	pointer, err := mem.alloc(ctx, c.tableBytes)
	if err != nil {
		return nil, err
	}
	c.tables, c.bytes = pointer, c.tableBytes
	for _, bank := range c.ordered {
		bank.table = pointer
		pointer += driver.DevicePtr(uint64(len(bank.pointers)) * pointerBytes)
	}
	c.routedMap, c.routed, c.cleared = pointer, make([]uint32, words), make([]byte, words*wordBytes)
	if err = mem.write(ctx, c.routedMap, c.cleared); err == nil {
		err = c.fill(ctx, limit)
	}
	if err != nil {
		return nil, errors.Join(err, c.close())
	}
	return c, nil
}

// expertRun is a run of a bank's experts the fill reads: into pinned host
// memory at host, or through a staging buffer to the card at card.
type expertRun struct {
	bank         *expertBank
	first, count uint64
	host         []byte
	card         driver.DevicePtr
}

// fill places experts in index order across the banks on the card while the
// limit allows, so each bank keeps the same share there, and every other
// expert in pinned host memory. Readers fill pinned memory concurrently, and
// the worker copies the card's runs as they arrive.
func (c *expertResidency) fill(ctx context.Context, limit uint64) (err error) {
	cards := make([]uint64, len(c.ordered))
	for index := range uint64(len(slices.MaxFunc(c.ordered, func(left, right *expertBank) int { return len(left.pointers) - len(right.pointers) }).pointers)) {
		for i, bank := range c.ordered {
			if index < uint64(len(bank.pointers)) && cards[i] == index && bank.stride <= limit-c.bytes {
				cards[i]++
				c.bytes += bank.stride
			}
		}
	}
	// staged: the staging buffer that holds any run bound for the card.
	var pending []expertRun
	var staged uint64
	for i, bank := range c.ordered {
		experts, onCard := uint64(len(bank.pointers)), cards[i]
		if onCard > 0 {
			staged = max(staged, bank.stride, min(expertFillStagingBytes, onCard*bank.stride))
			card, err := c.mem.alloc(ctx, onCard*bank.stride)
			if err != nil {
				return err
			}
			c.cards = append(c.cards, expertChunk{card, onCard * bank.stride, bank.stride})
			for index := range onCard {
				bank.pointers[index] = card + driver.DevicePtr(index*bank.stride)
				key := expertKey{bank.info.Name, uint32(index)}
				c.entries[key] = c.classes[bank.stride].order.PushBack(&residentExpert{key: key})
			}
			for first, step := uint64(0), max(1, expertFillStagingBytes/bank.stride); first < onCard; first += step {
				pending = append(pending, expertRun{bank: bank, first: first, count: min(step, onCard-first), card: bank.pointers[first]})
			}
		}
		if onCard < experts {
			host, mapped, err := c.mem.hostAlloc(ctx, (experts-onCard)*bank.stride)
			if err != nil {
				return err
			}
			c.pinned = append(c.pinned, host)
			for index := onCard; index < experts; index++ {
				offset := (index - onCard) * bank.stride
				bank.hosted[index], bank.pointers[index] = host[offset:offset+bank.stride], mapped+driver.DevicePtr(offset)
			}
			pending = append(pending, expertRun{bank: bank, first: onCard, count: experts - onCard, host: host})
		}
		bank.changed = true
	}
	runs := make(chan expertRun, len(pending))
	for _, run := range pending {
		runs <- run
	}
	close(runs)
	free := make(chan []byte, expertFillReaders)
	var staging [][]byte
	defer func() {
		for _, host := range staging {
			err = errors.Join(err, c.mem.hostFree(host))
		}
	}()
	for range expertFillReaders {
		if staged == 0 {
			break
		}
		host, _, allocErr := c.mem.hostAlloc(ctx, staged)
		if allocErr != nil {
			return allocErr
		}
		staging = append(staging, host)
		free <- host
	}
	readCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	uploads := make(chan expertRun)
	var readers sync.WaitGroup
	for range expertFillReaders {
		readers.Go(func() {
			if readErr := c.readRuns(readCtx, runs, free, uploads); readErr != nil {
				cancel(readErr)
			}
		})
	}
	go func() {
		readers.Wait()
		close(uploads)
	}()
	for run := range uploads {
		if readCtx.Err() == nil {
			if writeErr := c.mem.write(readCtx, run.card, run.host[:run.count*run.bank.stride]); writeErr != nil {
				cancel(writeErr)
			}
		}
		free <- run.host
	}
	if err = context.Cause(readCtx); err == nil {
		err = c.publish(ctx)
	}
	return err
}

// readRuns reads runs until none is left: a host run into its pinned memory,
// a card run into a staging buffer it passes on to the upload.
func (c *expertResidency) readRuns(ctx context.Context, runs <-chan expertRun, free <-chan []byte, uploads chan<- expertRun) error {
	for run := range runs {
		if run.host == nil {
			select {
			case run.host = <-free:
			case <-ctx.Done():
				return nil
			}
		}
		// A bank's experts lie back to back in the file, and in a run whose slots add no padding.
		together := uint64(1)
		if run.bank.stride == run.bank.bytes {
			together = run.count
		}
		for index := uint64(0); index < run.count; index += together {
			if ctx.Err() != nil {
				return nil
			}
			if err := c.file.ReadTensorRange(run.bank.info, (run.first+index)*run.bank.bytes, run.host[index*run.bank.stride:][:together*run.bank.bytes]); err != nil {
				return err
			}
		}
		if run.card != 0 {
			select {
			case uploads <- run:
			case <-ctx.Done():
				return nil
			}
		}
	}
	return nil
}

// IsExpertBank reports a bank whose experts are found through its table on the owning worker.
func (w *DeviceWeights) IsExpertBank(name string) bool {
	return w != nil && w.experts != nil && w.experts.banks[name] != nil
}

// HasExpertResidency distinguishes file-backed expert execution from full device residency.
func (w *DeviceWeights) HasExpertResidency() bool { return w != nil && w.experts != nil }

// ExpertMemory reports the card bytes of experts, which a reservation can release.
// Pointer tables remain resident and are excluded.
func (w *DeviceWeights) ExpertMemory(state *device.State) (uint64, error) {
	if w == nil || w.experts == nil || w.experts.state != state {
		return 0, errors.New("expert memory query requires its owning worker")
	}
	return w.experts.bytes - w.experts.tableBytes, nil
}

// ExpertRoutes returns the bitmap a routed node's routes mark, for the
// residency to read between runs. The nodes are the routed node's banks.
func (w *DeviceWeights) ExpertRoutes(state *device.State, nodes []*tensor.Tensor) (driver.DevicePtr, error) {
	if w == nil || w.experts == nil || w.experts.state != state || len(nodes) == 0 {
		return 0, errors.New("expert routes require their owning CUDA worker")
	}
	c := w.experts
	head := c.banks[nodes[0].Name]
	if head == nil || !slices.EqualFunc(head.group, nodes, func(bank *expertBank, node *tensor.Tensor) bool { return bank.info.Name == node.Name }) {
		group := make([]*expertBank, len(nodes))
		for i, node := range nodes {
			bank := c.banks[node.Name]
			if bank == nil || head == nil || node.Type != bank.info.Type || len(bank.pointers) != len(head.pointers) ||
				!slices.Equal(node.Shape.Dims[:node.Shape.Rank], bank.info.Shape[:bank.info.Dimensions]) {
				return 0, fmt.Errorf("expert bank %q differs from its file descriptor", node.Name)
			}
			group[i] = bank
		}
		head.group = group
	}
	return c.routedMap + driver.DevicePtr(head.routed*uint64(unsafe.Sizeof(uint32(0)))), nil
}

// AdvanceExperts applies the last run's routes between runs: each expert it
// routed to host memory moves to the card slot of an expert it did not route.
func (w *DeviceWeights) AdvanceExperts(ctx context.Context, state *device.State) error {
	if w == nil || w.experts == nil || w.experts.state != state {
		return errors.New("expert residency requires its owning CUDA worker")
	}
	return w.experts.advance(ctx)
}

func (c *expertResidency) advance(ctx context.Context) error {
	// A copy on the legacy stream waits for the last run's kernels.
	if err := c.mem.read(ctx, driver.Bytes(c.routed), c.routedMap); err != nil {
		return err
	}
	if !slices.ContainsFunc(c.routed, func(word uint32) bool { return word != 0 }) {
		return nil
	}
	if err := c.mem.write(ctx, c.routedMap, c.cleared); err != nil {
		return err
	}
	c.step++
	var missed []expertKey
	for _, head := range c.ordered {
		for word, set := range c.routed[head.routed : head.routed+(uint64(len(head.pointers))+routedWordBits-1)/routedWordBits] {
			for ; set != 0; set &= set - 1 {
				for _, bank := range head.group {
					key := expertKey{bank.info.Name, uint32(word*routedWordBits + bits.TrailingZeros32(set))}
					if entry := c.entries[key]; entry != nil {
						entry.Value.(*residentExpert).step = c.step
						c.classes[bank.stride].order.MoveToBack(entry)
					} else {
						missed = append(missed, key)
					}
				}
			}
		}
	}
	for _, key := range missed {
		if err := c.promote(ctx, key); err != nil {
			// The tables still name where each expert's bytes are.
			return errors.Join(err, c.publish(context.WithoutCancel(ctx)))
		}
	}
	return c.publish(ctx)
}

// promote copies a host expert to the card slot of the card expert of its
// stride the routes left longest unnamed; it stays in host memory when every
// such card expert was routed this step.
func (c *expertResidency) promote(ctx context.Context, key expertKey) error {
	bank := c.banks[key.bank]
	class := c.classes[bank.stride]
	victim := class.order.Front()
	if err := ctx.Err(); err != nil || victim == nil || victim.Value.(*residentExpert).step == c.step {
		return err
	}
	card, err := c.demote(ctx, victim)
	if err == nil {
		err = c.mem.write(ctx, card, bank.hosted[key.index][:bank.bytes])
	}
	if err != nil {
		return err
	}
	class.free = append(class.free, expertSlot{bank.pointers[key.index], bank.hosted[key.index]})
	bank.pointers[key.index], bank.hosted[key.index], bank.changed = card, nil, true
	c.entries[key] = class.order.PushBack(&residentExpert{key: key, step: c.step})
	return nil
}

// demote copies a card expert to a pinned host slot and returns the card slot it left.
func (c *expertResidency) demote(ctx context.Context, entry *list.Element) (driver.DevicePtr, error) {
	key := entry.Value.(*residentExpert).key
	bank := c.banks[key.bank]
	class := c.classes[bank.stride]
	if len(class.free) == 0 {
		host, mapped, err := c.mem.hostAlloc(ctx, expertHostChunkSlots*class.stride)
		if err != nil {
			return 0, err
		}
		c.pinned = append(c.pinned, host)
		for offset := uint64(0); offset < uint64(len(host)); offset += class.stride {
			class.free = append(class.free, expertSlot{mapped + driver.DevicePtr(offset), host[offset : offset+class.stride]})
		}
	}
	slot, card := class.free[len(class.free)-1], bank.pointers[key.index]
	if err := c.mem.read(ctx, slot.host[:bank.bytes], card); err != nil {
		return 0, err
	}
	class.free = class.free[:len(class.free)-1]
	bank.pointers[key.index], bank.hosted[key.index], bank.changed = slot.device, slot.host, true
	class.order.Remove(entry)
	delete(c.entries, key)
	return card, nil
}

// publish writes the tables whose experts moved.
func (c *expertResidency) publish(ctx context.Context) error {
	for _, bank := range c.ordered {
		if bank.changed {
			if err := c.mem.write(ctx, bank.table, driver.Bytes(bank.pointers)); err != nil {
				return err
			}
			bank.changed = false
		}
	}
	return nil
}

// ReserveExpertMemory makes room for another executor owner before allocation.
// It moves the experts of card allocations to host memory, last allocated
// first, and frees the allocations until the request fits.
func (w *DeviceWeights) ReserveExpertMemory(state *device.State, required uint64) error {
	if w == nil || w.experts == nil || w.experts.state != state {
		return errors.New("expert memory reservation requires its owning worker")
	}
	free, _, err := state.Driver.MemInfo()
	if err != nil || free >= required {
		return err
	}
	if err = state.Driver.StreamSynchronize(state.Stream); err != nil {
		return err
	}
	for free < required && len(w.experts.cards) > 0 {
		if err = w.experts.release(context.Background()); err != nil {
			return err
		}
		if free, _, err = state.Driver.MemInfo(); err != nil {
			return err
		}
	}
	if free < required {
		return fmt.Errorf("expert memory reservation needs %d free bytes, available %d: %w", required, free, &driver.ResultError{Operation: "expert reservation", Code: expertAllocationFailure})
	}
	return nil
}

// release moves the experts of the last card allocation to host memory,
// publishes their tables and frees the allocation.
func (c *expertResidency) release(ctx context.Context) error {
	chunk := c.cards[len(c.cards)-1]
	class := c.classes[chunk.stride]
	for entry := class.order.Front(); entry != nil; {
		next := entry.Next()
		key := entry.Value.(*residentExpert).key
		if pointer := c.banks[key.bank].pointers[key.index]; pointer >= chunk.pointer && uint64(pointer-chunk.pointer) < chunk.bytes {
			if _, err := c.demote(ctx, entry); err != nil {
				return err
			}
		}
		entry = next
	}
	if err := c.publish(ctx); err != nil {
		return err
	}
	c.cards, c.bytes = c.cards[:len(c.cards)-1], c.bytes-chunk.bytes
	return c.mem.free(ctx, []driver.DevicePtr{chunk.pointer})
}

// Close releases expert tables and experts after their actual device completion.
func (w *DeviceWeights) Close() error {
	if w == nil {
		return nil
	}
	if w.experts != nil {
		err := w.worker.Do(context.Background(), func(state *device.State) error {
			if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
				return err
			}
			if err := w.experts.close(); err != nil {
				return err
			}
			w.experts = nil
			return nil
		})
		if err != nil {
			return err
		}
	}
	return w.deviceTensorStore.Close()
}

func (w *DeviceWeights) expertInput(builder *tensor.Builder, name string) (*tensor.Tensor, driver.DevicePtr, error) {
	bank := w.experts.banks[name]
	shape, err := tensor.NewShape(bank.info.Shape[:bank.info.Dimensions]...)
	if err != nil {
		return nil, 0, err
	}
	node := builder.Input(name, bank.info.Type, shape)
	return node, bank.table, builder.Err()
}

func (c *expertResidency) close() error {
	if c == nil {
		return nil
	}
	var errs []error
	for _, chunk := range c.cards {
		errs = append(errs, c.mem.free(context.Background(), []driver.DevicePtr{chunk.pointer}))
	}
	if c.tables != 0 {
		errs = append(errs, c.mem.free(context.Background(), []driver.DevicePtr{c.tables}))
	}
	for _, host := range c.pinned {
		errs = append(errs, c.mem.hostFree(host))
	}
	c.cards, c.tables, c.pinned, c.entries, c.classes = nil, 0, nil, nil, nil
	return errors.Join(errs...)
}

// stateMemory performs synchronous transfers inside the existing worker callback.
type stateMemory struct{ state *device.State }

func (m stateMemory) alloc(ctx context.Context, size uint64) (driver.DevicePtr, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return m.state.Driver.MemAlloc(size)
}
func (m stateMemory) write(ctx context.Context, pointer driver.DevicePtr, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return m.state.Driver.MemcpyHtoD(pointer, data)
}
func (m stateMemory) read(ctx context.Context, data []byte, pointer driver.DevicePtr) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return m.state.Driver.MemcpyDtoH(data, pointer)
}
func (m stateMemory) free(_ context.Context, pointers []driver.DevicePtr) error {
	var errs []error
	for _, pointer := range pointers {
		errs = append(errs, m.state.Driver.MemFree(pointer))
	}
	return errors.Join(errs...)
}
func (m stateMemory) hostAlloc(ctx context.Context, size uint64) ([]byte, driver.DevicePtr, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	return m.state.Driver.MemHostAlloc(size)
}
func (m stateMemory) hostFree(data []byte) error { return m.state.Driver.MemFreeHost(data) }
