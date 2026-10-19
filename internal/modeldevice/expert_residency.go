package modeldevice

import (
	"bytes"
	"container/list"
	"context"
	"errors"
	"fmt"
	"slices"
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
	// expertBroadRouteShare: a run that routed more than 1/expertBroadRouteShare of a node's experts moves none to the card.
	//   - a decode step of Flash-Next routes 10 of 512 experts a token, a verify span a few times that
	//   - after its 1,561-token document the advance made about 19,900 copies, 1.9 s before the next run
	expertBroadRouteShare = 2
)

// expertDevice runs residency work where the residency's memory is usable:
// on the worker's thread, with its state, for a store with a worker.
type expertDevice func(context.Context, func(*device.State, expertMemory) error) error

// residentExpert is an expert on the card and the last advance whose routes named it.
type residentExpert struct {
	key  expertKey
	step uint64
}

// expertBank keeps where each of its experts lives: pointers is the host
// copy of its device table, and hosted holds the pinned bytes of each expert
// not on the card. A routed node's first bank holds the node's banks in group
// and the flags its routes mark from flag routed.
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
// share is filled once under the limit, by the load pass that reads the file;
// between runs the experts the last run routed to host memory take the card
// slots of experts it did not route.
// routed is pinned host memory the card maps at routedMap, one byte an expert, which the route kernel sets and the advance after the run reads and clears.
type expertResidency struct {
	state             *device.State
	mem               expertMemory
	banks             map[string]*expertBank
	ordered           []*expertBank
	entries           map[expertKey]*list.Element
	classes           map[uint64]*expertClass
	cards             []expertChunk
	pinned            [][]byte
	tables, routedMap driver.DevicePtr
	routed            []byte
	bytes, tableBytes uint64
	step              uint64
	// routes and cardRoutes are ExpertPlacement's counts.
	routes, cardRoutes uint64
}

// newExpertResidency allocates the tables, the route flags and every
// expert's slot for the file's banks, under the limit, which includes the
// tables and the card experts but no other weight store. The slots hold the
// experts once a load pass places them, and the tables name them once it
// publishes.
func newExpertResidency(ctx context.Context, file *gguf.File, infos []gguf.TensorInfo, limit uint64, mem expertMemory) (*expertResidency, error) {
	c := &expertResidency{mem: mem, banks: make(map[string]*expertBank),
		entries: make(map[expertKey]*list.Element), classes: make(map[uint64]*expertClass)}
	// Each table entry is a CUDA device address, whose ABI is uint64.
	const pointerBytes = uint64(unsafe.Sizeof(driver.DevicePtr(0)))
	var flags uint64
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
		bank := &expertBank{info: info, bytes: info.Size / experts, stride: stride, routed: flags,
			pointers: make([]driver.DevicePtr, experts), hosted: make([][]byte, experts)}
		if c.classes[stride] == nil {
			c.classes[stride] = &expertClass{stride: stride}
		}
		c.banks[info.Name] = bank
		c.ordered = append(c.ordered, bank)
		c.tableBytes += experts * pointerBytes
		flags += experts
	}
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
	routed, mapped, err := mem.hostAlloc(ctx, flags)
	if err == nil {
		clear(routed)
		c.pinned, c.routedMap, c.routed = append(c.pinned, routed), mapped, routed
		err = c.allocate(ctx, limit)
	}
	if err != nil {
		return nil, errors.Join(err, c.close())
	}
	return c, nil
}

// allocate places experts in index order across the banks on the card while
// the limit allows, so each bank keeps the same share there, and every other
// expert in pinned host memory: one card allocation and one pinned
// allocation a bank.
func (c *expertResidency) allocate(ctx context.Context, limit uint64) error {
	cards := make([]uint64, len(c.ordered))
	for index := range uint64(len(slices.MaxFunc(c.ordered, func(left, right *expertBank) int { return len(left.pointers) - len(right.pointers) }).pointers)) {
		for i, bank := range c.ordered {
			if index < uint64(len(bank.pointers)) && cards[i] == index && bank.stride <= limit-c.bytes {
				cards[i]++
				c.bytes += bank.stride
			}
		}
	}
	for i, bank := range c.ordered {
		experts, onCard := uint64(len(bank.pointers)), cards[i]
		if onCard > 0 {
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
		}
		bank.changed = true
	}
	return nil
}

// place copies file bytes of a bank, offset bytes into it, to their experts'
// slots: into pinned host memory directly, and to the card through onDevice,
// one copy for each run of card slots the bytes fill back to back.
func (c *expertResidency) place(ctx context.Context, onDevice expertDevice, bank *expertBank, offset uint64, data []byte) error {
	var card driver.DevicePtr
	var pending []byte
	flush := func() error {
		at, bytes := card, pending
		pending = nil
		if len(bytes) == 0 {
			return nil
		}
		return onDevice(ctx, func(*device.State, expertMemory) error { return c.mem.write(ctx, at, bytes) })
	}
	for done := uint64(0); done < uint64(len(data)); {
		index, within := (offset+done)/bank.bytes, (offset+done)%bank.bytes
		length := min(uint64(len(data))-done, bank.bytes-within)
		if host := bank.hosted[index]; host != nil {
			copy(host[within:], data[done:done+length])
		} else if pointer := bank.pointers[index] + driver.DevicePtr(within); len(pending) != 0 && card+driver.DevicePtr(len(pending)) == pointer {
			pending = data[done-uint64(len(pending)) : done+length]
		} else {
			if err := flush(); err != nil {
				return err
			}
			card, pending = pointer, data[done:done+length]
		}
		done += length
	}
	return flush()
}

// ExpertPlacement is where a residency's experts are and where its runs found the experts they routed.
//   - bytes are slot bytes, card memory and pinned host memory
//   - Routes counts the experts the runs' routes named, a bank's expert once a run
//   - CardRoutes counts those a run read on the card, and the rest it read over the PCIe link
//
// A run's routes count when the next run starts, because the residency reads them only between runs.
type ExpertPlacement struct {
	CardExperts, HostExperts uint64
	CardBytes, HostBytes     uint64
	Routes, CardRoutes       uint64
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

// ExpertPlacement reports where the experts are on their owning worker.
func (w *DeviceWeights) ExpertPlacement(state *device.State) (ExpertPlacement, error) {
	if w == nil || w.experts == nil || w.experts.state != state {
		return ExpertPlacement{}, errors.New("expert placement requires its owning worker")
	}
	c := w.experts
	placement := ExpertPlacement{Routes: c.routes, CardRoutes: c.cardRoutes}
	for _, class := range c.classes {
		placement.CardExperts += uint64(class.order.Len())
		placement.CardBytes += uint64(class.order.Len()) * class.stride
	}
	for _, bank := range c.ordered {
		placement.HostExperts += uint64(len(bank.pointers))
		placement.HostBytes += uint64(len(bank.pointers)) * bank.stride
	}
	placement.HostExperts -= placement.CardExperts
	placement.HostBytes -= placement.CardBytes
	return placement, nil
}

// ExpertRoutes returns the flags a routed node's routes mark, in host
// memory the card maps, for the residency to read between runs. The nodes
// are the routed node's banks.
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
	return c.routedMap + driver.DevicePtr(head.routed), nil
}

// HostedExpertBytes writes into hosted, for each expert of a routed node's
// banks, the bytes of its banks in pinned host memory, which the card reads
// over the PCIe link. The nodes are the routed node's banks.
func (w *DeviceWeights) HostedExpertBytes(state *device.State, nodes []*tensor.Tensor, hosted []uint64) error {
	if w == nil || w.experts == nil || w.experts.state != state {
		return errors.New("expert residency requires its owning CUDA worker")
	}
	clear(hosted)
	for _, node := range nodes {
		bank := w.experts.banks[node.Name]
		if bank == nil || len(bank.hosted) != len(hosted) {
			return fmt.Errorf("expert bank %q differs from its routed node", node.Name)
		}
		for index, host := range bank.hosted {
			if host != nil {
				hosted[index] += bank.bytes
			}
		}
	}
	return nil
}

// AdvanceExperts applies the last run's routes between runs: each expert it
// routed to host memory moves to the card slot of an expert it did not route,
// unless the run routed most of the node's experts.
func (w *DeviceWeights) AdvanceExperts(ctx context.Context, state *device.State) error {
	if w == nil || w.experts == nil || w.experts.state != state {
		return errors.New("expert residency requires its owning CUDA worker")
	}
	return w.experts.advance(ctx)
}

// advance reads the flags the last run's routes set, which its synchronization made visible, and clears them for the next run.
func (c *expertResidency) advance(ctx context.Context) error {
	if bytes.IndexByte(c.routed, 1) < 0 {
		// A table a failed publication left naming a moved expert's old slot is
		// written before any run reads it; a failed write refuses the run.
		return c.publish(ctx)
	}
	c.step++
	var missed []expertKey
	for _, head := range c.ordered {
		// Only a routed node's first bank has a group, and only its flags are set.
		if head.group == nil {
			continue
		}
		flags := c.routed[head.routed : head.routed+uint64(len(head.pointers))]
		// A run that routed most of a node's experts, a prompt, favours none of them, so its host experts stay there.
		broad := bytes.Count(flags, []byte{1})*expertBroadRouteShare > len(flags)
		for index, flag := range flags {
			if flag == 0 {
				continue
			}
			for _, bank := range head.group {
				key := expertKey{bank.info.Name, uint32(index)}
				c.routes++
				if entry := c.entries[key]; entry != nil {
					c.cardRoutes++
					entry.Value.(*residentExpert).step = c.step
					c.classes[bank.stride].order.MoveToBack(entry)
				} else if !broad {
					missed = append(missed, key)
				}
			}
		}
		clear(flags)
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
	c.cards, c.tables, c.pinned, c.routed, c.entries, c.classes = nil, 0, nil, nil, nil, nil
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
