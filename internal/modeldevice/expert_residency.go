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
	bank  *expertBank
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
	// expertRefillBytes: the card memory free beyond the headroom that moves
	// experts in host memory to the card between runs, as when a load found
	// the card held; memory a run frees for a moment moves none.
	expertRefillBytes = 2 << 30
	// expertStagedVictims: the coldest card experts of a stride kept copied to
	// pinned host memory ahead of need, so a promotion displaces one with no
	// copy back. Flash-Next's short chat promotes about five experts a token.
	expertStagedVictims = 32
	// expertVictimSearch: the coldest card experts of a stride a promotion looks
	// through for a staged one, those staged and those the routes named since.
	expertVictimSearch = 2 * expertStagedVictims
	// expertStreamNonBlocking is CU_STREAM_NON_BLOCKING: the residency's
	// copies run beside the synchronous writes of the legacy stream, which
	// would otherwise wait for them.
	expertStreamNonBlocking = 1
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
	// pointers is the bank's table on the host, the part of the residency's pointers from first.
	pointers []driver.DevicePtr
	first    int
	hosted   [][]byte
	routed   uint64
	group    []*expertBank
	changed  bool
	// head is the first bank of the routed node the bank serves, once one names it; a head's onHostStale marks its
	// group's card flags of experts on host for the next publish.
	head        *expertBank
	onHostStale bool
	// entries are the bank's card experts by index, nil for one in host memory; staged holds a card expert's landed
	// copy in pinned host memory, a slot with host bytes; moving marks an expert whose copy is in flight. An advance
	// reads them for every routed expert, where maps keyed by bank and index took about 0.5 ms of a Flash-Next window.
	entries []*list.Element
	staged  []expertSlot
	moving  []bool
	// class keeps the card experts of the bank's stride.
	class *expertClass
}

// stage keeps a card expert's landed copy in host memory.
func (b *expertBank) stage(index uint32, slot expertSlot) {
	b.staged[index] = slot
	b.class.staged++
}

// unstage forgets a card expert's copy in host memory.
func (b *expertBank) unstage(index uint32) {
	b.staged[index] = expertSlot{}
	b.class.staged--
}

// cardExperts counts the experts on the card.
func (c *expertResidency) cardExperts() uint64 {
	var experts uint64
	for _, class := range c.classes {
		experts += uint64(class.order.Len())
	}
	return experts
}

// expertClass keeps the experts of one stride: those on the card, least
// recently routed first, and the pinned host slots that hold none.
type expertClass struct {
	stride uint64
	order  list.List
	free   []expertSlot
	// staged counts the class's card experts whose copies in host memory have landed.
	staged int
}

// expertSlot is pinned host memory that holds one expert, at device on the card.
type expertSlot struct {
	device driver.DevicePtr
	host   []byte
}

// expertMove is a copy of an expert in flight: to its card slot card from
// its host slot, or from the card into its host slot.
type expertMove struct {
	key  expertKey
	card driver.DevicePtr
	slot expertSlot
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
	available(ctx context.Context) (uint64, error)
	// newStream makes a stream that copies beside the runs, and dropStream ends it.
	newStream(ctx context.Context) (driver.Stream, error)
	dropStream(stream driver.Stream) error
	// copyToCard and copyToHost queue a copy on stream of pinned host bytes to the card and of card bytes to pinned
	// host memory; copied reports whether every copy queued on stream has landed, waiting for them where wait is set.
	copyToCard(ctx context.Context, stream driver.Stream, destination driver.DevicePtr, source []byte) error
	copyToHost(ctx context.Context, stream driver.Stream, destination []byte, source driver.DevicePtr) error
	copied(ctx context.Context, stream driver.Stream, wait bool) (bool, error)
}

// expertResidency keeps every expert of the file-backed banks readable by
// the card: a card copy, or pinned host memory the card maps. The card's
// share is filled once under the limit, by the load pass that reads the file;
// between runs the experts the last run routed to host memory take the card
// slots of experts it did not route.
// routed is pinned host memory the card maps at routedMap, one byte an expert, which the route kernel sets and the advance after the run reads and clears.
// onHost is the card's flags at onHostFlags, laid out as routed, a routed node's first bank's byte set where every
// bank of the node holds the expert in host memory; the CPU computes those experts of a decode step (ExpertsOnHost).
type expertResidency struct {
	state                          *device.State
	mem                            expertMemory
	banks                          map[string]*expertBank
	ordered                        []*expertBank
	classes                        map[uint64]*expertClass
	cards                          []expertChunk
	pinned                         [][]byte
	tables, routedMap, onHostFlags driver.DevicePtr
	// pointers is every bank's table on the host, in the order the card's tables lie, each bank's pointers a part.
	pointers          []driver.DevicePtr
	routed, onHost    []byte
	onHostUnwritten   bool
	bytes, tableBytes uint64
	step              uint64
	// routes, cardRoutes and loadCard are ExpertPlacement's counts.
	routes, cardRoutes, loadCard uint64
	// limit bounds the card bytes a refill reaches: the load's limit, or none where the load read it from free memory.
	limit uint64
	// copies is the stream the residency copies on beside the runs. promoting are the promotions whose copies to
	// the card are in flight and staging the copies of card experts to host memory in flight, moving the experts
	// of either, and staged the card experts whose copy in host memory has landed.
	copies    driver.Stream
	promoting []expertMove
	staging   []expertMove
	// queuedPromoting and queuedStaging count the leading moves of promoting and staging whose copies are queued; an
	// advance records the rest, whose copies start once the next run is launched.
	queuedPromoting, queuedStaging int
}

// newExpertResidency allocates the tables, the route flags and every
// expert's slot for the file's banks, under the limit, which includes the
// tables and the card experts but no other weight store. The slots hold the
// experts once a load pass places them, and the tables name them once it
// publishes.
func newExpertResidency(ctx context.Context, file *gguf.File, infos []gguf.TensorInfo, limit uint64, mem expertMemory) (*expertResidency, error) {
	c := &expertResidency{mem: mem, banks: make(map[string]*expertBank),
		classes: make(map[uint64]*expertClass)}
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
			pointers: make([]driver.DevicePtr, experts), hosted: make([][]byte, experts),
			entries: make([]*list.Element, experts), staged: make([]expertSlot, experts), moving: make([]bool, experts)}
		if c.classes[stride] == nil {
			c.classes[stride] = &expertClass{stride: stride}
		}
		bank.class = c.classes[stride]
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
	c.pointers = make([]driver.DevicePtr, c.tableBytes/pointerBytes)
	c.tables, c.bytes = pointer, c.tableBytes
	first := 0
	for _, bank := range c.ordered {
		last := first + len(bank.pointers)
		bank.table, bank.pointers, bank.first = pointer+driver.DevicePtr(uint64(first)*pointerBytes), c.pointers[first:last:last], first
		first = last
	}
	routed, mapped, err := mem.hostAlloc(ctx, flags)
	if err == nil {
		clear(routed)
		c.pinned, c.routedMap, c.routed = append(c.pinned, routed), mapped, routed
		// The flags of experts on host start clear: until a publish sets them, the card reads every expert itself.
		c.onHost = make([]byte, flags)
		if c.onHostFlags, err = mem.alloc(ctx, flags); err == nil {
			err = mem.write(ctx, c.onHostFlags, c.onHost)
		}
	}
	if err == nil {
		err = c.allocate(ctx, limit)
		c.loadCard = c.cardExperts()
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
				key := expertKey{bank, uint32(index)}
				bank.entries[index] = c.classes[bank.stride].order.PushBack(&residentExpert{key: key})
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
//   - LoadCardExperts counts the experts the load placed on the card, few when it found the card held
//
// A run's routes count when the next run starts, because the residency reads them only between runs.
type ExpertPlacement struct {
	CardExperts, HostExperts uint64
	CardBytes, HostBytes     uint64
	Routes, CardRoutes       uint64
	LoadCardExperts          uint64
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
	placement := ExpertPlacement{Routes: c.routes, CardRoutes: c.cardRoutes, LoadCardExperts: c.loadCard}
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
	head, err := w.experts.routedGroup(nodes)
	if err != nil {
		return 0, err
	}
	return w.experts.routedMap + driver.DevicePtr(head.routed), nil
}

// ExpertsOnHost returns the card's flags of a routed node's experts, one byte an expert, set where every bank of the
// node holds the expert in host memory, as the last publish left them: a run reads them unchanged, and the hosted
// bytes ExpertSources names for those experts hold until the next advance. A node's flags are first set by the
// advance after it is named. The nodes are the routed node's banks.
func (w *DeviceWeights) ExpertsOnHost(state *device.State, nodes []*tensor.Tensor) (driver.DevicePtr, error) {
	if w == nil || w.experts == nil || w.experts.state != state || len(nodes) == 0 {
		return 0, errors.New("experts on host require their owning CUDA worker")
	}
	head, err := w.experts.routedGroup(nodes)
	if err != nil {
		return 0, err
	}
	for _, bank := range head.group {
		if bank.head != head {
			bank.head, head.onHostStale = head, true
		}
	}
	return w.experts.onHostFlags + driver.DevicePtr(head.routed), nil
}

// routedGroup returns the first bank of a routed node, its group the node's banks.
func (c *expertResidency) routedGroup(nodes []*tensor.Tensor) (*expertBank, error) {
	head := c.banks[nodes[0].Name]
	if head == nil || !slices.EqualFunc(head.group, nodes, func(bank *expertBank, node *tensor.Tensor) bool { return bank.info.Name == node.Name }) {
		group := make([]*expertBank, len(nodes))
		for i, node := range nodes {
			bank := c.banks[node.Name]
			if bank == nil || head == nil || node.Type != bank.info.Type || len(bank.pointers) != len(head.pointers) ||
				!slices.Equal(node.Shape.Dims[:node.Shape.Rank], bank.info.Shape[:bank.info.Dimensions]) {
				return nil, fmt.Errorf("expert bank %q differs from its file descriptor", node.Name)
			}
			group[i] = bank
		}
		head.group = group
	}
	return head, nil
}

// ExpertSources writes, for each expert of a routed node's bank, the address
// its table names into pointers and, for an expert in host memory, its pinned
// bytes into hosted, which a copy to the card reads at the link's rate.
func (w *DeviceWeights) ExpertSources(state *device.State, node *tensor.Tensor, pointers []driver.DevicePtr, hosted [][]byte) error {
	if w == nil || w.experts == nil || w.experts.state != state {
		return errors.New("expert residency requires its owning CUDA worker")
	}
	bank := w.experts.banks[node.Name]
	if bank == nil || len(bank.pointers) != len(pointers) || len(bank.hosted) != len(hosted) {
		return fmt.Errorf("expert bank %q differs from its routed node", node.Name)
	}
	copy(pointers, bank.pointers)
	for index, host := range bank.hosted {
		hosted[index] = host[:min(uint64(len(host)), bank.bytes)]
	}
	return nil
}

// AdvanceExperts applies the last run's routes between runs: each expert it
// routed to host memory moves to the card slot of an expert it did not route,
// unless the run was a prompt. A prompt's routes name most of a node's
// experts and favour none of them: after Flash-Next's 1,561-token document
// the moves took about 19,900 copies, 1.9 s before the next run. A decode
// step or a verify span moves its experts whatever share of a bank it names.
func (w *DeviceWeights) AdvanceExperts(ctx context.Context, state *device.State, prompt bool) error {
	if w == nil || w.experts == nil || w.experts.state != state {
		return errors.New("expert residency requires its owning CUDA worker")
	}
	return w.experts.advance(ctx, prompt)
}

// StartExpertCopies starts the copies the last advance recorded, once the run it prepared is launched.
func (w *DeviceWeights) StartExpertCopies(ctx context.Context, state *device.State) error {
	if w == nil || w.experts == nil {
		return nil
	}
	if w.experts.state != state {
		return errors.New("expert residency requires its owning CUDA worker")
	}
	return w.experts.start(ctx)
}

// advance reads the flags the last run's routes set, which its synchronization made visible, and clears them for the next run.
// The copies an earlier advance queued apply once they have landed, and the coldest card experts are staged for the next.
func (c *expertResidency) advance(ctx context.Context, prompt bool) error {
	if err := c.land(ctx, false); err != nil {
		return errors.Join(err, c.publish(context.WithoutCancel(ctx)))
	}
	if err := c.refill(ctx); err != nil {
		return errors.Join(err, c.publish(context.WithoutCancel(ctx)))
	}
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
		for index, flag := range flags {
			if flag == 0 {
				continue
			}
			for _, bank := range head.group {
				c.routes++
				if entry := bank.entries[index]; entry != nil {
					c.cardRoutes++
					entry.Value.(*residentExpert).step = c.step
					bank.class.order.MoveToBack(entry)
					// An expert the routes name is no victim; its staged copy's slot serves another.
					if slot := bank.staged[index]; slot.host != nil {
						bank.unstage(uint32(index))
						bank.class.free = append(bank.class.free, slot)
					}
				} else if !prompt && !bank.moving[index] {
					missed = append(missed, expertKey{bank, uint32(index)})
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
	if err := c.publish(ctx); err != nil || len(missed) == 0 {
		return err
	}
	return c.stage(ctx)
}

// land applies the copies the residency queued once they have landed,
// waiting for them where wait is set: a promoted expert takes its card slot
// and frees its host slot, and a staged copy is kept while its expert is on
// the card.
func (c *expertResidency) land(ctx context.Context, wait bool) error {
	// Moves no launched run started copying start here, before any lands.
	if err := c.start(ctx); err != nil {
		return err
	}
	if len(c.promoting) == 0 && len(c.staging) == 0 {
		return nil
	}
	done, err := c.mem.copied(ctx, c.copies, wait)
	if err != nil || !done {
		return err
	}
	for _, move := range c.promoting {
		bank, index := move.key.bank, move.key.index
		bank.pointers[index], bank.hosted[index], bank.changed = move.card, nil, true
		bank.class.free = append(bank.class.free, move.slot)
		bank.entries[index] = bank.class.order.PushBack(&residentExpert{key: move.key, step: c.step})
		bank.moving[index] = false
	}
	for _, move := range c.staging {
		bank, index := move.key.bank, move.key.index
		bank.moving[index] = false
		if bank.entries[index] != nil {
			bank.stage(index, move.slot)
		} else {
			bank.class.free = append(bank.class.free, move.slot)
		}
	}
	c.promoting, c.staging, c.queuedPromoting, c.queuedStaging = c.promoting[:0], c.staging[:0], 0, 0
	return nil
}

// stage copies the coldest card experts of each stride, those the routes
// left unnamed longest and not this step, to pinned host memory beside the
// next run, until expertStagedVictims of the stride are staged or staging.
func (c *expertResidency) stage(ctx context.Context) error {
	// Every staged or staging copy counts against its stride's budget, wherever the routes have since moved its expert.
	held := make(map[*expertClass]int, len(c.classes))
	for _, move := range c.staging {
		held[move.key.bank.class]++
	}
	for _, class := range c.classes {
		count := held[class] + class.staged
		for entry := class.order.Front(); entry != nil && count < expertStagedVictims; entry = entry.Next() {
			resident := entry.Value.(*residentExpert)
			if resident.step == c.step {
				break
			}
			bank, index := resident.key.bank, resident.key.index
			if bank.staged[index].host != nil || bank.moving[index] {
				continue
			}
			count++
			slot, err := c.hostSlot(ctx, class)
			if err != nil {
				return err
			}
			// The copy starts once the next run is launched (start).
			c.staging = append(c.staging, expertMove{key: resident.key, card: bank.pointers[index], slot: slot})
			bank.moving[index] = true
		}
	}
	return nil
}

// start queues the copies of the moves an advance recorded, on the residency's stream: an advance runs before the
// run it prepares, whose own small copies to the card would otherwise wait behind them on the copy engine (Flash-Next
// decode steps waited about 0.5 ms a step). A move whose copy is refused is dropped: a promoted expert stays in host
// memory and a staging copy's slot is freed.
func (c *expertResidency) start(ctx context.Context) error {
	if c.queuedPromoting == len(c.promoting) && c.queuedStaging == len(c.staging) {
		return nil
	}
	if err := c.stream(ctx); err != nil {
		return err
	}
	var err error
	for _, move := range c.promoting[c.queuedPromoting:] {
		bank := move.key.bank
		if err = c.mem.copyToCard(ctx, c.copies, move.card, move.slot.host[:bank.bytes]); err != nil {
			break
		}
		c.queuedPromoting++
	}
	if err != nil {
		for _, move := range c.promoting[c.queuedPromoting:] {
			move.key.bank.moving[move.key.index] = false
		}
		c.promoting = c.promoting[:c.queuedPromoting]
	}
	for _, move := range c.staging[c.queuedStaging:] {
		if err == nil {
			bank := move.key.bank
			if err = c.mem.copyToHost(ctx, c.copies, move.slot.host[:bank.bytes], move.card); err == nil {
				c.queuedStaging++
				continue
			}
		}
		move.key.bank.moving[move.key.index] = false
		move.key.bank.class.free = append(move.key.bank.class.free, move.slot)
	}
	c.staging = c.staging[:c.queuedStaging]
	if err != nil {
		return err
	}
	// Asking after the copies submits them, which the driver may otherwise hold until the next run's wait.
	_, err = c.mem.copied(ctx, c.copies, false)
	return err
}

// stream makes the stream the residency copies on, once.
func (c *expertResidency) stream(ctx context.Context) error {
	if c.copies != 0 {
		return nil
	}
	stream, err := c.mem.newStream(ctx)
	c.copies = stream
	return err
}

// refill moves experts in host memory to the card once expertRefillBytes
// beyond the headroom is free, up to free memory less the headroom and
// within the limit, bank by bank in one card allocation a bank. Memory
// another process takes first ends the refill.
func (c *expertResidency) refill(ctx context.Context) error {
	free, err := c.mem.available(ctx)
	if err != nil || free < expertFillHeadroomBytes+expertRefillBytes {
		return err
	}
	budget := min(free-expertFillHeadroomBytes, c.limit-min(c.limit, c.bytes))
	landed := false
	for _, bank := range c.ordered {
		if budget < bank.stride {
			continue
		}
		if !landed {
			// A promotion in flight still names its expert's host slot: the copies land before host experts are counted.
			if err := c.land(ctx, true); err != nil {
				return err
			}
			landed = true
		}
		var count uint64
		for _, host := range bank.hosted {
			if host != nil {
				count++
			}
		}
		if count = min(count, budget/bank.stride); count == 0 {
			continue
		}
		card, err := c.mem.alloc(ctx, count*bank.stride)
		if driver.IsOutOfMemory(err) {
			return nil
		}
		if err != nil {
			return err
		}
		budget -= count * bank.stride
		c.cards, c.bytes = append(c.cards, expertChunk{card, count * bank.stride, bank.stride}), c.bytes+count*bank.stride
		for index := 0; count > 0; index++ {
			if bank.hosted[index] != nil {
				if err := c.move(ctx, expertKey{bank, uint32(index)}, card); err != nil {
					return err
				}
				card += driver.DevicePtr(bank.stride)
				count--
			}
		}
	}
	return nil
}

// promote moves a host expert the routes named to the card. It takes the
// card slot of the coldest staged expert the routes did not name this step,
// which goes to its staged copy at once, and is copied there beside the
// next run, staying in host memory until the copy lands. Where none is
// staged it takes the slot of the coldest card expert at once, copying that
// expert back first; it stays in host memory when every card expert of its
// stride was routed this step.
func (c *expertResidency) promote(ctx context.Context, key expertKey) error {
	bank := key.bank
	class := c.classes[bank.stride]
	if err := ctx.Err(); err != nil {
		return err
	}
	for entry, seen := class.order.Front(), 0; entry != nil && seen < expertVictimSearch; entry, seen = entry.Next(), seen+1 {
		resident := entry.Value.(*residentExpert)
		if resident.step == c.step {
			break
		}
		if resident.key.bank.staged[resident.key.index].host == nil {
			continue
		}
		// The copy starts once the next run is launched (start); the run reads the victim's staged copy and the
		// promoted expert where it is.
		card := resident.key.bank.pointers[resident.key.index]
		c.promoting = append(c.promoting, expertMove{key, card, expertSlot{bank.pointers[key.index], bank.hosted[key.index]}})
		bank.moving[key.index] = true
		_, err := c.demote(ctx, entry)
		return err
	}
	victim := class.order.Front()
	if victim == nil || victim.Value.(*residentExpert).step == c.step {
		return nil
	}
	card, err := c.demote(ctx, victim)
	if err != nil {
		return err
	}
	return c.move(ctx, key, card)
}

// move copies a host expert to its card slot, whose table entry it becomes;
// the host slot stays pinned for a later demotion.
func (c *expertResidency) move(ctx context.Context, key expertKey, card driver.DevicePtr) error {
	bank := key.bank
	class := c.classes[bank.stride]
	if err := c.mem.write(ctx, card, bank.hosted[key.index][:bank.bytes]); err != nil {
		return err
	}
	class.free = append(class.free, expertSlot{bank.pointers[key.index], bank.hosted[key.index]})
	bank.pointers[key.index], bank.hosted[key.index], bank.changed = card, nil, true
	bank.entries[key.index] = class.order.PushBack(&residentExpert{key: key, step: c.step})
	return nil
}

// demote moves a card expert to its staged copy, or else copies it to a
// pinned host slot, and returns the card slot it left.
func (c *expertResidency) demote(ctx context.Context, entry *list.Element) (driver.DevicePtr, error) {
	key := entry.Value.(*residentExpert).key
	bank := key.bank
	class := c.classes[bank.stride]
	card := bank.pointers[key.index]
	slot := bank.staged[key.index]
	if slot.host != nil {
		bank.unstage(key.index)
	} else {
		var err error
		if slot, err = c.hostSlot(ctx, class); err != nil {
			return 0, err
		}
		if err := c.mem.read(ctx, slot.host[:bank.bytes], card); err != nil {
			class.free = append(class.free, slot)
			return 0, err
		}
	}
	bank.pointers[key.index], bank.hosted[key.index], bank.changed = slot.device, slot.host, true
	class.order.Remove(entry)
	bank.entries[key.index] = nil
	return card, nil
}

// hostSlot takes a free pinned host slot of class, pinning
// expertHostChunkSlots more where none is free.
func (c *expertResidency) hostSlot(ctx context.Context, class *expertClass) (expertSlot, error) {
	if len(class.free) == 0 {
		host, mapped, err := c.mem.hostAlloc(ctx, expertHostChunkSlots*class.stride)
		if err != nil {
			return expertSlot{}, err
		}
		c.pinned = append(c.pinned, host)
		for offset := uint64(0); offset < uint64(len(host)); offset += class.stride {
			class.free = append(class.free, expertSlot{mapped + driver.DevicePtr(offset), host[offset : offset+class.stride]})
		}
	}
	slot := class.free[len(class.free)-1]
	class.free = class.free[:len(class.free)-1]
	return slot, nil
}

// publish writes the tables whose experts moved as one copy, from the first such table through the last: the tables
// lie in one allocation in the residency's order, and a decode step that wrote each in turn waited about 25 us a table
// (eleven a Flash-Next step).
func (c *expertResidency) publish(ctx context.Context) error {
	first, last := len(c.ordered), -1
	for index, bank := range c.ordered {
		if bank.changed {
			first, last = min(first, index), index
			if bank.head != nil {
				bank.head.onHostStale = true
			}
		}
	}
	if err := c.publishOnHost(ctx); err != nil || last < 0 {
		return err
	}
	from, through := c.ordered[first], c.ordered[last]
	if err := c.mem.write(ctx, from.table, driver.Bytes(c.pointers[from.first:through.first+len(through.pointers)])); err != nil {
		return err
	}
	for _, bank := range c.ordered[first : last+1] {
		bank.changed = false
	}
	return nil
}

// publishOnHost sets the card's flags of experts on host of each group whose banks moved an expert, as one copy from
// the first flag that changed through the last.
func (c *expertResidency) publishOnHost(ctx context.Context) error {
	first, end := len(c.onHost), 0
	for _, head := range c.ordered {
		if !head.onHostStale {
			continue
		}
		flags := c.onHost[head.routed : head.routed+uint64(len(head.pointers))]
		for index := range flags {
			onHost := byte(1)
			for _, bank := range head.group {
				if bank.hosted[index] == nil {
					onHost = 0
				}
			}
			if flags[index] != onHost {
				flags[index] = onHost
				first, end = min(first, int(head.routed)+index), max(end, int(head.routed)+index+1)
			}
		}
		head.onHostStale = false
	}
	if c.onHostUnwritten {
		first, end = 0, len(c.onHost)
	}
	if first >= end {
		return nil
	}
	// A failed write leaves the card's flags unknown, so the next publish writes them all.
	err := c.mem.write(ctx, c.onHostFlags+driver.DevicePtr(first), c.onHost[first:end])
	c.onHostUnwritten = err != nil
	return err
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
	// Copies in flight land first: a promotion may be copying into this allocation.
	if err := c.land(ctx, true); err != nil {
		return err
	}
	chunk := c.cards[len(c.cards)-1]
	class := c.classes[chunk.stride]
	for entry := class.order.Front(); entry != nil; {
		next := entry.Next()
		key := entry.Value.(*residentExpert).key
		if pointer := key.bank.pointers[key.index]; pointer >= chunk.pointer && uint64(pointer-chunk.pointer) < chunk.bytes {
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
	if c.copies != 0 {
		_, err := c.mem.copied(context.Background(), c.copies, true)
		errs = append(errs, err, c.mem.dropStream(c.copies))
		c.copies = 0
	}
	for _, chunk := range c.cards {
		errs = append(errs, c.mem.free(context.Background(), []driver.DevicePtr{chunk.pointer}))
	}
	for _, pointer := range []driver.DevicePtr{c.tables, c.onHostFlags} {
		if pointer != 0 {
			errs = append(errs, c.mem.free(context.Background(), []driver.DevicePtr{pointer}))
		}
	}
	for _, host := range c.pinned {
		errs = append(errs, c.mem.hostFree(host))
	}
	c.cards, c.tables, c.onHostFlags, c.pinned, c.routed, c.onHost, c.classes = nil, 0, 0, nil, nil, nil, nil
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
func (m stateMemory) newStream(ctx context.Context) (driver.Stream, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return m.state.Driver.StreamCreate(expertStreamNonBlocking)
}
func (m stateMemory) dropStream(stream driver.Stream) error {
	return m.state.Driver.StreamDestroy(stream)
}
func (m stateMemory) copyToCard(ctx context.Context, stream driver.Stream, destination driver.DevicePtr, source []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return m.state.Driver.MemcpyHtoDAsync(destination, source, stream)
}
func (m stateMemory) copyToHost(ctx context.Context, stream driver.Stream, destination []byte, source driver.DevicePtr) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return m.state.Driver.MemcpyDtoHAsync(destination, source, stream)
}
func (m stateMemory) copied(_ context.Context, stream driver.Stream, wait bool) (bool, error) {
	if wait {
		return true, m.state.Driver.StreamSynchronize(stream)
	}
	return m.state.Driver.StreamQuery(stream)
}
func (m stateMemory) available(context.Context) (uint64, error) {
	free, _, err := m.state.Driver.MemInfo()
	return free, err
}
