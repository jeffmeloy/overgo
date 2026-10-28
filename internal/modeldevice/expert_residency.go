package modeldevice

import (
	"bytes"
	"cmp"
	"container/list"
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
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
	// expertFillHeadroomBytes and expertSpareHeadroomBytes: the card memory a fill without an explicit limit, and a
	// decode's free slots and a refill (spare), leave free, which minimizes the experts held off the card against an
	// allocation outside the reservation hook failing; what the requests allocate through the hook (their K/V, buffers,
	// arenas, staging, and room for each graph the executor captures) the experts give up as it comes. Outside it are
	// the stores and the draft a runner opens at its first request and the libraries' workspaces: about 0.8 GB at load,
	// none after. At 2 GiB and 1 GiB the card held a gigabyte fewer experts and chats decoded 9.8% slower over three
	// alternating rounds; Strata keeps 700 MiB free at its fill (--vram-reserve-mib) and serves with 324 MiB. At 512 MiB a
	// round's refill left Flash-Next 460-634 MiB of its card unused, 200-280 experts in host memory, the reservations
	// having freed whole layers' allocations past their need.
	expertFillHeadroomBytes  = 1 << 30
	expertSpareHeadroomBytes = 256 << 20
	// A decode's miss moves to the card on its first route, the most used first, at most expertPromotionsPerAdvance an
	// advance, into the slot of the card expert of its stride routed least recently: a promotion pays where the uses it
	// saves outlast the expert it displaces by more than a copy's cost over a miss's, and a copy beside the run (about
	// 1.5 MB over the link) cost less than the misses it saved (the CPU's 50 us an expert, the card waiting): the bench's
	// routes replayed (tmp/derive-tuning/routesim) admitted at a first miss, the CPU computed 14.1 experts a window for
	// 9.9 copies, at a second 32.0 for 2.9 (the hindsight bound 9.5 for 8.3), and served, over three alternating rounds,
	// 4K decode rose 5.2%, the agent's prompt 6.1% and chat's 15.4%, no cell falling. expertUsageDecay and
	// expertUsageRounds are Strata's adaptive tier's (src/program/generate.cpp: --adapt-decay 0.7, --adapt-every 4): an
	// expert counts the decode runs that route it, the counts falling by expertUsageDecay every expertUsageRounds of them,
	// which orders the misses an advance admits and the experts a release keeps. The bound binds rarely (the replay at 64
	// and at 256 moved the same).
	expertUsageDecay           = 0.7
	expertUsageRounds          = 4
	expertPromotionsPerAdvance = 64
	// expertSpareSlots: the free card slots a stride class takes at once (spare), which trades an allocation's cost (an
	// allocation and Windows clearing it, about 18 ms a GB) against slots held empty until misses fill them; carved from
	// lent memory it costs nothing.
	expertSpareSlots = 64
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
// and the flags its routes mark from flag routed. The bank's experts lie in its
// blob's slots, offset bytes in; stride is the blob's slot.
type expertBank struct {
	info          gguf.TensorInfo
	bytes, stride uint64
	blob          *expertBlob
	offset        uint64
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
	// entries are the bank's card experts by index, nil for one in host memory; home holds a card expert's copy in
	// pinned host memory, which every expert keeps; moving marks an expert whose copy is in flight. An advance reads
	// them for every routed expert, where maps keyed by bank and index took about 0.5 ms of a Flash-Next window.
	entries []*list.Element
	home    []expertSlot
	moving  []bool
	// usage counts the decode runs that routed each expert, decayed (expertUsageDecay); decodeRuns counts them whole,
	// from which an expert profile is ranked.
	usage      []float32
	decodeRuns []uint64
	// class keeps the card experts of the bank's stride.
	class *expertClass
}

// expertBlob is the banks whose experts lie together: an expert's bytes of each bank back to back in the banks' order,
// each part at its bank's offset, as Strata's expert blob holds an expert's gate, up and down rows (iq_kernels.hpp,
// NativeExpertLayout), so one card slot and one host slot hold the expert of every bank and it moves as one. A layer's
// banks ("blk.N.") form a blob; any other bank is a blob alone. The blob's first bank keeps its experts' entries in the
// card order and their slots, from its offset 0 to the slot's end.
type expertBlob struct {
	banks         []*expertBank
	bytes, stride uint64
	// layer is the routed layer the banks name ("blk.N."), -1 for a bank alone.
	layer int32
	// rank is each expert's place in the profile a load ranked the card by, the most routed first, past every ranked
	// place for one it does not rank; nil without a profile.
	rank []uint32
}

// head is the bank whose slots and entries stand for the blob's.
func (b *expertBlob) head() *expertBank { return b.banks[0] }

// expertBlobParts are a layer's banks in a blob in the order Strata's blob holds them: the gate rows (or the fused gate
// and up rows), the up rows, then the down rows.
var expertBlobParts = [][]string{{"ffn_gate_exps", "ffn_gate_up_exps"}, {"ffn_up_exps"}, {"ffn_down_exps"}}

// expertBlobOrder is a bank's place in its blob: its part's, any other bank after them.
func expertBlobOrder(name string) int {
	for order, parts := range expertBlobParts {
		for _, part := range parts {
			if strings.Contains(name, "."+part+".") {
				return order
			}
		}
	}
	return len(expertBlobParts)
}

// newExpertBlobs groups the banks into blobs and lays each blob's expert out: its banks in expertBlobOrder, each part
// at a slot-aligned offset after the one before, so a part starts where its own slot would.
func newExpertBlobs(ordered []*expertBank) ([]*expertBlob, error) {
	byLayer := map[uint32]*expertBlob{}
	var blobs []*expertBlob
	for _, bank := range ordered {
		layer, err := expertBankLayer(bank)
		blob := byLayer[layer]
		if err != nil || blob == nil {
			blob = &expertBlob{layer: -1}
			blobs = append(blobs, blob)
			if err == nil {
				blob.layer, byLayer[layer] = int32(layer), blob
			}
		}
		if len(blob.banks) > 0 && len(blob.head().pointers) != len(bank.pointers) {
			return nil, fmt.Errorf("expert banks %q and %q of one layer hold different expert counts", blob.head().info.Name, bank.info.Name)
		}
		blob.banks = append(blob.banks, bank)
	}
	for _, blob := range blobs {
		slices.SortStableFunc(blob.banks, func(a, b *expertBank) int {
			return cmp.Or(cmp.Compare(expertBlobOrder(a.info.Name), expertBlobOrder(b.info.Name)), cmp.Compare(a.info.Name, b.info.Name))
		})
		for _, bank := range blob.banks {
			offset, ok := checked.Align(blob.bytes, expertSlotAlignment)
			if !ok {
				return nil, fmt.Errorf("expert blob of %q overflows", bank.info.Name)
			}
			bank.blob, bank.offset, blob.bytes = blob, offset, offset+bank.bytes
		}
		stride, ok := checked.Align(blob.bytes, expertSlotAlignment)
		if !ok {
			return nil, fmt.Errorf("expert blob of %q overflows", blob.head().info.Name)
		}
		blob.stride = stride
		for _, bank := range blob.banks {
			bank.stride = stride
		}
	}
	return blobs, nil
}

// cardExperts counts the experts on the card.
func (c *expertResidency) cardExperts() uint64 {
	var experts uint64
	for _, class := range c.classes {
		experts += uint64(class.order.Len())
	}
	return experts
}

// expertClass keeps the card experts of one stride, least recently routed first.
type expertClass struct {
	stride uint64
	order  list.List
	// free are card slots of the stride no expert holds, which a promotion takes before a victim's (spare).
	free []driver.DevicePtr
}

// expertSlot is pinned host memory that holds one expert, at device on the card.
type expertSlot struct {
	device driver.DevicePtr
	host   []byte
}

// expertMove is a promotion's copy in flight: an expert to its card slot card from its host slot.
type expertMove struct {
	key  expertKey
	card driver.DevicePtr
	slot expertSlot
}

// expertChunk is one card allocation of slots of one stride.
type expertChunk struct {
	pointer       driver.DevicePtr
	bytes, stride uint64
	// lent is the loan the chunk is carved from, which its owner frees; nil for the residency's own allocation.
	lent *expertLoan
}

// expertLoan is card memory another owner lends the residency between its own uses (LendExpertMemory): spare slots are
// carved from it before the residency allocates, from its end down, and its owner takes back the part it uses next
// from its start (ReclaimExpertMemory), so memory a prompt's read borrows changes owner without a free, an allocation
// or Windows clearing it (about 18 ms a GB), and a short read takes back only the little it uses.
//   - floor is the start of the part lent, its owner using the memory below it; top is the lowest slot carved
type expertLoan struct {
	pointer           driver.DevicePtr
	bytes, floor, top uint64
}

// expertMemory is the device surface a residency needs beyond a load.
type expertMemory interface {
	deviceMemory
	hostAlloc(ctx context.Context, size uint64) ([]byte, driver.DevicePtr, error)
	hostFree(data []byte) error
	available(ctx context.Context) (uint64, error)
	// newStream makes a stream that copies beside the runs, and dropStream ends it.
	newStream(ctx context.Context) (driver.Stream, error)
	dropStream(stream driver.Stream) error
	// copyToCard queues a copy on stream of pinned host bytes to the card; copied reports whether every copy queued
	// on stream has landed, waiting for them where wait is set.
	copyToCard(ctx context.Context, stream driver.Stream, destination driver.DevicePtr, source []byte) error
	copied(ctx context.Context, stream driver.Stream, wait bool) (bool, error)
	// copyOnCard copies bytes between two places on the card, synchronously.
	copyOnCard(ctx context.Context, destination, source driver.DevicePtr, bytes uint64) error
	// writeOrdered queues a write of data behind the work the compute stream holds, data copied before it returns, so
	// the next run reads it without the host waiting for the run in flight: a table published while a decode window's
	// draft ran waited about 1 ms a window for the draft to end.
	writeOrdered(ctx context.Context, destination driver.DevicePtr, data []byte) error
}

// expertResidency keeps every expert of the file-backed banks readable by
// the card: a card copy, or pinned host memory the card maps. Every expert's
// bytes stay in pinned host memory, as Strata's expert arena keeps them, and
// the card holds copies: its share is filled once under the limit, by the
// load pass that reads the file; between runs the experts the last run routed
// to host memory take the card slots of experts it did not route, which read
// their host copies at once.
// routed is pinned host memory the card maps at routedMap, one byte an expert, which the route kernel sets and the advance after the run reads and clears.
// onHost is the card's flags at onHostFlags, laid out as routed, a routed node's first bank's byte set where every
// bank of the node holds the expert in host memory; the CPU computes those experts of a decode step (ExpertsOnHost).
type expertResidency struct {
	state   *device.State
	mem     expertMemory
	banks   map[string]*expertBank
	ordered []*expertBank
	blobs   []*expertBlob
	classes map[uint64]*expertClass
	// resident is the card's table of each routed layer's experts, a layer's experts in index order, 0 where the card
	// holds the expert and -1 where host memory does (Strata's d_res, which its verify window's doorbell reads), and
	// residentHost its host copy, published with the tables; zero where the layers' banks hold different expert counts.
	resident        driver.DevicePtr
	residentHost    []int32
	residentExperts int
	residentStale   bool
	cards           []expertChunk
	// loans is the card memory other owners lend, and lentBytes the expert slots carved from it.
	loans     []*expertLoan
	lentBytes uint64
	// reclaimed is the experts moved to host memory to give their card memory to another owner - a prompt read's arena
	// or a loan taken back - since the last refill (RefillExperts) or decode advance.
	reclaimed                      []expertKey
	pinned                         [][]byte
	tables, routedMap, onHostFlags driver.DevicePtr
	// pointers is every bank's table on the host, in the order the card's tables lie, each bank's pointers a part.
	pointers          []driver.DevicePtr
	routed, onHost    []byte
	onHostUnwritten   bool
	bytes, tableBytes uint64
	step              uint64
	// decodes counts the decode advances, whose every expertUsageRounds-th decays the usage counts.
	decodes uint64
	// routes, cardRoutes and loadCard are ExpertPlacement's counts.
	routes, cardRoutes, loadCard uint64
	// limit bounds the card bytes a refill reaches: the load's limit, or none where the load read it from free memory.
	limit uint64
	// copies is the stream the residency copies on beside the runs, and promoting the promotions whose copies to the
	// card are in flight, moving their experts.
	copies    driver.Stream
	promoting []expertMove
	// queuedPromoting counts the leading moves of promoting whose copies are queued; an advance records the rest,
	// whose copies start once the next run is launched.
	queuedPromoting int
	// held stops promotion between runs: every expert stays where the load placed it, so runs of one prompt
	// compute the same experts in the same places and decode the same tokens, as a library validation's
	// recording and replay must.
	held bool
	// tiered leaves the experts' places to an adaptive tier (SwapExperts) and the prompt's loans and refills, as
	// Strata's cache changes by nothing else: every advance reads its routes as a prompt's, promoting no miss and
	// keeping the experts a reclaim moved for the next refill.
	tiered bool
}

// newExpertResidency allocates the tables, the route flags and every
// expert's slot for the file's banks, under the limit, which includes the
// tables and the card experts but no other weight store. The slots hold the
// experts once a load pass places them, and the tables name them once it
// publishes.
func newExpertResidency(ctx context.Context, file *gguf.File, infos []gguf.TensorInfo, limit uint64, mem expertMemory, profile *ExpertProfile) (*expertResidency, error) {
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
		bank := &expertBank{info: info, bytes: info.Size / experts, routed: flags, hosted: make([][]byte, experts),
			pointers: make([]driver.DevicePtr, experts), entries: make([]*list.Element, experts),
			home: make([]expertSlot, experts), moving: make([]bool, experts), usage: make([]float32, experts),
			decodeRuns: make([]uint64, experts)}
		c.banks[info.Name] = bank
		c.ordered = append(c.ordered, bank)
		c.tableBytes += experts * pointerBytes
		flags += experts
	}
	blobs, err := newExpertBlobs(c.ordered)
	if err != nil {
		return nil, err
	}
	c.blobs = blobs
	for _, blob := range blobs {
		if c.classes[blob.stride] == nil {
			c.classes[blob.stride] = &expertClass{stride: blob.stride}
		}
		for _, bank := range blob.banks {
			bank.class = c.classes[blob.stride]
		}
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
		last := first + len(bank.hosted)
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
		err = c.allocateResident(ctx)
	}
	if err == nil {
		err = c.allocate(ctx, limit, profile)
		c.loadCard = c.cardExperts()
	}
	if err != nil {
		return nil, errors.Join(err, c.close())
	}
	return c, nil
}

// allocate places experts on the card while the limit allows and every other expert in pinned host memory: one card
// allocation and one pinned allocation a bank. Without a profile it fills in index order across the banks, so each
// bank keeps the same share there; with one, each layer's top-ranked experts, its banks together.
//   - one card allocation a bank, not a few experts each: Flash-Next's experts in allocations of 32 (some 540) decoded
//     at a third of the rate, Windows' driver making each allocation resident for every submission, and lost 500 MB
//     to its 2 MiB rounding
func (c *expertResidency) allocate(ctx context.Context, limit uint64, profile *ExpertProfile) error {
	var marks map[*expertBlob][]bool
	if profile != nil {
		var err error
		if marks, err = c.rankedCardExperts(*profile, limit); err != nil {
			return err
		}
	} else {
		marks = c.indexCardExperts(limit)
	}
	for _, blob := range c.blobs {
		experts, onCard := uint64(len(blob.head().pointers)), uint64(0)
		for _, card := range marks[blob] {
			if card {
				onCard++
			}
		}
		var card, mapped driver.DevicePtr
		var host []byte
		var err error
		if onCard > 0 {
			if card, err = c.mem.alloc(ctx, onCard*blob.stride); err != nil {
				return err
			}
			c.cards = append(c.cards, expertChunk{card, onCard * blob.stride, blob.stride, nil})
		}
		// Every expert keeps its bytes in pinned host memory, as Strata's expert arena does: a card expert's copy there
		// is its home for good, so a demotion copies nothing back and the card is a cache of the host's experts.
		if host, mapped, err = c.mem.hostAlloc(ctx, experts*blob.stride); err != nil {
			return err
		}
		c.pinned = append(c.pinned, host)
		var cardSlot uint64
		for index := range experts {
			offset := uint64(index) * blob.stride
			slot := expertSlot{device: mapped + driver.DevicePtr(offset), host: host[offset : offset+blob.stride]}
			if marks[blob][index] {
				c.placeOnCard(blob, uint32(index), card+driver.DevicePtr(cardSlot*blob.stride), slot)
				cardSlot++
			} else {
				c.placeOnHost(blob, uint32(index), slot)
			}
		}
	}
	return nil
}

// allocateResident makes the card's residency table where every routed layer holds the same expert count: every expert
// in host memory until a placement names it, written at the next publish.
func (c *expertResidency) allocateResident(ctx context.Context) error {
	var layers int32
	experts := -1
	for _, blob := range c.blobs {
		if blob.layer < 0 {
			continue
		}
		if experts >= 0 && experts != len(blob.head().pointers) {
			return nil
		}
		experts, layers = len(blob.head().pointers), max(layers, blob.layer+1)
	}
	if experts <= 0 {
		return nil
	}
	c.residentHost, c.residentExperts = make([]int32, int(layers)*experts), experts
	for index := range c.residentHost {
		c.residentHost[index] = -1
	}
	var err error
	c.resident, err = c.mem.alloc(ctx, uint64(len(c.residentHost))*uint64(unsafe.Sizeof(int32(0))))
	c.residentStale = true
	return err
}

// markResident sets a layer's expert's entry in the residency table's host copy.
func (c *expertResidency) markResident(blob *expertBlob, index uint32, onCard bool) {
	if c.residentHost == nil || blob.layer < 0 {
		return
	}
	value := int32(-1)
	if onCard {
		value = 0
	}
	c.residentHost[int(blob.layer)*c.residentExperts+int(index)], c.residentStale = value, true
}

// bankSlot is a bank's part of its blob's slot: from the bank's offset to the slot's end.
func bankSlot(bank *expertBank, slot expertSlot) expertSlot {
	return expertSlot{device: slot.device + driver.DevicePtr(bank.offset), host: slot.host[bank.offset:]}
}

// placeOnCard names a blob's expert on the card at card, its host copy home, every bank's table naming its part, and
// enters it last in its class's order.
func (c *expertResidency) placeOnCard(blob *expertBlob, index uint32, card driver.DevicePtr, home expertSlot) {
	entry := blob.head().class.order.PushBack(&residentExpert{key: expertKey{blob.head(), index}, step: c.step})
	for _, bank := range blob.banks {
		bank.pointers[index], bank.hosted[index], bank.changed = card+driver.DevicePtr(bank.offset), nil, true
		bank.home[index], bank.entries[index] = bankSlot(bank, home), entry
	}
	c.markResident(blob, index, true)
}

// placeOnHost names a blob's expert in host memory at slot, every bank's table naming its part.
func (c *expertResidency) placeOnHost(blob *expertBlob, index uint32, slot expertSlot) {
	for _, bank := range blob.banks {
		part := bankSlot(bank, slot)
		bank.pointers[index], bank.hosted[index], bank.changed = part.device, part.host, true
		bank.home[index], bank.entries[index] = expertSlot{}, nil
	}
	c.markResident(blob, index, false)
}

// setMoving marks a blob's expert whose copy is in flight, in every bank.
func (b *expertBlob) setMoving(index uint32, moving bool) {
	for _, bank := range b.banks {
		bank.moving[index] = moving
	}
}

// indexCardExperts marks experts in index order across the banks while the limit allows.
func (c *expertResidency) indexCardExperts(limit uint64) map[*expertBlob][]bool {
	marks := make(map[*expertBlob][]bool, len(c.blobs))
	for _, blob := range c.blobs {
		marks[blob] = make([]bool, len(blob.head().pointers))
	}
	most := slices.MaxFunc(c.blobs, func(left, right *expertBlob) int { return len(left.head().pointers) - len(right.head().pointers) })
	for index := range len(most.head().pointers) {
		for _, blob := range c.blobs {
			if index < len(blob.head().pointers) && (index == 0 || marks[blob][index-1]) && blob.stride <= limit-c.bytes {
				marks[blob][index] = true
				c.bytes += blob.stride
			}
		}
	}
	return marks
}

// place copies file bytes of a bank, offset bytes into it, to their experts'
// slots: into pinned host memory directly, and to the card through onDevice,
// one copy for each run of card slots the bytes fill back to back.
func (c *expertResidency) place(ctx context.Context, onDevice expertDevice, bank *expertBank, offset uint64, data []byte) error {
	var card driver.DevicePtr
	var pending []byte
	// end is where pending ends in data: a run grows only while both the card slots and the file bytes follow on,
	// since a ranked fill gives neighbouring slots experts the file holds apart.
	var end uint64
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
		if home := bank.home[index].host; home != nil {
			copy(home[within:], data[done:done+length])
		}
		if host := bank.hosted[index]; host != nil {
			copy(host[within:], data[done:done+length])
		} else if pointer := bank.pointers[index] + driver.DevicePtr(within); len(pending) != 0 && end == done && card+driver.DevicePtr(len(pending)) == pointer {
			pending, end = data[done-uint64(len(pending)):done+length], done+length
		} else {
			if err := flush(); err != nil {
				return err
			}
			card, pending, end = pointer, data[done:done+length], done+length
		}
		done += length
	}
	return flush()
}

// ExpertPlacement is where a residency's experts are and where its runs found the experts they routed.
//   - experts are blobs, a layer's banks' expert once; bytes are slot bytes, card memory and pinned host memory;
//     every expert keeps its pinned copy, a card expert's too, so HostExperts and HostBytes count them all and overlap
//     CardExperts and CardBytes
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

// HoldExperts keeps every expert where the load placed it: no promotion between runs, so runs of one prompt
// compute the same experts in the same places and decode the same tokens. It is set before the first run.
func (w *DeviceWeights) HoldExperts() {
	if w != nil && w.experts != nil {
		w.experts.held = true
	}
}

// TierExperts leaves the experts' places to an adaptive tier (SwapExperts) and the prompt's loans and refills
// (RefillExperts), as Strata's expert cache changes by nothing else: a run's routes promote no miss. A prompt read's
// short tail, a run of a decode's width, had otherwise promoted its misses and dropped the experts its reclaims had
// moved, which the refill then never returned (Flash-Next's 16K read left 943 of them on host). It is set before the
// first run.
func (w *DeviceWeights) TierExperts() {
	if w != nil && w.experts != nil {
		w.experts.tiered = true
	}
}

// ExpertMemory reports the card bytes of experts, which a reservation can release.
// Pointer tables remain resident and are excluded.
func (w *DeviceWeights) ExpertMemory(state *device.State) (uint64, error) {
	if w == nil || w.experts == nil || w.experts.state != state {
		return 0, errors.New("expert memory query requires its owning worker")
	}
	return w.experts.bytes - w.experts.tableBytes - w.experts.lentBytes, nil
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
	for _, blob := range c.blobs {
		placement.HostExperts += uint64(len(blob.head().pointers))
		placement.HostBytes += uint64(len(blob.head().pointers)) * blob.stride
	}
	return placement, nil
}

// ExpertRoutes returns the flags a routed node's routes mark, one byte a bank
// expert, in host memory the card maps, for the residency to read between
// runs: their address on the card and the host's view of them, which a run
// whose routes reach the host marks there. The nodes are the routed node's
// banks.
func (w *DeviceWeights) ExpertRoutes(state *device.State, nodes []*tensor.Tensor) (driver.DevicePtr, []byte, error) {
	if w == nil || w.experts == nil || w.experts.state != state || len(nodes) == 0 {
		return 0, nil, errors.New("expert routes require their owning CUDA worker")
	}
	head, err := w.experts.routedGroup(nodes)
	if err != nil {
		return 0, nil, err
	}
	flags := w.experts.routed[head.routed : head.routed+uint64(len(head.pointers))]
	return w.experts.routedMap + driver.DevicePtr(head.routed), flags, nil
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

// ExpertSwap is one of Strata's adaptive tier's swaps (src/program/generate.cpp's adapt): the routed layer's host
// expert In moves into the card slot of its card expert Out.
type ExpertSwap struct{ Layer, In, Out int }

// SwapExperts queues the adaptive tier's swaps: each Out goes to its host copy at once (the CPU computes it meanwhile),
// each In is copied into Out's card slot on the residency's stream and stays in host memory until its copy lands
// (LandExperts); the tables are published. A held residency swaps nothing, nor a swap whose In is not in host memory or
// whose Out is not on the card. It answers the swaps queued.
func (w *DeviceWeights) SwapExperts(ctx context.Context, state *device.State, swaps []ExpertSwap) (int, error) {
	if w == nil || w.experts == nil || w.experts.state != state {
		return 0, errors.New("expert residency requires its owning CUDA worker")
	}
	c := w.experts
	if c.held {
		return 0, nil
	}
	queued := 0
	for _, swap := range swaps {
		layer, found := w.ExpertBlobLayer(swap.Layer)
		if !found {
			return queued, fmt.Errorf("expert swap: no blob names layer %d", swap.Layer)
		}
		head := layer.blob.head()
		in, out := uint32(swap.In), uint32(swap.Out)
		if int(max(in, out)) >= len(head.pointers) || head.entries[in] != nil || head.moving[in] || head.entries[out] == nil ||
			head.moving[out] {
			continue
		}
		c.promoting = append(c.promoting, expertMove{expertKey{head, in}, head.pointers[out], expertSlot{head.pointers[in], head.hosted[in]}})
		layer.blob.setMoving(in, true)
		c.demote(head.entries[out])
		queued++
	}
	if err := c.start(ctx); err != nil {
		return queued, errors.Join(err, c.publish(context.WithoutCancel(ctx)))
	}
	return queued, c.publish(ctx)
}

// RefillExperts is the prompt path's refill after its loan (src/program/generate.cpp's refill): the experts moved to
// host memory for a prompt read's memory (ReserveExpertMemory, ReclaimExpertMemory) return to the card, into the memory
// lent back first, each copied on the residency's stream and staying in host memory until its copy lands
// (LandExperts); the tables are published. A held residency refills nothing. It answers the experts queued.
func (w *DeviceWeights) RefillExperts(ctx context.Context, state *device.State) (int, error) {
	if w == nil || w.experts == nil || w.experts.state != state {
		return 0, errors.New("expert residency requires its owning CUDA worker")
	}
	c := w.experts
	seen := make(map[expertKey]bool, len(c.reclaimed))
	keys := slices.DeleteFunc(c.reclaimed, func(key expertKey) bool {
		again := seen[key]
		seen[key] = true
		return again || key.bank.entries[key.index] != nil || key.bank.moving[key.index]
	})
	c.reclaimed = nil
	// The hottest first, as the room allows; the rest wait in host memory for the next refill's room. A refill that
	// dropped them kept every reservation's shortfall for good: Flash-Next's round held 16.0K experts through its agent
	// turns, of 17.2K at load.
	slices.SortStableFunc(keys, func(a, b expertKey) int { return c.colderKey(b, a) })
	if c.held || len(keys) == 0 {
		return 0, nil
	}
	if err := c.spare(ctx, keys); err != nil {
		return 0, errors.Join(err, c.publish(context.WithoutCancel(ctx)))
	}
	queued := 0
	for _, key := range keys {
		class := c.classes[key.bank.stride]
		if len(class.free) == 0 {
			c.reclaimed = append(c.reclaimed, key)
			continue
		}
		card := class.free[len(class.free)-1]
		class.free = class.free[:len(class.free)-1]
		c.promoting = append(c.promoting, expertMove{key, card, expertSlot{key.bank.pointers[key.index], key.bank.hosted[key.index]}})
		key.bank.blob.setMoving(key.index, true)
		queued++
	}
	if err := c.start(ctx); err != nil {
		return queued, errors.Join(err, c.publish(context.WithoutCancel(ctx)))
	}
	return queued, c.publish(ctx)
}

// LandExperts places the swapped experts on the card once their copies have landed, waited for, and publishes the
// tables (the adaptive tier's apply_pending).
func (w *DeviceWeights) LandExperts(ctx context.Context, state *device.State) error {
	if w == nil || w.experts == nil || w.experts.state != state {
		return errors.New("expert residency requires its owning CUDA worker")
	}
	return errors.Join(w.experts.land(ctx, true), w.experts.publish(ctx))
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
// The copies an earlier advance queued land first, waited for, and the decode's misses take the card slots of the
// coldest card experts.
//   - waited for because where an expert lies decides who computes it (the doorbell's CPU or the card), so a copy
//     that landed by one advance and not another changed a decoded text with the host's speed; a copy queued behind
//     the last run has landed by its end on Flash-Next, so the wait costs nothing there
func (c *expertResidency) advance(ctx context.Context, prompt bool) error {
	prompt = prompt || c.tiered
	if err := c.land(ctx, true); err != nil {
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
			first := len(missed)
			for _, bank := range head.group {
				c.routes++
				if !prompt {
					bank.usage[index]++
					bank.decodeRuns[index]++
				}
				if entry := bank.entries[index]; entry != nil {
					c.cardRoutes++
					entry.Value.(*residentExpert).step = c.step
					bank.class.order.MoveToBack(entry)
				} else if key := (expertKey{bank.blob.head(), uint32(index)}); !prompt && !bank.moving[index] && !slices.Contains(missed[first:], key) {
					// A blob's expert misses once, however many of its banks the node names.
					missed = append(missed, key)
				}
			}
		}
		clear(flags)
	}
	if !prompt {
		// A decode's own misses refill the memory a prompt's reclaims emptied (spare).
		c.reclaimed = nil
		if c.decodes++; c.decodes%expertUsageRounds == 0 {
			for _, bank := range c.ordered {
				for index := range bank.usage {
					bank.usage[index] *= expertUsageDecay
				}
			}
		}
	}
	if c.held {
		missed = nil
	}
	missed = admitMisses(missed)
	if err := c.spare(ctx, missed); err != nil {
		return errors.Join(err, c.publish(context.WithoutCancel(ctx)))
	}
	for _, key := range missed {
		if err := c.promote(ctx, key); err != nil {
			// The tables still name where each expert's bytes are.
			return errors.Join(err, c.publish(context.WithoutCancel(ctx)))
		}
	}
	return c.publish(ctx)
}

// admitMisses orders an advance's admitted misses the most used first, in their order among equals, and keeps as
// many as an advance moves.
func admitMisses(missed []expertKey) []expertKey {
	slices.SortStableFunc(missed, func(a, b expertKey) int { return cmp.Compare(b.bank.usage[b.index], a.bank.usage[a.index]) })
	return missed[:min(len(missed), expertPromotionsPerAdvance)]
}

// land applies the promotions the residency queued once their copies have
// landed, waiting for them where wait is set: a promoted expert takes its card
// slot and keeps its host slot as its copy.
func (c *expertResidency) land(ctx context.Context, wait bool) error {
	// Moves no launched run started copying start here, before any lands.
	if err := c.start(ctx); err != nil {
		return err
	}
	if len(c.promoting) == 0 {
		return nil
	}
	done, err := c.mem.copied(ctx, c.copies, wait)
	if err != nil || !done {
		return err
	}
	for _, move := range c.promoting {
		blob, index := move.key.bank.blob, move.key.index
		// The promoted expert's host slot stays its copy, so a later demotion copies nothing back.
		c.placeOnCard(blob, index, move.card, move.slot)
		blob.setMoving(index, false)
	}
	c.promoting, c.queuedPromoting = c.promoting[:0], 0
	return nil
}

// start queues the copies of the promotions an advance recorded, on the residency's stream: an advance runs before the
// run it prepares, whose own small copies to the card would otherwise wait behind them on the copy engine (Flash-Next
// decode steps waited about 0.5 ms a step). A promotion whose copy is refused is dropped, its expert staying in host
// memory.
func (c *expertResidency) start(ctx context.Context) error {
	if c.queuedPromoting == len(c.promoting) {
		return nil
	}
	if err := c.stream(ctx); err != nil {
		return err
	}
	var err error
	for _, move := range c.promoting[c.queuedPromoting:] {
		if err = c.mem.copyToCard(ctx, c.copies, move.card, move.slot.host[:move.key.bank.blob.bytes]); err != nil {
			break
		}
		c.queuedPromoting++
	}
	if err != nil {
		for _, move := range c.promoting[c.queuedPromoting:] {
			move.key.bank.blob.setMoving(move.key.index, false)
		}
		c.promoting = c.promoting[:c.queuedPromoting]
	}
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

// spare gives the stride classes of an advance's admitted misses free card slots, carved from the memory other owners
// lend first (LendExpertMemory) and then allocated from the card memory free beyond expertSpareHeadroomBytes, within
// the limit, so those misses move to the card without a victim: memory a prompt's read lends between reads, or that
// another process freed, fills with the experts the decode routes, at the promotions' rate and with their copies. A
// refill that moved the first host experts of each bank instead, 128 MiB an advance, cost the decode of a 16K-token
// read 59.5 to 46.0 tok/s, its copies beside each run on the link the run's own copies cross.
func (c *expertResidency) spare(ctx context.Context, missed []expertKey) error {
	wanted := make(map[*expertClass]uint64)
	for _, key := range missed {
		wanted[c.classes[key.bank.stride]]++
	}
	// The classes short of free slots, the narrowest stride first.
	var short []*expertClass
	for class, count := range wanted {
		if count > uint64(len(class.free)) {
			short = append(short, class)
		}
	}
	slices.SortFunc(short, func(a, b *expertClass) int { return cmp.Compare(a.stride, b.stride) })
	add := func(class *expertClass, chunk expertChunk) {
		c.cards, c.bytes = append(c.cards, chunk), c.bytes+chunk.bytes
		for slot := range chunk.bytes / class.stride {
			class.free = append(class.free, chunk.pointer+driver.DevicePtr(slot*class.stride))
		}
	}
	var free uint64
	read := false
	for _, class := range short {
		// A class takes expertSpareSlots at once, which the advances after it fill before taking more.
		need := min(max(wanted[class]-uint64(len(class.free)), expertSpareSlots), (c.limit-min(c.limit, c.bytes))/class.stride)
		for _, loan := range c.loans {
			slots := min(need, (loan.top-min(loan.top, loan.floor))/class.stride)
			if slots == 0 {
				continue
			}
			loan.top -= slots * class.stride
			add(class, expertChunk{loan.pointer + driver.DevicePtr(loan.top), slots * class.stride, class.stride, loan})
			c.lentBytes, need = c.lentBytes+slots*class.stride, need-slots
		}
		if need == 0 {
			continue
		}
		if !read {
			available, err := c.mem.available(ctx)
			if err != nil {
				return err
			}
			free, read = available-min(available, expertSpareHeadroomBytes), true
		}
		slots := min(need, free/class.stride)
		if slots == 0 {
			continue
		}
		card, err := c.mem.alloc(ctx, slots*class.stride)
		if driver.IsOutOfMemory(err) {
			return nil
		}
		if err != nil {
			return err
		}
		free -= slots * class.stride
		add(class, expertChunk{card, slots * class.stride, class.stride, nil})
	}
	return nil
}

// promote moves a host expert the routes named to the card. It takes the card
// slot of the coldest card expert the routes did not name this step, which
// reads its host copy at once, and is copied there beside the next run,
// staying in host memory until the copy lands. Where the routes named every
// card expert of its stride this step it stays in host memory.
func (c *expertResidency) promote(ctx context.Context, key expertKey) error {
	bank := key.bank
	if err := ctx.Err(); err != nil {
		return err
	}
	// A free slot of the stride takes the expert before any card expert's.
	if class := c.classes[bank.stride]; len(class.free) > 0 {
		card := class.free[len(class.free)-1]
		class.free = class.free[:len(class.free)-1]
		c.promoting = append(c.promoting, expertMove{key, card, expertSlot{bank.pointers[key.index], bank.hosted[key.index]}})
		bank.blob.setMoving(key.index, true)
		return nil
	}
	victim := c.classes[bank.stride].order.Front()
	if victim == nil || victim.Value.(*residentExpert).step == c.step {
		return nil
	}
	// The copy starts once the next run is launched (start); the run reads the victim's host copy and the promoted
	// expert where it is.
	resident := victim.Value.(*residentExpert)
	card := resident.key.bank.pointers[resident.key.index]
	c.promoting = append(c.promoting, expertMove{key, card, expertSlot{bank.pointers[key.index], bank.hosted[key.index]}})
	bank.blob.setMoving(key.index, true)
	c.demote(victim)
	return nil
}

// demote moves a card expert to its host copy, which the table names at once, and frees nothing: the caller keeps or
// frees the card slot it left.
func (c *expertResidency) demote(entry *list.Element) {
	key := entry.Value.(*residentExpert).key
	bank := key.bank
	c.classes[bank.stride].order.Remove(entry)
	c.placeOnHost(bank.blob, key.index, bank.home[key.index])
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
	if err := c.mem.writeOrdered(ctx, from.table, driver.Bytes(c.pointers[from.first:through.first+len(through.pointers)])); err != nil {
		return err
	}
	for _, bank := range c.ordered[first : last+1] {
		bank.changed = false
	}
	if c.residentStale {
		if err := c.mem.writeOrdered(ctx, c.resident, unsafe.Slice((*byte)(unsafe.Pointer(&c.residentHost[0])),
			len(c.residentHost)*int(unsafe.Sizeof(c.residentHost[0])))); err != nil {
			return err
		}
		c.residentStale = false
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
	err := c.mem.writeOrdered(ctx, c.onHostFlags+driver.DevicePtr(first), c.onHost[first:end])
	c.onHostUnwritten = err != nil
	return err
}

// ReserveExpertMemory makes room for another executor owner before allocation.
// It frees the card allocations whose freeing moves the coldest experts to
// host memory (release) until the request fits.
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
	for free < required && w.experts.owned() > 0 {
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

// release frees one of the residency's own card allocations for another owner (drop): the one whose freeing moves the
// coldest experts to host memory, as Strata's cache gives up its lowest-ranked slots. Freeing a chunk takes its slots
// from its stride's class, whose coldest experts, as many as its free slots do not cover, go to host memory; so a
// reservation's loss follows the load profile's ranking over every layer of a class, where freeing the last allocation
// had emptied half of the slots of the layers that share its stride (Flash-Next's round took 594 experts of layers 4,
// 30, 46 and 47, whose misses the adaptive tier, swapping within a layer, could never take back). Among equals, the
// last allocated.
func (c *expertResidency) release(ctx context.Context) error {
	slots := make(map[uint64]uint64)
	for _, chunk := range c.cards {
		slots[chunk.stride] += chunk.bytes / chunk.stride
	}
	coldest := make(map[uint64][]*list.Element)
	// demoted answers the hottest expert freeing the chunk moves to host memory, or nil where its class's free slots
	// cover its own.
	demoted := func(chunk expertChunk) *list.Element {
		class := c.classes[chunk.stride]
		held, kept := uint64(class.order.Len()), slots[chunk.stride]-chunk.bytes/chunk.stride
		if held <= kept {
			return nil
		}
		ordered, sorted := coldest[chunk.stride]
		if !sorted {
			for entry := class.order.Front(); entry != nil; entry = entry.Next() {
				ordered = append(ordered, entry)
			}
			slices.SortStableFunc(ordered, c.colder)
			coldest[chunk.stride] = ordered
		}
		return ordered[held-kept-1]
	}
	chosen, found := expertChunk{}, false
	var cost *list.Element
	for at := len(c.cards) - 1; at >= 0; at-- {
		chunk := c.cards[at]
		if chunk.lent != nil {
			continue
		}
		hottest := demoted(chunk)
		if !found || cost != nil && (hottest == nil || c.colder(hottest, cost) < 0) {
			chosen, cost, found = chunk, hottest, true
		}
	}
	if !found {
		return nil
	}
	return c.drop(ctx, func(candidate expertChunk) bool { return candidate.pointer == chosen.pointer }, &c.reclaimed)
}

// colder orders two card experts the colder first: the less routed, and where the routes tie, as before any decode,
// the lower in the profile the load ranked by, as Strata's cache gives up its lowest-ranked slots first.
func (c *expertResidency) colder(a, b *list.Element) int {
	return c.colderKey(a.Value.(*residentExpert).key, b.Value.(*residentExpert).key)
}

// colderKey is colder for two experts wherever they lie.
func (c *expertResidency) colderKey(x, y expertKey) int {
	if order := cmp.Compare(x.bank.usage[x.index], y.bank.usage[y.index]); order != 0 || x.bank.blob.rank == nil || y.bank.blob.rank == nil {
		return order
	}
	return cmp.Compare(y.bank.blob.rank[y.index], x.bank.blob.rank[x.index])
}

// owned counts the card allocations the residency made itself, which a release frees.
func (c *expertResidency) owned() int {
	count := 0
	for _, chunk := range c.cards {
		if chunk.lent == nil {
			count++
		}
	}
	return count
}

// drop moves the experts out of the chunks leaves names, moving the coldest experts of each stride to host memory
// rather than the chunks' own (Strata lends its cache's coldest slots, never a layer's): each expert a chunk holds,
// the most used first, takes a free slot of its stride elsewhere, then the card slot of the coldest expert of its
// stride elsewhere where that one is colder - used less, or where the uses tie ranked lower by the load's profile -
// copied on the card, and the expert it displaces and the rest of the chunks' go to host memory. A long prompt that
// freed whole banks of the last layers had left those layers' every expert on host for the decode after it, as had a
// round's first request, before any decode had used an expert, its buffers taking Flash-Next's layer 46. It publishes the tables once, frees the chunks the residency allocated and
// gives the lent ones back to their loans. The experts it moves to host memory are added to moved where it is set.
func (c *expertResidency) drop(ctx context.Context, leaves func(expertChunk) bool, moved *[]expertKey) error {
	if !slices.ContainsFunc(c.cards, leaves) {
		return nil
	}
	// Copies in flight land first: a promotion may be copying into these chunks.
	if err := c.land(ctx, true); err != nil {
		return err
	}
	var leaving, kept []expertChunk
	strides := make(map[uint64]bool)
	for _, chunk := range c.cards {
		if leaves(chunk) {
			leaving, strides[chunk.stride] = append(leaving, chunk), true
		} else {
			kept = append(kept, chunk)
		}
	}
	inSlots := func(stride uint64, pointer driver.DevicePtr) bool {
		for _, chunk := range leaving {
			if chunk.stride == stride && pointer >= chunk.pointer && uint64(pointer-chunk.pointer) < chunk.bytes {
				return true
			}
		}
		return false
	}
	colder := c.colder
	out := func(entry *list.Element) {
		if moved != nil {
			*moved = append(*moved, entry.Value.(*residentExpert).key)
		}
		c.demote(entry)
	}
	for stride := range strides {
		class := c.classes[stride]
		class.free = slices.DeleteFunc(class.free, func(pointer driver.DevicePtr) bool { return inSlots(stride, pointer) })
		var held, others []*list.Element
		for entry := class.order.Front(); entry != nil; entry = entry.Next() {
			key := entry.Value.(*residentExpert).key
			if inSlots(stride, key.bank.pointers[key.index]) {
				held = append(held, entry)
			} else {
				others = append(others, entry)
			}
		}
		slices.SortStableFunc(held, func(a, b *list.Element) int { return colder(b, a) })
		slices.SortStableFunc(others, colder)
		victims := 0
		for _, entry := range held {
			key := entry.Value.(*residentExpert).key
			var slot driver.DevicePtr
			var victim *list.Element
			switch {
			case len(class.free) > 0:
				slot, class.free = class.free[len(class.free)-1], class.free[:len(class.free)-1]
			case victims < len(others) && colder(others[victims], entry) < 0:
				victim, victims = others[victims], victims+1
				cold := victim.Value.(*residentExpert).key
				slot = cold.bank.pointers[cold.index]
			default:
				out(entry)
				continue
			}
			if err := c.mem.copyOnCard(ctx, slot, key.bank.pointers[key.index], key.bank.blob.bytes); err != nil {
				return errors.Join(err, c.publish(context.WithoutCancel(ctx)))
			}
			if victim != nil {
				out(victim)
			}
			for _, bank := range key.bank.blob.banks {
				bank.pointers[key.index], bank.changed = slot+driver.DevicePtr(bank.offset), true
			}
		}
	}
	if err := c.publish(ctx); err != nil {
		return err
	}
	c.cards = kept
	var freed []driver.DevicePtr
	for _, chunk := range leaving {
		c.bytes -= chunk.bytes
		if chunk.lent != nil {
			c.lentBytes -= chunk.bytes
			continue
		}
		freed = append(freed, chunk.pointer)
	}
	if len(freed) == 0 {
		return nil
	}
	return c.mem.free(ctx, freed)
}

// LendExpertMemory lends the residency the bytes at pointer another owner holds between its own uses, as a prompt's
// read lends its arena and staging until the next read: the decode's promotions take their slots before any victim's.
// Memory already lent is lent again whole, its owner done with the part it took back.
func (w *DeviceWeights) LendExpertMemory(_ context.Context, state *device.State, pointer driver.DevicePtr, bytes uint64) error {
	if w == nil || w.experts == nil || w.experts.state != state {
		return errors.New("expert memory loan requires its owning worker")
	}
	if pointer == 0 {
		return errors.New("expert memory loan requires memory")
	}
	if at := slices.IndexFunc(w.experts.loans, func(loan *expertLoan) bool { return loan.pointer == pointer }); at >= 0 {
		if loan := w.experts.loans[at]; loan.bytes == bytes {
			loan.floor = 0
			return nil
		}
		return errors.New("expert memory loan changed its size while lent")
	}
	top := bytes / expertSlotAlignment * expertSlotAlignment
	w.experts.loans = append(w.experts.loans, &expertLoan{pointer: pointer, bytes: bytes, top: top})
	return nil
}

// ReclaimExpertMemory takes back the first bytes of the memory lent at pointer, the experts in them moved out as a
// release moves them (drop); taking back all of it ends the loan, and memory never lent is a no-op.
func (w *DeviceWeights) ReclaimExpertMemory(ctx context.Context, state *device.State, pointer driver.DevicePtr, bytes uint64) error {
	if w == nil || w.experts == nil || w.experts.state != state {
		return errors.New("expert memory reclaim requires its owning worker")
	}
	c := w.experts
	at := slices.IndexFunc(c.loans, func(loan *expertLoan) bool { return loan.pointer == pointer })
	if at < 0 {
		return nil
	}
	loan := c.loans[at]
	if bytes < loan.bytes && bytes <= loan.floor {
		return nil
	}
	end := loan.pointer + driver.DevicePtr(bytes)
	// A chunk the part taken back ends inside keeps its slots past the end, as Strata's prompt lends only the slots it
	// needs: a short read takes back the few slots it uses of however large a chunk a refill carved.
	c.cards = splitExpertChunks(c.cards, loan, end)
	// The runs in flight end before the experts move: a kernel may read the slot a hotter expert takes.
	if state != nil && slices.ContainsFunc(c.cards, func(chunk expertChunk) bool { return chunk.lent == loan && chunk.pointer < end }) {
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
	}
	if err := c.drop(ctx, func(chunk expertChunk) bool { return chunk.lent == loan && chunk.pointer < end }, &c.reclaimed); err != nil {
		return err
	}
	if bytes >= loan.bytes {
		c.loans = slices.Delete(c.loans, at, at+1)
		return nil
	}
	// The chunks carved lie from top to the end, those below the part taken back gone.
	loan.floor, loan.top = bytes, loan.bytes/expertSlotAlignment*expertSlotAlignment
	for _, chunk := range c.cards {
		if chunk.lent == loan {
			loan.top = min(loan.top, uint64(chunk.pointer-loan.pointer))
		}
	}
	return nil
}

// splitExpertChunks cuts each chunk carved from loan that end falls inside at its first slot boundary at or past end.
func splitExpertChunks(chunks []expertChunk, loan *expertLoan, end driver.DevicePtr) []expertChunk {
	var split []expertChunk
	for _, chunk := range chunks {
		if chunk.lent == loan && chunk.pointer < end {
			if cut := (uint64(end-chunk.pointer) + chunk.stride - 1) / chunk.stride * chunk.stride; cut < chunk.bytes {
				split = append(split, expertChunk{chunk.pointer, cut, chunk.stride, loan},
					expertChunk{chunk.pointer + driver.DevicePtr(cut), chunk.bytes - cut, chunk.stride, loan})
				continue
			}
		}
		split = append(split, chunk)
	}
	return split
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
		if chunk.lent == nil {
			errs = append(errs, c.mem.free(context.Background(), []driver.DevicePtr{chunk.pointer}))
		}
	}
	for _, pointer := range []driver.DevicePtr{c.tables, c.onHostFlags, c.resident} {
		if pointer != 0 {
			errs = append(errs, c.mem.free(context.Background(), []driver.DevicePtr{pointer}))
		}
	}
	for _, host := range c.pinned {
		errs = append(errs, c.mem.hostFree(host))
	}
	c.cards, c.tables, c.onHostFlags, c.pinned, c.routed, c.onHost, c.classes = nil, 0, 0, nil, nil, nil, nil
	c.resident, c.residentHost = 0, nil
	c.loans, c.lentBytes = nil, 0
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
func (m stateMemory) copied(_ context.Context, stream driver.Stream, wait bool) (bool, error) {
	// A query answers copies that have landed without a wait: a copy queued beside a run has landed by its end.
	done, err := m.state.Driver.StreamQuery(stream)
	if err != nil || done || !wait {
		return done, err
	}
	return true, m.state.Driver.StreamSynchronize(stream)
}
func (m stateMemory) writeOrdered(ctx context.Context, destination driver.DevicePtr, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return m.state.Driver.MemcpyHtoDAsync(destination, data, m.state.Stream)
}
func (m stateMemory) copyOnCard(ctx context.Context, destination, source driver.DevicePtr, bytes uint64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return m.state.Driver.MemcpyDtoD(destination, source, bytes)
}
func (m stateMemory) available(context.Context) (uint64, error) {
	free, _, err := m.state.Driver.MemInfo()
	return free, err
}
