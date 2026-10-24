package modeldevice

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// fakeExpertDevice adds pinned host memory to fakeDevice: a host allocation
// is a live allocation whose bytes the host slice shares, as the card reads
// mapped host memory.
type fakeExpertDevice struct {
	*fakeDevice
	pinned    map[*byte]driver.DevicePtr
	failHost  int // the host allocation that fails; 0 for none
	hostCalls int
	freeBytes uint64 // the card memory the fake reports free
	// queued are the copies a stream holds, which land when the residency asks after them unless hold is set.
	queued  []func() error
	hold    bool
	streams int
	// ordered counts the writes queued behind the compute stream's work (writeOrdered).
	ordered int
}

func (d *fakeExpertDevice) newStream(context.Context) (driver.Stream, error) {
	d.streams++
	return driver.Stream(d.streams), nil
}

func (d *fakeExpertDevice) dropStream(driver.Stream) error {
	if len(d.queued) != 0 {
		return errors.New("a stream ended with copies in flight")
	}
	return nil
}

func (d *fakeExpertDevice) copyToCard(ctx context.Context, _ driver.Stream, destination driver.DevicePtr, source []byte) error {
	d.queued = append(d.queued, func() error { return d.write(ctx, destination, source) })
	return nil
}

func (d *fakeExpertDevice) writeOrdered(ctx context.Context, destination driver.DevicePtr, data []byte) error {
	d.ordered++
	return d.write(ctx, destination, data)
}

func (d *fakeExpertDevice) copyOnCard(ctx context.Context, destination, source driver.DevicePtr, size uint64) error {
	data, err := d.at(source, size)
	if err != nil {
		return err
	}
	return d.write(ctx, destination, bytes.Clone(data))
}

func (d *fakeExpertDevice) copied(_ context.Context, _ driver.Stream, wait bool) (bool, error) {
	if d.hold && !wait {
		return false, nil
	}
	var errs []error
	for _, copy := range d.queued {
		errs = append(errs, copy())
	}
	d.queued = nil
	return true, errors.Join(errs...)
}

func (d *fakeExpertDevice) available(context.Context) (uint64, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.freeBytes, nil
}

func newFakeExpertDevice() *fakeExpertDevice {
	return &fakeExpertDevice{fakeDevice: newFakeDevice(), pinned: make(map[*byte]driver.DevicePtr)}
}

func (d *fakeExpertDevice) hostAlloc(_ context.Context, size uint64) ([]byte, driver.DevicePtr, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.hostCalls++
	if d.hostCalls == d.failHost {
		return nil, 0, errors.New("injected pinning failure")
	}
	pointer := d.next
	d.next += driver.DevicePtr(size) + 512
	data := make([]byte, size)
	d.live[pointer], d.pinned[&data[0]] = data, pointer
	return data, pointer, nil
}

func (d *fakeExpertDevice) hostFree(data []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	pointer, found := d.pinned[&data[0]]
	if !found {
		return errors.New("free of host memory that is not pinned")
	}
	delete(d.pinned, &data[0])
	delete(d.live, pointer)
	return nil
}

// onDevice runs residency work on the fake directly, where a worker would run
// it on its thread.
func (d *fakeExpertDevice) onDevice(_ context.Context, work func(*device.State, expertMemory) error) error {
	return work(nil, d)
}

// weights is a raw store on the fake.
func (d *fakeExpertDevice) weights() *DeviceWeights {
	weights := &DeviceWeights{deviceTensorStore: newDeviceTensorStore(nil, "device"), onDevice: d.onDevice}
	weights.mem = d
	return weights
}

// loadExperts fills a residency of every bank of the file through a load pass
// on the fake, verifying expect when given.
func (d *fakeExpertDevice) loadExperts(ctx context.Context, file *gguf.File, limit uint64, expect []artifact.ID) (*DeviceWeights, error) {
	weights := d.weights()
	loader := NewLoader(file)
	loader.Experts(weights, file.Tensors, limit)
	return weights, loader.Load(ctx, expect)
}

// at returns the bytes of a range inside one live allocation.
func (d *fakeExpertDevice) at(pointer driver.DevicePtr, size uint64) ([]byte, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for base, held := range d.live {
		if pointer >= base && uint64(pointer-base)+size <= uint64(len(held)) {
			return held[pointer-base : uint64(pointer-base)+size], nil
		}
	}
	return nil, fmt.Errorf("range of %d bytes at %#x leaves every allocation", size, uint64(pointer))
}

func TestFlashNextExpertResidency(t *testing.T) {
	// Gate and down experts take slots of two strides; a gate expert is
	// smaller than its slot, so the fill reads it alone.
	f := writeFixture(t, rand.New(rand.NewPCG(19, 31)), []fixtureTensor{
		{name: "gate", shape: []uint64{2, 2, 3}, kind: dtype.BF16},
		{name: "down", shape: []uint64{16, 16, 3}, kind: dtype.F32},
	}, 0, 0)
	file := f.parse(t)
	const (
		experts = 3
		// A pointer an expert in each bank's table; the route flags are pinned host memory.
		tableBytes             = 2 * experts * 8
		gateStride, downStride = 256, 1024
		// One expert of each bank fits on the card.
		oneEach = tableBytes + gateStride + downStride
	)
	names := []string{"gate", "down"}
	open := func(t *testing.T, limit uint64) (*expertResidency, *fakeExpertDevice) {
		t.Helper()
		mem := newFakeExpertDevice()
		weights, err := mem.loadExperts(t.Context(), file, limit, nil)
		if err != nil {
			t.Fatal(err)
		}
		cache := weights.experts
		t.Cleanup(func() {
			if err := cache.close(); err != nil {
				t.Error(err)
			}
			if mem.liveCount() != 0 {
				t.Error("expert allocations survived close")
			}
		})
		return cache, mem
	}
	// readable holds every table entry to its expert's own bytes and, unless
	// card is nil, the card to hold exactly the named experts.
	readable := func(t *testing.T, cache *expertResidency, mem *fakeExpertDevice, card map[string][]uint32) {
		t.Helper()
		for _, name := range names {
			want := card[name]
			if card == nil {
				want = onCard(cache, name)
			}
			bank := cache.banks[name]
			table, err := mem.at(bank.table, experts*8)
			if err != nil {
				t.Fatal(err)
			}
			for index := range uint32(experts) {
				pointer := driver.DevicePtr(binary.LittleEndian.Uint64(table[index*8:]))
				actual, err := mem.at(pointer, bank.bytes)
				start := uint64(index) * bank.bytes
				if err != nil || !bytes.Equal(actual, f.payloads[name][start:start+bank.bytes]) {
					t.Fatalf("%s expert %d does not read its bytes: %v", name, index, err)
				}
				resident := cache.banks[name].entries[index] != nil
				if resident != (bank.hosted[index] == nil) || resident != slices.Contains(want, index) {
					t.Fatalf("%s expert %d on the card %t, want the card to hold %v", name, index, resident, want)
				}
			}
		}
		if cache.bytes > cache.limit {
			t.Fatal("card bytes exceed the limit")
		}
	}
	nodes := func(t *testing.T) []*tensor.Tensor {
		builder := tensor.NewBuilder()
		var result []*tensor.Tensor
		for _, name := range names {
			bank := file.Tensors[map[string]int{"gate": 0, "down": 1}[name]]
			result = append(result, builder.Input(name, bank.Type, tensor.MustShape(bank.Shape[:bank.Dimensions]...)))
		}
		return result
	}
	// route sets the flags of experts the node routes, as the route kernel does in host memory the card maps.
	route := func(t *testing.T, cache *expertResidency, indices ...uint32) {
		t.Helper()
		bitmap, _, err := (&DeviceWeights{experts: cache}).ExpertRoutes(nil, nodes(t))
		if err != nil {
			t.Fatal(err)
		}
		for _, index := range indices {
			cache.routed[uint64(bitmap-cache.routedMap)+uint64(index)] = 1
		}
	}
	// advance applies a decode run's routes.
	advance := func(t *testing.T, cache *expertResidency) {
		t.Helper()
		if err := cache.advance(t.Context(), false); err != nil {
			t.Fatal(err)
		}
	}
	t.Run("fill-splits-card-and-host", func(t *testing.T) {
		cache, mem := open(t, oneEach)
		readable(t, cache, mem, map[string][]uint32{"gate": {0}, "down": {0}})
		// The tables, the card flags of experts on host, the pinned route flags, and one card allocation and one
		// pinned allocation a bank.
		if len(cache.cards) != 2 || len(cache.pinned) != 3 || mem.liveCount() != 7 {
			t.Fatalf("fill made %d card and %d pinned allocations, %d live", len(cache.cards), len(cache.pinned), mem.liveCount())
		}
		reclaimable, err := (&DeviceWeights{experts: cache}).ExpertMemory(nil)
		if err != nil || reclaimable != gateStride+downStride {
			t.Fatalf("reclaimable card bytes = %d: %v", reclaimable, err)
		}
		placement, err := (&DeviceWeights{experts: cache}).ExpertPlacement(nil)
		want := ExpertPlacement{CardExperts: 2, HostExperts: 2*experts - 2, CardBytes: gateStride + downStride, HostBytes: (experts - 1) * (gateStride + downStride), LoadCardExperts: 2}
		if err != nil || placement != want {
			t.Fatalf("placement = %+v, want %+v: %v", placement, want, err)
		}
		// Every expert keeps its bytes in pinned host memory, those on the card too.
		all, _ := open(t, tableBytes+experts*(gateStride+downStride))
		if len(all.pinned) != 3 || all.cardExperts() != 2*experts {
			t.Fatal("a limit that holds every expert left one in host memory or kept no host copies")
		}
	})
	t.Run("every-expert-keeps-its-host-copy", func(t *testing.T) {
		// Each expert's bytes lie in pinned host memory, a card expert's too, so releasing the card's allocations moves
		// every expert to its host copy with nothing pinned or copied back.
		cache, mem := open(t, tableBytes+2*(gateStride+downStride))
		for _, name := range names {
			bank := cache.banks[name]
			for index := range uint32(experts) {
				host := bank.hosted[index]
				if bank.entries[index] != nil {
					host = bank.home[index].host
				}
				start := uint64(index) * bank.bytes
				if len(host) < int(bank.bytes) || !bytes.Equal(host[:bank.bytes], f.payloads[name][start:start+bank.bytes]) {
					t.Fatalf("%s expert %d keeps no host copy of its bytes", name, index)
				}
			}
		}
		pinned := len(cache.pinned)
		for len(cache.cards) > 0 {
			if err := cache.release(t.Context()); err != nil {
				t.Fatal(err)
			}
		}
		readable(t, cache, mem, map[string][]uint32{})
		if len(cache.pinned) != pinned {
			t.Fatalf("releasing the card pinned %d more allocations", len(cache.pinned)-pinned)
		}
	})
	t.Run("routes-promote-between-runs", func(t *testing.T) {
		cache, mem := open(t, oneEach)
		writes := mem.writes
		advance(t, cache)
		if mem.writes != writes {
			t.Fatal("a run that routed nothing moved experts")
		}
		// A miss takes the slot of the coldest card expert, 0, which reads its host copy at once, and its own copy lands
		// by the advance after.
		pinned := len(cache.pinned)
		route(t, cache, 2)
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{})
		if bytes.IndexByte(cache.routed, 1) >= 0 || len(cache.promoting) != 2 {
			t.Fatalf("the routes stayed marked or %d experts are promoting, want expert 2 of each bank", len(cache.promoting))
		}
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{"gate": {2}, "down": {2}})
		// Expert 2 served this step too, so expert 0's miss finds no card expert the step left unnamed: it stays in host
		// memory.
		route(t, cache, 0, 2)
		advance(t, cache)
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{"gate": {2}, "down": {2}})
		// Expert 1's miss takes the slot of expert 2, unnamed this step.
		route(t, cache, 1)
		advance(t, cache)
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{"gate": {1}, "down": {1}})
		if len(cache.pinned) != pinned {
			t.Fatal("swaps pinned more host memory")
		}
		// Runs named expert 2, then 0 and 2, then 1, in both banks; only the second found its expert 2 on the card.
		placement, err := (&DeviceWeights{experts: cache}).ExpertPlacement(nil)
		if err != nil || placement.Routes != 8 || placement.CardRoutes != 2 || placement.CardExperts != 2 {
			t.Fatalf("placement after the runs = %+v: %v", placement, err)
		}
	})
	t.Run("experts-on-host-follow-moves", func(t *testing.T) {
		cache, mem := open(t, oneEach)
		flags, err := (&DeviceWeights{experts: cache}).ExpertsOnHost(nil, nodes(t))
		if err != nil {
			t.Fatal(err)
		}
		// onCard holds the card's flags to the experts every bank holds on the card, and no other.
		onCard := func(t *testing.T, card ...uint32) {
			t.Helper()
			got, err := mem.at(flags, experts)
			if err != nil {
				t.Fatal(err)
			}
			for index := range uint32(experts) {
				if want := !slices.Contains(card, index); (got[index] == 1) != want {
					t.Fatalf("expert %d on host %t, want %t", index, got[index] == 1, want)
				}
			}
		}
		// A node's flags are clear until the advance after it is named, so its first run reads every expert itself.
		onCard(t, 0, 1, 2)
		advance(t, cache)
		onCard(t, 0)
		// Expert 2's miss leaves 0 to its host copy while 2's copy is in flight.
		route(t, cache, 2)
		advance(t, cache)
		onCard(t)
		advance(t, cache)
		onCard(t, 2)
	})
	t.Run("promotions-land-beside-the-next-run", func(t *testing.T) {
		// Two experts of each bank fit: 0 and 1 on the card, 2 in host memory.
		cache, mem := open(t, tableBytes+2*(gateStride+downStride))
		// Expert 2's miss takes the slot of the coldest card expert the step left unnamed, 0, which reads its host copy
		// at once, with no copy back, and 2 stays in host memory while its copy to the card
		// is in flight.
		route(t, cache, 2)
		mem.hold = true
		writes, pinned := mem.writes, len(cache.pinned)
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{"gate": {1}, "down": {1}})
		if len(cache.promoting) != 2 || mem.writes-writes != 1 {
			t.Fatalf("%d promotions in flight after %d writes, want two promotions and one write of both banks' tables", len(cache.promoting), mem.writes-writes)
		}
		// The moves' copies wait for the run the advance prepared to be launched, behind its own copies.
		if len(mem.queued) != 0 {
			t.Fatalf("the advance queued %d copies before its run was launched", len(mem.queued))
		}
		if err := cache.start(t.Context()); err != nil || len(mem.queued) != len(cache.promoting) {
			t.Fatalf("starting the moves queued %d copies (%v), want one for each of %d moves", len(mem.queued), err, len(cache.promoting))
		}
		// The next advance waits for the copies still in flight, so where an expert lies never follows the host's
		// speed.
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{"gate": {1, 2}, "down": {1, 2}})
		if len(cache.pinned) != pinned || len(cache.promoting) != 0 {
			t.Fatalf("the promotions pinned %d more allocations and left %d in flight", len(cache.pinned)-pinned, len(cache.promoting))
		}
	})
	t.Run("prompt-routes-move-nothing", func(t *testing.T) {
		// A prompt's routes move no expert, however few experts they name.
		cache, mem := open(t, oneEach)
		writes := mem.writes
		route(t, cache, 1)
		if err := cache.advance(t.Context(), true); err != nil {
			t.Fatal(err)
		}
		readable(t, cache, mem, map[string][]uint32{"gate": {0}, "down": {0}})
		if mem.writes != writes || bytes.IndexByte(cache.routed, 1) >= 0 {
			t.Fatal("a prompt's routes moved experts or stayed marked")
		}
		// A decode run's routes move their experts though they name two of three, as a verify span may of a small bank:
		// at their first misses expert 1 takes expert 0's slot, and expert 2 finds no other.
		for range 3 {
			route(t, cache, 1, 2)
			advance(t, cache)
		}
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{"gate": {1}, "down": {1}})
	})
	t.Run("release-moves-a-card-allocation-to-host", func(t *testing.T) {
		cache, mem := open(t, oneEach)
		route(t, cache, 0)
		advance(t, cache)
		if err := cache.release(t.Context()); err != nil {
			t.Fatal(err)
		}
		if len(cache.cards) != 1 || cache.bytes != tableBytes+downStride && cache.bytes != tableBytes+gateStride {
			t.Fatalf("release left %d card allocations of %d bytes", len(cache.cards), cache.bytes)
		}
		remaining := cache.cards[0].stride
		card := map[string][]uint32{}
		for _, name := range names {
			if cache.banks[name].stride == remaining {
				card[name] = []uint32{0}
			}
		}
		readable(t, cache, mem, card)
		if err := cache.release(t.Context()); err != nil {
			t.Fatal(err)
		}
		readable(t, cache, mem, map[string][]uint32{})
		if cache.bytes != tableBytes {
			t.Fatal("releasing every card allocation left card bytes")
		}
		route(t, cache, 1)
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{})
		// A release while promotions are in flight lands them first: expert 2's copies target the allocations the
		// releases free, into expert 0's slots, and no table names a freed allocation, then or after the next advance.
		cache, mem = open(t, tableBytes+2*(gateStride+downStride))
		route(t, cache, 2)
		mem.hold = true
		advance(t, cache)
		if err := cache.start(t.Context()); err != nil || len(cache.promoting) != 2 {
			t.Fatalf("%d promotions in flight before the release (%v), want one a bank", len(cache.promoting), err)
		}
		for range 2 {
			if err := cache.release(t.Context()); err != nil {
				t.Fatal(err)
			}
		}
		mem.hold = false
		readable(t, cache, mem, map[string][]uint32{})
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{})
	})
	t.Run("failed-promotion-keeps-tables-readable", func(t *testing.T) {
		for _, fail := range []string{"write", "cancel"} {
			cache, mem := open(t, oneEach)
			ctx, cancel := context.WithCancel(t.Context())
			// The miss takes expert 0's slot.
			route(t, cache, 2)
			if fail == "write" {
				// The tables naming the moves.
				mem.failWrite = mem.writes + 1
			} else {
				cancel()
			}
			if err := cache.advance(ctx, false); err == nil {
				t.Fatalf("%s: a failed promotion was accepted", fail)
			}
			cancel()
			mem.failWrite = 0
			readable(t, cache, mem, nil)
		}
	})
	t.Run("failed-publication-is-repaired-before-the-next-run", func(t *testing.T) {
		// Expert 2's miss takes expert 0's slots, then the table write fails: the tables still name the card slots
		// 2's copies overwrite until a later advance writes them.
		cache, mem := open(t, oneEach)
		route(t, cache, 2)
		mem.failWrite = mem.writes + 1
		if err := cache.advance(t.Context(), false); err == nil {
			t.Fatal("a failed table write was accepted")
		}
		// The next run brings no routes; it may run only after the tables are repaired, past the two copies that land
		// first.
		mem.failWrite = mem.writes + 3
		if err := cache.advance(t.Context(), false); err == nil {
			t.Fatal("a run was admitted over tables a failed repair left stale")
		}
		mem.failWrite = 0
		advance(t, cache)
		readable(t, cache, mem, nil)
	})
	t.Run("failed-fill-releases-everything", func(t *testing.T) {
		for _, fail := range []string{"table", "card", "pinning", "copy", "cancel"} {
			mem := newFakeExpertDevice()
			ctx, cancel := context.WithCancel(t.Context())
			switch fail {
			case "table":
				mem.failAlloc = 1
			case "card":
				mem.failAlloc = 2
			case "pinning":
				mem.failHost = 1
			case "copy":
				mem.failWrite = 2
			case "cancel":
				mem.onWrite = cancel
			}
			weights, err := mem.loadExperts(ctx, file, oneEach, nil)
			if err == nil || weights.HasExpertResidency() {
				t.Fatalf("%s: a failed fill was accepted", fail)
			}
			cancel()
			if mem.liveCount() != 0 {
				t.Fatalf("%s: a failed fill left %d allocations", fail, mem.liveCount())
			}
		}
	})
	t.Run("descriptors", func(t *testing.T) {
		changed := append([]gguf.TensorInfo(nil), file.Tensors...)
		changed[0].Offset++
		mem := newFakeExpertDevice()
		if _, err := newExpertResidency(t.Context(), file, changed, oneEach, mem, nil); err == nil || mem.liveCount() != 0 {
			t.Fatal("foreign descriptor accepted or allocated")
		}
		if _, err := newExpertResidency(t.Context(), file, file.Tensors, tableBytes-1, mem, nil); err == nil || mem.liveCount() != 0 {
			t.Fatal("tables beyond the limit accepted or allocated")
		}
		cache, _ := open(t, oneEach)
		wrong := nodes(t)
		wrong[1] = tensor.NewBuilder().Input("down", dtype.F32, tensor.MustShape(16, 8, 6))
		if _, _, err := (&DeviceWeights{experts: cache}).ExpertRoutes(nil, wrong); err == nil {
			t.Fatal("a bank of another shape was accepted")
		}
	})
}

// TestExpertAdmissionTakesTheMostUsed: an advance moves its admitted misses the most used first, in routing order
// among equals, and no more than expertPromotionsPerAdvance; decode runs count an expert's usage, decayed every
// expertUsageRounds of them, and a prompt's count nothing.
func TestExpertAdmissionTakesTheMostUsed(t *testing.T) {
	const experts = 2 * expertPromotionsPerAdvance
	bank := &expertBank{usage: make([]float32, experts)}
	var missed []expertKey
	for index := range uint32(experts) {
		// Every third expert is used three times, the rest twice.
		bank.usage[index] = 2 + float32(index%3/2)
		missed = append(missed, expertKey{bank, index})
	}
	admitted := admitMisses(slices.Clone(missed))
	if len(admitted) != expertPromotionsPerAdvance {
		t.Fatalf("an advance admitted %d misses, want %d", len(admitted), expertPromotionsPerAdvance)
	}
	var most []uint32
	for index := range uint32(experts) {
		if index%3 == 2 {
			most = append(most, index)
		}
	}
	for at, key := range admitted[:len(most)] {
		if key.index != most[at] {
			t.Fatalf("admission %d took expert %d, want the most used in order, %v", at, key.index, most)
		}
	}
	f := writeFixture(t, rand.New(rand.NewPCG(5, 7)), []fixtureTensor{{name: "gate", shape: []uint64{2, 2, 3}, kind: dtype.BF16}}, 0, 0)
	file := f.parse(t)
	mem := newFakeExpertDevice()
	weights, err := mem.loadExperts(t.Context(), file, 3*8+3*256, nil)
	if err != nil {
		t.Fatal(err)
	}
	cache := weights.experts
	defer cache.close()
	node := tensor.NewBuilder().Input("gate", file.Tensors[0].Type, tensor.MustShape(file.Tensors[0].Shape[:file.Tensors[0].Dimensions]...))
	run := func(prompt bool) {
		t.Helper()
		flags, _, err := (&DeviceWeights{experts: cache}).ExpertRoutes(nil, []*tensor.Tensor{node})
		if err != nil {
			t.Fatal(err)
		}
		cache.routed[uint64(flags-cache.routedMap)] = 1
		if err := cache.advance(t.Context(), prompt); err != nil {
			t.Fatal(err)
		}
	}
	usage := cache.banks["gate"].usage
	run(true)
	if usage[0] != 0 {
		t.Fatalf("a prompt counted expert 0's use: %v", usage[0])
	}
	for range expertUsageRounds - 1 {
		run(false)
	}
	if usage[0] != expertUsageRounds-1 {
		t.Fatalf("%d decode runs counted %v", expertUsageRounds-1, usage[0])
	}
	run(false)
	if want := float32(expertUsageRounds) * expertUsageDecay; usage[0] != want {
		t.Fatalf("the %dth decode run left a count of %v, want %v", expertUsageRounds, usage[0], want)
	}
}

// TestExpertResidencyFillsTheCardWhenItFrees loads two banks of three experts while the card is held, so the load's
// limit, free memory less the headroom, holds one expert of each bank, then routes the other two as a decode does once
// the card frees.
//   - the placement reports the experts the load placed on the card
//   - an admitted miss moves into a free slot of memory freed beyond the headroom, one allocation a stride class, and
//     evicts no card expert: once its copies land every expert is on the card and every table entry reads its bytes
//   - with no memory free beyond the headroom, past the load's limit, or where the allocation fails, a miss takes a
//     card expert's slot as before, the card holding as many experts
func TestExpertResidencyFillsTheCardWhenItFrees(t *testing.T) {
	f := writeFixture(t, rand.New(rand.NewPCG(19, 31)), []fixtureTensor{
		{name: "gate", shape: []uint64{2, 2, 3}, kind: dtype.BF16},
		{name: "down", shape: []uint64{16, 16, 3}, kind: dtype.F32},
	}, 0, 0)
	file := f.parse(t)
	const (
		experts = 3
		// A pointer an expert in each bank's table.
		tableBytes             = 2 * experts * 8
		gateStride, downStride = 256, 1024
		oneEach                = tableBytes + gateStride + downStride
		// held is the card a load finds held, freed the card once the holder exits: a class's spare slots each.
		held  = expertFillHeadroomBytes + oneEach
		freed = expertSpareHeadroomBytes + expertSpareSlots*(gateStride+downStride)
	)
	builder := tensor.NewBuilder()
	var nodes []*tensor.Tensor
	for _, bank := range file.Tensors {
		nodes = append(nodes, builder.Input(bank.Name, bank.Type, tensor.MustShape(bank.Shape[:bank.Dimensions]...)))
	}
	open := func(t *testing.T, limit uint64) (*expertResidency, *fakeExpertDevice) {
		t.Helper()
		mem := newFakeExpertDevice()
		mem.freeBytes = held
		weights, err := mem.loadExperts(t.Context(), file, limit, nil)
		if err != nil {
			t.Fatal(err)
		}
		cache := weights.experts
		t.Cleanup(func() {
			if err := cache.close(); err != nil || mem.liveCount() != 0 {
				t.Errorf("close left %d allocations: %v", mem.liveCount(), err)
			}
		})
		return cache, mem
	}
	// admit runs three advances after decode runs that routed experts 1 and 2, the first admitting them and the second
	// landing their copies, and returns the experts on the card, holding every table entry to its expert's bytes.
	admit := func(t *testing.T, cache *expertResidency, mem *fakeExpertDevice) uint64 {
		t.Helper()
		flags, _, err := (&DeviceWeights{experts: cache}).ExpertRoutes(nil, nodes)
		if err != nil {
			t.Fatal(err)
		}
		for range 3 {
			for _, index := range []uint64{1, 2} {
				cache.routed[uint64(flags-cache.routedMap)+index] = 1
			}
			if err := cache.advance(t.Context(), false); err != nil {
				t.Fatal(err)
			}
		}
		for _, name := range []string{"gate", "down"} {
			bank := cache.banks[name]
			table, err := mem.at(bank.table, experts*8)
			if err != nil {
				t.Fatal(err)
			}
			for index := range uint64(experts) {
				actual, err := mem.at(driver.DevicePtr(binary.LittleEndian.Uint64(table[index*8:])), bank.bytes)
				if err != nil || !bytes.Equal(actual, f.payloads[name][index*bank.bytes:(index+1)*bank.bytes]) {
					t.Fatalf("%s expert %d does not read its bytes: %v", name, index, err)
				}
			}
		}
		placement, err := (&DeviceWeights{experts: cache}).ExpertPlacement(nil)
		if err != nil || placement.LoadCardExperts != 2 || placement.CardExperts+placement.HostExperts != 2*experts {
			t.Fatalf("placement %+v, want the load's two card experts of six: %v", placement, err)
		}
		return placement.CardExperts
	}
	cache, mem := open(t, 0)
	mem.freeBytes = freed
	allocs := mem.allocs
	if card := admit(t, cache, mem); card != 2*experts || mem.allocs != allocs+2 {
		t.Fatalf("the freed card holds %d experts after %d allocations, want every expert in one allocation a class", card, mem.allocs-allocs)
	}
	if got := onCard(cache, "gate"); !slices.Contains(got, 0) {
		t.Fatalf("a promotion into freed memory evicted the load's expert: the card holds %v", got)
	}
	full, mem := open(t, 0)
	mem.freeBytes = expertSpareHeadroomBytes
	if card := admit(t, full, mem); card != 2 {
		t.Fatalf("with no memory free beyond the headroom the card holds %d experts", card)
	}
	limited, mem := open(t, oneEach)
	mem.freeBytes = freed
	if card := admit(t, limited, mem); card != 2 {
		t.Fatalf("promotions took %d experts past the load's limit", card)
	}
	taken, mem := open(t, 0)
	mem.freeBytes, mem.failAlloc = freed, mem.allocs+1
	// The failed allocation fails no run: its misses take the load's experts' slots, and the next advance's allocation
	// takes the rest.
	if card := admit(t, taken, mem); card != 2*experts-2 {
		t.Fatalf("promotions after an allocation failed left %d experts on the card, want %d", card, 2*experts-2)
	}
	// Memory another owner lends takes the misses without an allocation or a victim, carved from its end down; its owner
	// takes back the part it uses next, the experts there moved out, the most used into the card slots of colder ones,
	// and then all of it, none of it freed.
	lending, mem := open(t, 0)
	mem.freeBytes = expertSpareHeadroomBytes
	const lentBytes = expertSpareSlots * (gateStride + downStride)
	loan, err := mem.alloc(t.Context(), lentBytes)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := mem.free(context.Background(), []driver.DevicePtr{loan}); err != nil {
			t.Error(err)
		}
	})
	lender := &DeviceWeights{experts: lending}
	if err := lender.LendExpertMemory(t.Context(), nil, loan, lentBytes); err != nil {
		t.Fatal(err)
	}
	allocs, live := mem.allocs, mem.liveCount()
	if card := admit(t, lending, mem); card != 2*experts || mem.allocs != allocs {
		t.Fatalf("lent memory holds %d experts after %d allocations, want every expert in it and none", card, mem.allocs-allocs)
	}
	// The down experts' slots, carved last, lie at the loan's start.
	for _, taken := range []struct {
		bytes uint64
		card  int
	}{{1, 2*experts - 2}, {lentBytes, 2}} {
		if err := lender.ReclaimExpertMemory(t.Context(), nil, loan, taken.bytes); err != nil {
			t.Fatal(err)
		}
		readsItsBytes(t, lending, mem, f.payloads)
		if card := len(onCard(lending, "gate")) + len(onCard(lending, "down")); card != taken.card || mem.liveCount() != live {
			t.Fatalf("taking back %d bytes left %d experts on the card and %d allocations of %d, want %d and none freed", taken.bytes, card, mem.liveCount(), live, taken.card)
		}
	}
	if len(lending.loans) != 0 || lending.lentBytes != 0 {
		t.Fatalf("taking back the whole loan left %d loans and %d lent bytes", len(lending.loans), lending.lentBytes)
	}
}

// readsItsBytes holds every expert table entry of the residency to its expert's own bytes in payloads.
func readsItsBytes(t *testing.T, cache *expertResidency, mem *fakeExpertDevice, payloads map[string][]byte) {
	t.Helper()
	for name, bank := range cache.banks {
		experts := uint64(len(bank.pointers))
		table, err := mem.at(bank.table, experts*8)
		if err != nil {
			t.Fatal(err)
		}
		for index := range experts {
			actual, err := mem.at(driver.DevicePtr(binary.LittleEndian.Uint64(table[index*8:])), bank.bytes)
			if err != nil || !bytes.Equal(actual, payloads[name][index*bank.bytes:(index+1)*bank.bytes]) {
				t.Fatalf("%s expert %d does not read its bytes: %v", name, index, err)
			}
		}
	}
}

// onCard lists a bank's experts on the card.
func onCard(cache *expertResidency, name string) []uint32 {
	var indices []uint32
	for index, entry := range cache.banks[name].entries {
		if entry != nil {
			indices = append(indices, uint32(index))
		}
	}
	return indices
}

// TestReleaseKeepsTheHottestExperts releases the card allocation of a bank whose experts are used more than those of
// another bank of their stride: the released bank's experts take the colder bank's card slots, copied on the card,
// the colder experts go to host memory, and every table reads its expert's bytes; where the other bank's experts are
// the more used, the released bank's own experts go.
func TestReleaseKeepsTheHottestExperts(t *testing.T) {
	f := writeFixture(t, rand.New(rand.NewPCG(31, 37)), []fixtureTensor{
		{name: "gate", shape: []uint64{2, 2, 3}, kind: dtype.BF16},
		{name: "up", shape: []uint64{2, 2, 3}, kind: dtype.BF16},
	}, 0, 0)
	file := f.parse(t)
	const (
		experts    = 3
		tableBytes = 2 * experts * 8
		stride     = 256
		twoEach    = tableBytes + 2*2*stride
	)
	for _, hot := range []string{"up", "gate"} {
		t.Run("hot="+hot, func(t *testing.T) {
			mem := newFakeExpertDevice()
			weights, err := mem.loadExperts(t.Context(), file, twoEach, nil)
			if err != nil {
				t.Fatal(err)
			}
			cache := weights.experts
			t.Cleanup(func() {
				if err := cache.close(); err != nil || mem.liveCount() != 0 {
					t.Errorf("close left %d allocations: %v", mem.liveCount(), err)
				}
			})
			placed := map[string][]uint32{"gate": onCard(cache, "gate"), "up": onCard(cache, "up")}
			if len(placed["gate"]) != 2 || len(placed["up"]) != 2 || len(cache.cards) != 2 {
				t.Fatalf("the load placed %v in %d allocations, want two experts of each bank in one a bank", placed, len(cache.cards))
			}
			for _, index := range placed[hot] {
				cache.banks[hot].usage[index] = 5
			}
			// The up bank's allocation is the last, which the release frees.
			if err := cache.release(t.Context()); err != nil {
				t.Fatal(err)
			}
			kept, gone := "up", "gate"
			if hot == "gate" {
				kept, gone = "gate", "up"
			}
			if got := onCard(cache, kept); !slices.Equal(got, placed[kept]) {
				t.Fatalf("the release left %s's experts %v on the card, want %v", kept, got, placed[kept])
			}
			if got := onCard(cache, gone); len(got) != 0 {
				t.Fatalf("the release kept %s's experts %v on the card", gone, got)
			}
			for _, name := range []string{"gate", "up"} {
				bank := cache.banks[name]
				table, err := mem.at(bank.table, experts*8)
				if err != nil {
					t.Fatal(err)
				}
				for index := range uint64(experts) {
					actual, err := mem.at(driver.DevicePtr(binary.LittleEndian.Uint64(table[index*8:])), bank.bytes)
					if err != nil || !bytes.Equal(actual, f.payloads[name][index*bank.bytes:(index+1)*bank.bytes]) {
						t.Fatalf("%s expert %d does not read its bytes: %v", name, index, err)
					}
				}
			}
		})
	}
}

// TestPublishQueuesBehindTheRun holds the residency's publishes to the ordered write: an advance that promotes a
// routed expert into a card expert's slot writes the tables and the card's flags of experts on host only behind
// the compute stream's work, never with a write that waits for the run in flight, and every table still reads its
// expert's bytes once the promotion lands.
func TestPublishQueuesBehindTheRun(t *testing.T) {
	f := writeFixture(t, rand.New(rand.NewPCG(41, 43)), []fixtureTensor{
		{name: "gate", shape: []uint64{2, 2, 3}, kind: dtype.BF16},
		{name: "down", shape: []uint64{16, 16, 3}, kind: dtype.F32},
	}, 0, 0)
	file := f.parse(t)
	const (
		experts                = 3
		tableBytes             = 2 * experts * 8
		gateStride, downStride = 256, 1024
		oneEach                = tableBytes + gateStride + downStride
	)
	mem := newFakeExpertDevice()
	weights, err := mem.loadExperts(t.Context(), file, oneEach, nil)
	if err != nil {
		t.Fatal(err)
	}
	cache := weights.experts
	t.Cleanup(func() {
		if err := cache.close(); err != nil || mem.liveCount() != 0 {
			t.Errorf("close left %d allocations: %v", mem.liveCount(), err)
		}
	})
	builder := tensor.NewBuilder()
	var nodes []*tensor.Tensor
	for _, bank := range file.Tensors {
		nodes = append(nodes, builder.Input(bank.Name, bank.Type, tensor.MustShape(bank.Shape[:bank.Dimensions]...)))
	}
	flags, _, err := (&DeviceWeights{experts: cache}).ExpertRoutes(nil, nodes)
	if err != nil {
		t.Fatal(err)
	}
	advance := func() {
		t.Helper()
		cache.routed[uint64(flags-cache.routedMap)+2] = 1
		if err := cache.advance(t.Context(), false); err != nil {
			t.Fatal(err)
		}
	}
	writes, ordered := mem.writes, mem.ordered
	advance()
	if len(cache.promoting) == 0 {
		t.Fatal("the advance promoted no expert")
	}
	if mem.ordered == ordered || mem.writes-writes != mem.ordered-ordered {
		t.Fatalf("the promoting advance wrote %d times, %d of them ordered, want every write ordered", mem.writes-writes, mem.ordered-ordered)
	}
	advance()
	for _, bank := range cache.banks {
		table, err := mem.at(bank.table, experts*8)
		if err != nil {
			t.Fatal(err)
		}
		for index := range uint64(experts) {
			actual, err := mem.at(driver.DevicePtr(binary.LittleEndian.Uint64(table[index*8:])), bank.bytes)
			if err != nil || !bytes.Equal(actual, f.payloads[bank.info.Name][index*bank.bytes:(index+1)*bank.bytes]) {
				t.Fatalf("%s expert %d does not read its bytes: %v", bank.info.Name, index, err)
			}
		}
	}
}
