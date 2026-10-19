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

func (d *fakeExpertDevice) read(_ context.Context, data []byte, source driver.DevicePtr) error {
	held, err := d.at(source, uint64(len(data)))
	copy(data, held)
	return err
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
				_, resident := cache.entries[expertKey{name, index}]
				if resident != (bank.hosted[index] == nil) || resident != slices.Contains(want, index) {
					t.Fatalf("%s expert %d on the card %t, want the card to hold %v", name, index, resident, want)
				}
			}
		}
		if cache.bytes > oneEach {
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
		bitmap, err := (&DeviceWeights{experts: cache}).ExpertRoutes(nil, nodes(t))
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
		// The tables, the pinned route flags, and one card allocation and one pinned allocation a bank.
		if len(cache.cards) != 2 || len(cache.pinned) != 3 || mem.liveCount() != 6 {
			t.Fatalf("fill made %d card and %d pinned allocations, %d live", len(cache.cards), len(cache.pinned), mem.liveCount())
		}
		reclaimable, err := (&DeviceWeights{experts: cache}).ExpertMemory(nil)
		if err != nil || reclaimable != gateStride+downStride {
			t.Fatalf("reclaimable card bytes = %d: %v", reclaimable, err)
		}
		placement, err := (&DeviceWeights{experts: cache}).ExpertPlacement(nil)
		want := ExpertPlacement{CardExperts: 2, HostExperts: 2*experts - 2, CardBytes: gateStride + downStride, HostBytes: (experts - 1) * (gateStride + downStride)}
		if err != nil || placement != want {
			t.Fatalf("placement = %+v, want %+v: %v", placement, want, err)
		}
		all, _ := open(t, tableBytes+experts*(gateStride+downStride))
		if len(all.pinned) != 1 || len(all.entries) != 2*experts {
			t.Fatal("a limit that holds every expert left one in host memory")
		}
	})
	t.Run("routes-promote-between-runs", func(t *testing.T) {
		cache, mem := open(t, oneEach)
		writes := mem.writes
		advance(t, cache)
		if mem.writes != writes {
			t.Fatal("a run that routed nothing moved experts")
		}
		route(t, cache, 2)
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{"gate": {2}, "down": {2}})
		if bytes.IndexByte(cache.routed, 1) >= 0 {
			t.Fatal("the routes stayed marked after the advance")
		}
		pinned := len(cache.pinned)
		// Expert 2 served this step too, so expert 0 stays in host memory.
		route(t, cache, 0, 2)
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{"gate": {2}, "down": {2}})
		route(t, cache, 1)
		advance(t, cache)
		readable(t, cache, mem, map[string][]uint32{"gate": {1}, "down": {1}})
		if len(cache.pinned) != pinned {
			t.Fatal("swaps pinned more host memory")
		}
		// Three runs named expert 2, then 0 and 2, then 1, in both banks; only the second found its expert 2 on the card.
		placement, err := (&DeviceWeights{experts: cache}).ExpertPlacement(nil)
		if err != nil || placement.Routes != 8 || placement.CardRoutes != 2 || placement.CardExperts != 2 {
			t.Fatalf("placement after the runs = %+v: %v", placement, err)
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
		// expert 1 takes expert 0's slot, and expert 2 finds the slot taken by an expert this run routed.
		route(t, cache, 1, 2)
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
	})
	t.Run("failed-promotion-keeps-tables-readable", func(t *testing.T) {
		for _, fail := range []string{"write", "pinning", "cancel"} {
			cache, mem := open(t, oneEach)
			route(t, cache, 2)
			ctx, cancel := context.WithCancel(t.Context())
			switch fail {
			case "write":
				// The first copy to the card.
				mem.failWrite = mem.writes + 1
			case "pinning":
				mem.failHost = mem.hostCalls + 1
			case "cancel":
				mem.onWrite = cancel
			}
			if err := cache.advance(ctx, false); err == nil {
				t.Fatalf("%s: a failed promotion was accepted", fail)
			}
			cancel()
			mem.failWrite, mem.failHost, mem.onWrite = 0, 0, nil
			readable(t, cache, mem, nil)
		}
	})
	t.Run("failed-publication-is-repaired-before-the-next-run", func(t *testing.T) {
		// Both experts move, then the first table write fails: the tables still name
		// the card slots the moves overwrote until a later advance writes them.
		cache, mem := open(t, oneEach)
		route(t, cache, 2)
		mem.failWrite = mem.writes + 3
		if err := cache.advance(t.Context(), false); err == nil {
			t.Fatal("a failed table write was accepted")
		}
		// The next run brings no routes; it may run only after the tables are repaired.
		mem.failWrite = mem.writes + 1
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
		if _, err := newExpertResidency(t.Context(), file, changed, oneEach, mem); err == nil || mem.liveCount() != 0 {
			t.Fatal("foreign descriptor accepted or allocated")
		}
		if _, err := newExpertResidency(t.Context(), file, file.Tensors, tableBytes-1, mem); err == nil || mem.liveCount() != 0 {
			t.Fatal("tables beyond the limit accepted or allocated")
		}
		cache, _ := open(t, oneEach)
		wrong := nodes(t)
		wrong[1] = tensor.NewBuilder().Input("down", dtype.F32, tensor.MustShape(16, 8, 6))
		if _, err := (&DeviceWeights{experts: cache}).ExpertRoutes(nil, wrong); err == nil {
			t.Fatal("a bank of another shape was accepted")
		}
	})
}

// onCard lists a bank's experts on the card.
func onCard(cache *expertResidency, name string) []uint32 {
	var indices []uint32
	for key := range cache.entries {
		if key.bank == name {
			indices = append(indices, key.index)
		}
	}
	return indices
}
