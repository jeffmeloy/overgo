package modeldevice

import (
	"bytes"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestACardExpertWithoutAHostCopyStaysOnTheCard homes a residency's card experts on a machine whose commit holds the
// copies of all but the hottest (HomeExperts): that one stays on the card through a promotion and an adaptive swap
// that would take its slot, a drop of its allocation moves it to a free card slot pinning nothing, and a release with
// no slot left gives it a copy read from its slot as it leaves, every table reading its expert's bytes throughout; the
// experts with copies move as before.
func TestACardExpertWithoutAHostCopyStaysOnTheCard(t *testing.T) {
	const (
		gate, down = "blk.0.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight"
		experts    = 3
		tableBytes = 2 * experts * 8
		// A gate expert takes a slot of 256 bytes, a down expert one of 1,024: two of each fit on the card.
		gateStride, downStride = 256, 1024
		twoEach                = tableBytes + 2*(gateStride+downStride)
	)
	f := writeFixture(t, rand.New(rand.NewPCG(53, 59)), []fixtureTensor{
		{name: gate, shape: []uint64{2, 2, experts}, kind: dtype.BF16},
		{name: down, shape: []uint64{16, 16, experts}, kind: dtype.F32},
	}, 0, 0)
	file := f.parse(t)
	mem := newFakeExpertDevice()
	weights := mem.weights()
	loader := NewLoader(file)
	loader.Experts(weights, file.Tensors, twoEach)
	// A machine whose commit holds no copy of a card expert at the load: the copies wait for the homing.
	mem.commitLeft = 0
	testcheck.NoError(t, loader.Load(t.Context(), nil))
	cache := weights.experts
	t.Cleanup(residencyClosed(t, cache, mem))
	if !slices.Equal(onCard(cache, gate), []uint32{0, 1}) || !slices.Equal(onCard(cache, down), []uint32{0, 1}) {
		t.Fatalf("the load placed gate %v and down %v on the card, want experts 0 and 1 of each", onCard(cache, gate), onCard(cache, down))
	}
	// The load pins the experts in host memory alone; their copies wait for the homing.
	if placement := testcheck.Must(weights.ExpertPlacement(nil))(t); placement.CardOnlyExperts != 4 || placement.HostExperts != 2 {
		t.Fatalf("the load reports %+v, want four card experts without a copy and two in host memory", placement)
	}
	// Down expert 0 is the hottest, so it is homed last; the commit left past the reserve holds the other three.
	cache.banks[down].usage[0] = 5
	mem.commitLeft = expertHomeCommitReserveBytes + 2*gateStride + downStride
	homes := testcheck.Must(weights.HomeExperts(t.Context(), nil))(t)
	if homes.Homed != 3 || homes.CardOnly != 1 || homes.CardOnlyBytes != downStride || cardOnly(cache.banks[down].entries[0]) == false {
		t.Fatalf("the homing gave %+v, want three copies and down expert 0 left to the card", homes)
	}
	for name, indices := range map[string][]uint32{gate: {0, 1}, down: {1}} {
		bank := cache.banks[name]
		for _, index := range indices {
			start := uint64(index) * bank.bytes
			if !bytes.Equal(bank.home[index].host[:bank.bytes], f.payloads[name][start:start+bank.bytes]) {
				t.Fatalf("%s expert %d's copy does not hold its bytes", name, index)
			}
		}
	}
	readsItsBytes(t, cache, mem, f.payloads)
	builder := tensor.NewBuilder()
	node := func(name string) *tensor.Tensor {
		info, _ := file.Tensor(name)
		return builder.Input(name, info.Type, tensor.MustShape(info.Shape[:info.Dimensions]...))
	}
	nodes := map[string]*tensor.Tensor{gate: node(gate), down: node(down)}
	// route marks the experts a run routed in each named bank, and the advance after it moves them.
	route := func(t *testing.T, name string, indices ...uint32) {
		t.Helper()
		flags, _, err := weights.ExpertRoutes(nil, []*tensor.Tensor{nodes[name]})
		testcheck.NoError(t, err)
		for _, index := range indices {
			cache.routed[uint64(flags-cache.routedMap)+uint64(index)] = 1
		}
		testcheck.NoError(t, cache.advance(t.Context(), false))
		testcheck.NoError(t, cache.advance(t.Context(), false))
	}
	// Down expert 0 is the least recently routed, yet the miss takes expert 1's slot, which has a copy to read.
	route(t, down, 2)
	if got := onCard(cache, down); !slices.Equal(got, []uint32{0, 2}) {
		t.Fatalf("down's miss left %v on the card, want expert 0 kept and expert 2 in expert 1's slot", got)
	}
	// With expert 2 named this step, expert 1's miss finds no card expert with a copy to displace.
	route(t, down, 1, 2)
	if got := onCard(cache, down); !slices.Equal(got, []uint32{0, 2}) {
		t.Fatalf("a miss with no card expert to displace moved down's experts to %v", got)
	}
	route(t, gate, 2)
	if got := onCard(cache, gate); !slices.Equal(got, []uint32{1, 2}) {
		t.Fatalf("gate's miss left %v on the card, want expert 2 in expert 0's slot", got)
	}
	readsItsBytes(t, cache, mem, f.payloads)
	// The adaptive tier's swap out of down expert 0 is refused; out of expert 2 it is queued.
	queued := testcheck.Must(weights.SwapExperts(t.Context(), nil, []ExpertSwap{{Layer: 1, In: 1, Out: 0}}))(t)
	if queued != 0 || cache.banks[down].entries[0] == nil {
		t.Fatalf("a swap out of a card expert without a copy queued %d", queued)
	}
	if queued := testcheck.Must(weights.SwapExperts(t.Context(), nil, []ExpertSwap{{Layer: 1, In: 1, Out: 2}}))(t); queued != 1 {
		t.Fatalf("a swap out of a card expert with a copy queued %d", queued)
	}
	testcheck.NoError(t, weights.LandExperts(t.Context(), nil))
	if got := onCard(cache, down); !slices.Equal(got, []uint32{0, 1}) {
		t.Fatalf("the swaps left %v on the card, want experts 0 and 1", got)
	}
	readsItsBytes(t, cache, mem, f.payloads)
	// A miss with free card memory takes a new allocation's slot; dropping down's first allocation then moves expert 0
	// to a free slot of the new one, on the card, and pins nothing.
	cache.limit, mem.freeBytes = math.MaxUint64, 1<<30
	route(t, down, 2)
	if len(cache.cards) != 3 || !slices.Equal(onCard(cache, down), []uint32{0, 1, 2}) {
		t.Fatalf("down's miss with free memory left %d allocations and %v on the card", len(cache.cards), onCard(cache, down))
	}
	first := cache.cards[slices.IndexFunc(cache.cards, func(chunk expertChunk) bool { return chunk.stride == downStride })].pointer
	pinned := len(cache.pinned)
	testcheck.NoError(t, cache.drop(t.Context(), func(chunk expertChunk) bool { return chunk.pointer == first }, nil))
	if !slices.Equal(onCard(cache, down), []uint32{0, 1, 2}) || len(cache.pinned) != pinned || !cardOnly(cache.banks[down].entries[0]) {
		t.Fatalf("dropping down's first allocation left %v on the card and pinned %d allocations, want every expert kept and none pinned",
			onCard(cache, down), len(cache.pinned)-pinned)
	}
	kept := cache.cards[slices.IndexFunc(cache.cards, func(chunk expertChunk) bool { return chunk.stride == downStride })]
	if moved := cache.banks[down].pointers[0]; kept.pointer == first || moved < kept.pointer || moved >= kept.pointer+driver.DevicePtr(kept.bytes) {
		t.Fatal("down expert 0 did not move into the allocation the card kept")
	}
	readsItsBytes(t, cache, mem, f.payloads)
	if placement := testcheck.Must(weights.ExpertPlacement(nil))(t); placement.CardOnlyExperts != 1 || placement.CardOnlyBytes != downStride {
		t.Fatalf("placement reports %+v, want down expert 0 alone without a copy", placement)
	}
	// With no card slot left, as an open's reservation before the homing finds every card expert, expert 0 leaves with a
	// copy read from its slot: one pinned allocation.
	for len(cache.cards) > 0 {
		testcheck.NoError(t, cache.release(t.Context()))
	}
	bank := cache.banks[down]
	if len(onCard(cache, down)) != 0 || len(cache.pinned) != pinned+1 || !bytes.Equal(bank.hosted[0][:bank.bytes], f.payloads[down][:bank.bytes]) {
		t.Fatalf("releasing the card left %v of down on it and pinned %d allocations, want expert 0 in host memory with its bytes in one",
			onCard(cache, down), len(cache.pinned)-pinned)
	}
	readsItsBytes(t, cache, mem, f.payloads)
	if placement := testcheck.Must(weights.ExpertPlacement(nil))(t); placement.CardOnlyExperts != 0 || placement.HostExperts != 2*experts {
		t.Fatalf("placement reports %+v, want every expert in host memory", placement)
	}
}
