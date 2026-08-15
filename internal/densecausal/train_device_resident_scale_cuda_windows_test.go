//go:build windows

package densecausal

import (
	"math/rand"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/devicemath"
	"overgo/internal/testutil"
)

// scaleModelSpec is a multi-layer geometry where the per-layer transient scratch
// is a real fraction of device memory, so the scratch pool's peak reduction is
// measurable and the capacity derivation is exercised on >2 layers.
var scaleModelSpec = testutil.DenseCausalSpec{
	Vocab: 256, Hidden: 256, Heads: 8, HeadDim: 32,
	KVHeads: 2, Intermediate: 768, Layers: 12, Seed: 3,
}

func syntheticCausalModel(t *testing.T, spec testutil.DenseCausalSpec) *Model {
	t.Helper()
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	model, err := NewModel(weights, shapes, spec.Heads, spec.HeadDim, 10000, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func scaleTokens() []int {
	tokens := make([]int, 64)
	rng := rand.New(rand.NewSource(11))
	for i := range tokens {
		tokens[i] = rng.Intn(scaleModelSpec.Vocab)
	}
	return tokens
}

// TestTrainDeviceResidentDerivedCapacity proves milestone-3 part 2: the resident
// scale ceiling (how many layers fit) is DERIVED from measured free device memory
// and a measured allocator granularity G, not a supplied/hardcoded constant.
func TestTrainDeviceResidentDerivedCapacity(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	// G is MEASURED cold (free-memory drop for a 1-byte alloc), never a constant.
	g, err := devicemath.MeasureAllocGranularity(worker)
	if err != nil {
		t.Fatal(err)
	}
	if g == 0 {
		t.Fatal("measured granularity G is zero")
	}
	t.Logf("measured allocator granularity G = %d bytes (%.0f KiB)", g, float64(g)/1024)

	m := syntheticCausalModel(t, scaleModelSpec)
	seq := len(scaleTokens())
	capacity, err := m.DeriveResidentCapacity(worker, seq)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("derived capacity: free=%d bytes (%.1f GiB) G=%d perLayer=%d bytes reserve=%d bytes -> %d layers fit (model has %d)",
		capacity.FreeBytes, float64(capacity.FreeBytes)/(1<<30), capacity.Granularity,
		capacity.PerLayerBytes, capacity.ReserveBytes, capacity.Layers, m.Dims.Layers)

	if capacity.FreeBytes == 0 {
		t.Fatal("capacity.FreeBytes is zero (free memory not measured)")
	}
	if capacity.Granularity != g {
		t.Fatalf("capacity used G=%d but measured G=%d (G must be the measured value)", capacity.Granularity, g)
	}
	if capacity.ReserveBytes == 0 {
		t.Fatal("reserve bound is zero (granularity rounding waste must be reserved)")
	}
	if capacity.Layers < m.Dims.Layers {
		t.Fatalf("derived capacity %d < model layers %d on an idle 48GB GPU", capacity.Layers, m.Dims.Layers)
	}

	// Prove the capacity is a FUNCTION of the measured free memory (not a constant):
	// halving the measured free memory must roughly halve the derived layer count.
	fixed, perLayer := devicemath.ResidentLayerPlanSizes(
		seq, m.Dims.Hidden, m.Dims.Heads, m.Dims.KVHeads, m.Dims.HeadDim, m.Dims.Intermediate, m.perLayerWeightElems())
	half := devicemath.DeriveResidentCapacity(capacity.FreeBytes/2, g, fixed, perLayer)
	t.Logf("capacity(free/2) = %d layers (vs %d at full free) -- capacity tracks measured free", half.Layers, capacity.Layers)
	if !(half.Layers < capacity.Layers) {
		t.Fatalf("capacity is not a function of measured free: full=%d half=%d", capacity.Layers, half.Layers)
	}

	// Prove the reserve bound is a FUNCTION of the measured granularity G: a larger
	// G reserves strictly more rounding waste for the same plan.
	coarse := devicemath.DeriveResidentCapacity(capacity.FreeBytes, g*2, fixed, perLayer)
	if !(coarse.ReserveBytes > capacity.ReserveBytes) {
		t.Fatalf("reserve bound does not grow with G: G=%d reserve=%d, G=%d reserve=%d", g, capacity.ReserveBytes, g*2, coarse.ReserveBytes)
	}
	t.Logf("reserve(G=%d)=%d < reserve(G=%d)=%d -- reserve tracks measured G", g, capacity.ReserveBytes, g*2, coarse.ReserveBytes)
}
