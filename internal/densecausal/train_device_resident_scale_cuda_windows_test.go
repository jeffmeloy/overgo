//go:build windows

package densecausal

import (
	"context"
	"math"
	"math/rand"
	"slices"
	"testing"
	"time"

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

// TestTrainDeviceResidentScratchPoolPeak proves the milestone-3 scratch pool:
// (a) the resident trajectory is UNCHANGED whether scratch is pooled or not (exact
// parity, so the pool does not perturb the math), and (b) the peak device bytes --
// measured from cuMemGetInfo, real GPU memory -- drop when the pool is on. The peak
// is sampled at each layer-op boundary; pooling reuses one arena so peak scratch is
// O(1 layer) instead of O(all layers).
func TestTrainDeviceResidentScratchPoolPeak(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	tokens := scaleTokens()
	const steps = 2

	// peakResult carries two independent real measurements of a run's peak device
	// memory: freePeak = the drop in driver-reported free memory (cuMemGetInfo,
	// masked by the driver's freed-memory pool) and allocPeak = the high-water of
	// concurrently-live cuMemAlloc bytes (the driver's own allocation accounting,
	// reset per run). allocPeak is the unambiguous scratch-accumulation signal.
	type peakResult struct {
		freePeak  uint64
		allocPeak uint64
		traj      []float64
		wall      time.Duration
	}
	run := func(pool bool) peakResult {
		m := syntheticCausalModel(t, scaleModelSpec) // identical seeded weights each run
		prevPool := devicemath.SetScratchPoolEnabled(pool)
		defer devicemath.SetScratchPoolEnabled(prevPool)
		var probe devicemath.PeakProbe
		prevProbe := devicemath.SetPeakProbe(&probe)
		defer devicemath.SetPeakProbe(prevProbe)
		if err := worker.Do(context.Background(), func(s *device.State) error {
			s.Driver.ResetPeakBytes()
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		traj, err := m.TrainDeviceResidentBatches(worker, slices.Repeat([][]int{tokens}, steps), 0, 0.9)
		wall := time.Since(started)
		if err != nil {
			t.Fatalf("TrainDeviceResidentBatches(pool=%v): %v", pool, err)
		}
		stats, err := worker.MemoryStats(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		return peakResult{freePeak: probe.PeakBytes(), allocPeak: stats.PeakBytes, traj: traj, wall: wall}
	}

	unpooled := run(false)
	pooled := run(true)
	unpooledPeak, unpooledTraj := unpooled.freePeak, unpooled.traj
	pooledPeak, pooledTraj := pooled.freePeak, pooled.traj

	// (a) Exact parity between the pooled and unpooled paths.
	if len(pooledTraj) != len(unpooledTraj) {
		t.Fatalf("trajectory lengths differ: pooled %d unpooled %d", len(pooledTraj), len(unpooledTraj))
	}
	var worst float64
	for i := range pooledTraj {
		if d := math.Abs(pooledTraj[i] - unpooledTraj[i]); d > worst {
			worst = d
		}
	}
	t.Logf("pool parity worst |d|=%.3e (pooled[last]=%.6f unpooled[last]=%.6f)", worst, pooledTraj[len(pooledTraj)-1], unpooledTraj[len(unpooledTraj)-1])
	if worst > 1e-5 {
		t.Fatalf("scratch pool changed the trajectory: worst |d|=%.3e > 1e-5", worst)
	}

	// (b) Measured peak device bytes drop with the pool on -- two real signals.
	t.Logf("cuMemGetInfo free-drop peak: unpooled=%d (%.1f MiB)  pooled=%d (%.1f MiB)  saved=%.1f MiB",
		unpooledPeak, float64(unpooledPeak)/(1<<20),
		pooledPeak, float64(pooledPeak)/(1<<20),
		float64(unpooledPeak-pooledPeak)/(1<<20))
	t.Logf("live cuMemAlloc high-water: unpooled=%d (%.1f MiB)  pooled=%d (%.1f MiB)  saved=%.1f MiB",
		unpooled.allocPeak, float64(unpooled.allocPeak)/(1<<20),
		pooled.allocPeak, float64(pooled.allocPeak)/(1<<20),
		float64(unpooled.allocPeak-pooled.allocPeak)/(1<<20))
	t.Logf("resident training wall: unpooled=%s pooled=%s (%d steps, %.1f ms/step)",
		unpooled.wall, pooled.wall, steps, float64(pooled.wall.Microseconds())/1000/steps)
	if pooledPeak == 0 || unpooledPeak == 0 {
		t.Fatalf("peak probe returned zero (unpooled=%d pooled=%d)", unpooledPeak, pooledPeak)
	}
	if pooledPeak >= unpooledPeak {
		t.Fatalf("scratch pool did not lower cuMemGetInfo peak: pooled=%d >= unpooled=%d", pooledPeak, unpooledPeak)
	}
	if pooled.allocPeak >= unpooled.allocPeak {
		t.Fatalf("scratch pool did not lower live-alloc peak: pooled=%d >= unpooled=%d", pooled.allocPeak, unpooled.allocPeak)
	}
	if pooled.wall > 2*time.Second {
		t.Fatalf("pooled resident training wall %s exceeds 2s baseline", pooled.wall)
	}
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
