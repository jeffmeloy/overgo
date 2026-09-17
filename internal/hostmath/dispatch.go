// Calibrated dispatch: whether a fan-out pays, and across how many workers,
// is DERIVED from one-time host measurements rather than carried thresholds
// (a carried crossover goes stale with every host change; the calibration
// cannot). Ported from the reference implementation: adaptive extmodel
// hostDispatchCalibration + matrix.TimePerOpNs.
//
// Measured quantities:
//
//	perTaskNs    — no-op fanOut round trip / tasks submitted (one channel
//	               send + worker wakeup + Done per task, so total dispatch
//	               overhead is linear in tasks — structure, not tuning)
//	macNs[rate]  — inline per-MAC rate of the real range kernels
//
// Cost model for n units of unitMACs each at w workers (caller included):
//
//	wall(w) = S/w + perTaskNs*(w-1),  S = n*unitMACs*macNs
//
// Minimizing over w gives w* = sqrt(S/perTaskNs). Any w <= w* pays:
// w <= sqrt(S/perTaskNs) => S >= perTaskNs*w^2, so the time saved
// S*(1-1/w) = S*(w-1)/w >= perTaskNs*w*(w-1) >= perTaskNs*(w-1), the
// dispatch cost — a clamped w >= 2 already implies "pays"; no separate
// crossover check is needed.
package hostmath

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"

	"overgo/internal/processmeasure"
)

// calibrationMinBatchNs: a timed batch is long enough that scheduling noise
// inside it is a small fraction of the reading; the batch is read through
// the performance counter, whose step is far below this floor.
const calibrationMinBatchNs = 20e6

// calibrationDim: square kernel shape; dim^2 = 2^18 MACs keeps one op
// ~100us of real work so the doubling batcher qualifies in a few steps.
// Protocol size for timer signal, not a behavior threshold.
const calibrationDim = 1 << 9

// timePerOpNs: times op in doubling batches until one batch exceeds the
// granularity floor, then returns the MINIMUM per-op mean over three
// qualifying batches. Minimum, not mean: contention can only inflate a
// timing, so the smallest batch is closest to the uncontended cost.
func timePerOpNs(op func()) (float64, error) {
	const qualifyingBatches = 3
	best := 0.0
	reps := 1
	for measured := 0; measured < qualifyingBatches; {
		t0 := processmeasure.NewStopwatch()
		for range reps {
			op()
		}
		wall, err := t0.Elapsed()
		if err != nil {
			return 0, err
		}
		elapsed := float64(wall)
		if elapsed < calibrationMinBatchNs {
			reps *= 2
			continue
		}
		measured++
		if perOp := elapsed / float64(reps); best == 0 || perOp < best {
			best = perOp
		}
	}
	return best, nil
}

// macRate selects which measured kernel rate prices a call's unitMACs.
type macRate int

const (
	macF32       macRate = iota // f32-accumulate dot (the Linear kernel)
	macF64                      // f64-accumulate MAC (conv/attention kernels)
	macRateCount int     = iota
)

type dispatchCalibration struct {
	perTaskNs float64
	macNs     [macRateCount]float64
}

// dispatchOnce measures the host once; a failed measurement is reported
// once and every kernel then runs serial.
var dispatchOnce = sync.OnceValues(func() (dispatchCalibration, error) {
	cal, err := measureDispatch()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[hostmath-dispatch-calibration] failed: %v; every kernel runs serial\n", err)
	}
	return cal, err
})

func poolWorkerLimit() int { return runtime.GOMAXPROCS(0) }

// calibrationNop must be top-level so the overhead measurement times the
// dispatch machinery alone.
func calibrationNop(int, int) {}

func measureDispatch() (dispatchCalibration, error) {
	workers := poolWorkerLimit()
	if workers < 2 {
		workers = 2 // fanOut needs a task; a single-core host never dispatches anyway
	}
	var cal dispatchCalibration
	perTask, err := timePerOpNs(func() {
		fanOut(workers, workers, calibrationNop)
	})
	if err != nil {
		return dispatchCalibration{}, err
	}
	cal.perTaskNs = perTask / float64(workers-1)

	// f32 rate: the exact Linear column kernel, one row.
	x := make([]float32, calibrationDim)
	w := make([]float32, calibrationDim*calibrationDim)
	out := make([]float32, calibrationDim)
	for i := range x {
		x[i] = float32(i%17) / 17
	}
	for i := range w {
		w[i] = float32(i%23) / 23
	}
	f32, err := timePerOpNs(func() {
		linearCols(out, x, w, 1, calibrationDim, calibrationDim, 0, calibrationDim)
	})
	if err != nil {
		return dispatchCalibration{}, err
	}
	cal.macNs[macF32] = f32 / float64(calibrationDim*calibrationDim)

	// f64 rate: the exact CausalConv1d channel kernel. These protocol extents
	// realize the same MAC budget as the square F32 calibration while keeping
	// one output channel; closure evidence owns both local constants.
	const cIn, ck = 1 << 6, 1 << 3
	cx := make([]float32, cIn*calibrationDim)
	cw := make([]float32, cIn*ck)
	cOut := make([]float32, calibrationDim)
	for i := range cx {
		cx[i] = float32(i%19) / 19
	}
	for i := range cw {
		cw[i] = float32(i%23) / 23
	}
	f64, err := timePerOpNs(func() {
		causalConv1dChannels(cOut, cx, cw, nil, cIn, calibrationDim, calibrationDim, ck, 1, ck-1, 0, 1)
	})
	if err != nil {
		return dispatchCalibration{}, err
	}
	cal.macNs[macF64] = f64 / float64(calibrationDim*cIn*ck)

	// Loud one-line record of what this process derived its dispatch from: a
	// biased calibration is invisible in every downstream symptom (it just
	// makes different dispatch choices), so the evidence is emitted at the
	// moment it is measured.
	fmt.Fprintf(os.Stderr, "[hostmath-dispatch-calibration] task=%.0fns f32=%.4fns/mac f64=%.4fns/mac workers=%d\n",
		cal.perTaskNs, cal.macNs[macF32], cal.macNs[macF64], workers)
	return cal, nil
}

// dispatchWorkers: worker count for n units of unitMACs each; 1 means run
// serial. w* = sqrt(S/perTaskNs) clamped to [1, min(n, GOMAXPROCS)]; per the
// model above, any clamped w >= 2 pays for its own dispatch.
func dispatchWorkers(n, unitMACs int, rate macRate) int {
	if n < 2 {
		return 1
	}
	limit := poolWorkerLimit()
	limit = min(limit, n)
	if limit < 2 {
		return 1
	}
	cal, err := dispatchOnce()
	if err != nil {
		return 1
	}
	s := float64(n) * float64(unitMACs) * cal.macNs[rate]
	w := int(math.Sqrt(s / cal.perTaskNs))
	w = min(w, limit)
	if w < 2 {
		return 1
	}
	return w
}
