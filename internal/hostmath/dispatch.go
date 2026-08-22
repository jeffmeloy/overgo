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
	"time"
)

// calibrationMinBatchNs: a timed batch must exceed the WORST Windows
// monotonic-clock granularity, the 15.6ms interrupt tick (a shorter batch
// times as zero). Clock structure, not tuning.
const calibrationMinBatchNs = 20e6

// calibrationDim: square kernel shape; dim^2 = 2^18 MACs keeps one op
// ~100us of real work so the doubling batcher qualifies in a few steps.
// Protocol size for timer signal, not a behavior threshold.
const calibrationDim = 1 << 9

// timePerOpNs: times op in doubling batches until one batch exceeds the
// granularity floor, then returns the MINIMUM per-op mean over three
// qualifying batches. Minimum, not mean: contention can only inflate a
// timing, so the smallest batch is closest to the uncontended cost.
func timePerOpNs(op func()) float64 {
	const qualifyingBatches = 3
	best := 0.0
	reps := 1
	for measured := 0; measured < qualifyingBatches; {
		t0 := time.Now()
		for i := 0; i < reps; i++ {
			op()
		}
		elapsed := float64(time.Since(t0).Nanoseconds())
		if elapsed < calibrationMinBatchNs {
			reps *= 2
			continue
		}
		measured++
		if perOp := elapsed / float64(reps); best == 0 || perOp < best {
			best = perOp
		}
	}
	return best
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

var dispatchOnce = sync.OnceValue(measureDispatch)

func poolWorkerLimit() int { return runtime.GOMAXPROCS(0) }

// ParallelWorkerLimit reports the maximum worker count used by host kernels.
func ParallelWorkerLimit() int { return poolWorkerLimit() }

// calibrationNop must be top-level so the overhead measurement times the
// dispatch machinery alone.
func calibrationNop(int, int) {}

func measureDispatch() dispatchCalibration {
	workers := poolWorkerLimit()
	if workers < 2 {
		workers = 2 // fanOut needs a task; a single-core host never dispatches anyway
	}
	var cal dispatchCalibration
	cal.perTaskNs = timePerOpNs(func() {
		fanOut(workers, workers, calibrationNop)
	}) / float64(workers-1)

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
	cal.macNs[macF32] = timePerOpNs(func() {
		linearCols(out, x, w, 1, calibrationDim, calibrationDim, 0, calibrationDim)
	}) / float64(calibrationDim*calibrationDim)

	// f64 rate: the exact CausalConv1d channel kernel; shape realizes the
	// same MAC budget as outT*cIn*k = 512*64*8 = 2^18, one output channel.
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
	cal.macNs[macF64] = timePerOpNs(func() {
		causalConv1dChannels(cOut, cx, cw, nil, cIn, calibrationDim, calibrationDim, ck, 1, ck-1, 0, 1)
	}) / float64(calibrationDim*cIn*ck)

	// Loud one-line record of what this process derived its dispatch from: a
	// biased calibration is invisible in every downstream symptom (it just
	// makes different dispatch choices), so the evidence is emitted at the
	// moment it is measured.
	fmt.Fprintf(os.Stderr, "[hostmath-dispatch-calibration] task=%.0fns f32=%.4fns/mac f64=%.4fns/mac workers=%d\n",
		cal.perTaskNs, cal.macNs[macF32], cal.macNs[macF64], workers)
	return cal
}

// dispatchWorkers: worker count for n units of unitMACs each; 1 means run
// serial. w* = sqrt(S/perTaskNs) clamped to [1, min(n, GOMAXPROCS)]; per the
// model above, any clamped w >= 2 pays for its own dispatch.
func dispatchWorkers(n, unitMACs int, rate macRate) int {
	if n < 2 {
		return 1
	}
	limit := poolWorkerLimit()
	if limit > n {
		limit = n
	}
	if limit < 2 {
		return 1
	}
	cal := dispatchOnce()
	s := float64(n) * float64(unitMACs) * cal.macNs[rate]
	w := int(math.Sqrt(s / cal.perTaskNs))
	if w > limit {
		w = limit
	}
	if w < 2 {
		return 1
	}
	return w
}
