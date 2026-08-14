//go:build windows

package devicemath

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// --- Scratch pool -----------------------------------------------------------
// One device region reused across every layer's transient forward/backward
// scratch, so peak scratch is O(1 layer) not O(all layers). The activation
// caches, residuals and weight grads MUST persist across the fwd->bwd and stay
// per-layer resident; only the per-op scratch that never survives its layer is
// pooled here.

// scratchArena is a bump allocator over one pre-allocated device buffer. Each
// layer's forwardDevice/backwardDevice draws its transient buffers here and
// runStack rewinds the cursor after each layer, so the same addresses serve
// every layer. Stream ordering (one stream) makes reuse safe: the prior layer's
// kernels retire before the next layer's issue -- no sync needed.
type scratchArena struct {
	base     driver.DevicePtr
	capElems int
	cur      int
	high     int // high-water cursor (measurement only)
}

// alloc hands out the next n elements. Overflow is a loud error: it means the
// scratch size helpers below drifted from the actual forwardDevice/backwardDevice
// allocations (an undercount can never silently corrupt reused memory).
func (a *scratchArena) alloc(n int) (driver.DevicePtr, error) {
	if n <= 0 {
		return 0, errors.New("scratchArena: non-positive alloc")
	}
	if a.cur+n > a.capElems {
		return 0, fmt.Errorf("scratchArena overflow: need %d more, cap %d, cursor %d (scratch size helper drifted from ops)", n, a.capElems, a.cur)
	}
	p := ptrAt(a.base, a.cur)
	a.cur += n
	if a.cur > a.high {
		a.high = a.cur
	}
	return p, nil
}

func (a *scratchArena) reset() { a.cur = 0 }

// forwardScratchElems counts forwardDevice's transient scratch buffers
// (qHM,kHM,vHM,attnHM,attnOut,mlp,scores,pBuf) -- the arena's forward size.
// Single owner; drift is caught loudly by scratchArena.alloc overflow.
func forwardScratchElems(d layerDims) int {
	return 2*d.seq*d.width + 2*d.seq*d.kvWidth + 2*d.seq*d.hidden + 2*d.seq*d.seq
}

// backwardScratchElems counts backwardDevice's transient scratch buffers:
// 4*(seq*inter) + 9*(seq*hidden) + 5*(seq*width) + 6*(seq*kvWidth)
// + 4*(seq*seq) + (seq*hd). Single owner; drift caught by arena overflow.
func backwardScratchElems(d layerDims) int {
	return 4*d.seq*d.inter + 9*d.seq*d.hidden + 5*d.seq*d.width + 6*d.seq*d.kvWidth + 4*d.seq*d.seq + d.seq*d.hd
}

// maxScratchElems sizes the shared arena to one layer's larger pass (backward).
func maxScratchElems(d layerDims) int {
	if f := forwardScratchElems(d); f > backwardScratchElems(d) {
		return f
	}
	return backwardScratchElems(d)
}

// scratchPoolEnabled selects runStack's scratch strategy: true (production
// default) reuses one arena; false falls back to the legacy per-layer session
// allocations. The false path exists ONLY so a test can measure peak device
// bytes both ways on one GPU (see SetScratchPoolEnabled) -- production pools.
var scratchPoolEnabled = true

// SetScratchPoolEnabled toggles the arena and returns the prior value. A
// measurement seam for the peak before/after test, not for production callers.
func SetScratchPoolEnabled(on bool) bool {
	prev := scratchPoolEnabled
	scratchPoolEnabled = on
	return prev
}

// --- Peak device-byte probe -------------------------------------------------

// PeakProbe records the real device high-water of a resident stack run by
// sampling cuMemGetInfo at each layer-op boundary (the local allocation maxima).
// PeakBytes is the driver-reported memory consumed between session start and the
// busiest moment -- real GPU bytes, not an accounting sum.
type PeakProbe struct {
	startFree uint64
	minFree   uint64
	inited    bool
}

// PeakBytes returns the peak device bytes the probed run held live.
func (p *PeakProbe) PeakBytes() uint64 {
	if p == nil || !p.inited || p.startFree < p.minFree {
		return 0
	}
	return p.startFree - p.minFree
}

// activePeakProbe is the measurement seam (nil in production). runStack samples
// it at the top and after each layer op.
var activePeakProbe *PeakProbe

// SetPeakProbe installs (nil clears) the run's peak probe and returns the prior
// probe. A measurement seam for the peak before/after test.
func SetPeakProbe(p *PeakProbe) *PeakProbe {
	prev := activePeakProbe
	activePeakProbe = p
	return prev
}

// sampleFree records free memory into the active probe, initialising startFree
// on first call. Cheap: cuMemGetInfo reflects synchronous cuMemAlloc immediately.
func (o *layerOps) sampleFree() error {
	p := activePeakProbe
	if p == nil {
		return nil
	}
	free, _, err := o.s.state.Driver.MemInfo()
	if err != nil {
		return err
	}
	if !p.inited {
		p.startFree, p.minFree, p.inited = free, free, true
		return nil
	}
	if free < p.minFree {
		p.minFree = free
	}
	return nil
}

// --- Measured allocator granularity + derived resident capacity -------------

// roundUpTo rounds size up to a multiple of unit (unit>0; unit==0 is identity).
func roundUpTo(size, unit uint64) uint64 {
	if unit == 0 {
		return size
	}
	if r := size % unit; r != 0 {
		return size + (unit - r)
	}
	return size
}

// MeasureAllocGranularity measures the CUDA allocation granularity G COLD: the
// drop in driver-reported free memory (cuMemGetInfo) caused by a minimal 1-byte
// cuMemAlloc. G is whatever the live driver reserves per allocation -- measured,
// never a constant, so it cannot reintroduce the 2 MiB magic.
func MeasureAllocGranularity(worker *device.Worker) (uint64, error) {
	var g uint64
	err := worker.Do(context.Background(), func(state *device.State) error {
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		free0, _, err := state.Driver.MemInfo()
		if err != nil {
			return err
		}
		p, err := state.Driver.MemAlloc(1)
		if err != nil {
			return err
		}
		free1, _, err := state.Driver.MemInfo()
		if err != nil {
			_ = state.Driver.MemFree(p)
			return err
		}
		if err := state.Driver.MemFree(p); err != nil {
			return err
		}
		if free0 < free1 {
			return fmt.Errorf("MeasureAllocGranularity: free grew after alloc (%d -> %d)", free0, free1)
		}
		if g = free0 - free1; g == 0 {
			return errors.New("MeasureAllocGranularity: no measurable free drop for a 1-byte alloc")
		}
		return nil
	})
	return g, err
}

// MeasureFreeBytes returns the driver-reported free device bytes right now.
func MeasureFreeBytes(worker *device.Worker) (uint64, error) {
	var free uint64
	err := worker.Do(context.Background(), func(state *device.State) error {
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		f, _, err := state.Driver.MemInfo()
		free = f
		return err
	})
	return free, err
}

// ResidentCapacity is the derived answer to "how many contiguous layer-blocks
// fit resident" -- the n_gpu_layers-class parameter, DERIVED from measured free
// memory and the measured granularity G, not supplied.
type ResidentCapacity struct {
	FreeBytes     uint64 // measured (cuMemGetInfo) at derivation time
	Granularity   uint64 // measured cold (G)
	FixedBytes    uint64 // fixed (non-per-layer) allocations, each rounded up to G
	PerLayerBytes uint64 // one layer-block's allocations, each rounded up to G
	ReserveBytes  uint64 // granularity rounding waste reserved across the fitted plan
	Layers        int    // how many layer-blocks fit in FreeBytes
}

// ResidentLayerPlanSizes returns, in BYTES, the (fixed, perLayer) allocation
// plan for fully-resident training of a pre-norm transformer layer of the given
// shape -- fed to DeriveResidentCapacity. Fixed = the shared scratch pool (one
// layer's worth, reused across all layers). Per-layer = the resident weights,
// gradient and Muon momentum shares (three buffers) + the 11 forward activation
// cache buffers + the residual. perLayerWeightElems is one layer's weight element
// count (all nine matrices + two norm vectors), supplied by the caller.
func ResidentLayerPlanSizes(seq, hidden, heads, kvHeads, hd, inter, perLayerWeightElems int) (fixed, perLayer []uint64) {
	d := newLayerDims(seq, hidden, heads, kvHeads, hd, inter, 0)
	b := func(elems int) uint64 { return uint64(elems) * f32Bytes }
	fixed = []uint64{b(maxScratchElems(d))}
	w := b(perLayerWeightElems)
	perLayer = []uint64{w, w, w} // dW, dG, dM shares
	for _, e := range layerCacheElemSpec(d) {
		perLayer = append(perLayer, b(e))
	}
	perLayer = append(perLayer, b(seq*hidden)) // residual
	return fixed, perLayer
}

// DeriveResidentCapacity computes how many layer-blocks fit in measuredFree given
// granularity G, the fixed (non-per-layer) allocation byte sizes, and one
// layer-block's allocation byte sizes. Every individual allocation is rounded up
// to G; that rounding IS the reserve bound = sum of roundUp(size,G)-size. Nothing
// here is a constant -- the caller passes measured free and G.
func DeriveResidentCapacity(measuredFree, g uint64, fixedSizes, perLayerSizes []uint64) ResidentCapacity {
	sum := func(sizes []uint64) (rounded, raw uint64) {
		for _, s := range sizes {
			rounded += roundUpTo(s, g)
			raw += s
		}
		return
	}
	fixedRounded, fixedRaw := sum(fixedSizes)
	perRounded, perRaw := sum(perLayerSizes)
	c := ResidentCapacity{
		FreeBytes: measuredFree, Granularity: g,
		FixedBytes: fixedRounded, PerLayerBytes: perRounded,
	}
	if perRounded == 0 || measuredFree <= fixedRounded {
		c.ReserveBytes = fixedRounded - fixedRaw
		return c
	}
	c.Layers = int((measuredFree - fixedRounded) / perRounded)
	c.ReserveBytes = (fixedRounded - fixedRaw) + uint64(c.Layers)*(perRounded-perRaw)
	return c
}
