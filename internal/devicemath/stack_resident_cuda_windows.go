//go:build windows

package devicemath

import (
	"fmt"
	"slices"

	"overgo/internal/cuda/driver"
)

// stackFnNames is the deduped union of the forward and backward kernel sets, so
// one session loads every function the whole-stack driver launches.
var stackFnNames = func() []string {
	seen := map[string]bool{}
	var out []string
	for _, name := range append(slices.Clone(layerForwardFnNames), layerBackwardFnNames...) {
		if !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	return out
}()

// runStack executes the whole pre-norm stack inside the current session: forward
// over all layers (activation caches resident), a host round-trip through tail on
// the streamed-out final pre-norm, then backward over all layers writing weight
// grads into gp. The weight pointers wp and grad pointers gp are supplied by the
// caller, persistent across steps (StackForwardBackwardResidentWeights). Returns
// the device pointer for dX after layer 0 (the embedding-input gradient); the
// caller downloads what it needs. The layer math is forwardDevice/backwardDevice
// (single owner).
func (o *layerOps) runStack(embeds []float32, wp []layerWeightPtrs, gp []layerGradPtrs, tail func(final []float32) ([]float32, error)) (driver.DevicePtr, error) {
	input, err := o.s.upload(embeds)
	if err != nil {
		return 0, err
	}
	return o.runStackDevice(input, wp, gp, func(final driver.DevicePtr) (driver.DevicePtr, error) {
		host := make([]float32, o.d.seq*o.d.hidden)
		if err := o.s.finish(cudaDownload{host, final}); err != nil {
			return 0, err
		}
		gradient, err := tail(host)
		if err != nil {
			return 0, err
		}
		if len(gradient) != len(host) {
			return 0, fmt.Errorf("runStack: tail dOut len %d != %d", len(gradient), len(host))
		}
		return o.s.upload(gradient)
	})
}

type residentStackBuffers struct {
	xIn []driver.DevicePtr
	cp  []layerCachePtrs
	dX  []driver.DevicePtr
}

// runStackDevice retains the compiled stack buffers for its session.
func (o *layerOps) runStackDevice(input driver.DevicePtr, wp []layerWeightPtrs, gp []layerGradPtrs, tail func(driver.DevicePtr) (driver.DevicePtr, error)) (driver.DevicePtr, error) {
	d, nL := o.d, len(wp)
	seqHidden := d.seq * d.hidden
	if o.arena == nil {
		base, err := o.s.alloc(maxScratchElems(d))
		if err != nil {
			return 0, err
		}
		o.arena = &scratchArena{base: base, capElems: maxScratchElems(d)}
	}
	if o.stack == nil {
		o.stack = &residentStackBuffers{xIn: make([]driver.DevicePtr, nL+1), cp: make([]layerCachePtrs, nL), dX: make([]driver.DevicePtr, nL)}
		var err error
		for i := range nL {
			if o.stack.cp[i], err = allocLayerCache(o.s.cudaScope, d); err != nil {
				return 0, err
			}
			if o.stack.xIn[i+1], err = o.s.alloc(seqHidden); err != nil {
				return 0, err
			}
			if o.stack.dX[i], err = o.s.alloc(seqHidden); err != nil {
				return 0, err
			}
		}
	}
	xIn, cp := o.stack.xIn, o.stack.cp
	xIn[0] = input
	for i := range nL {
		if err := o.forwardDevice(xIn[i], wp[i], cp[i], xIn[i+1]); err != nil {
			return 0, err
		}
		o.arena.reset()
	}
	dOutP, err := tail(xIn[nL])
	if err != nil {
		return 0, err
	}
	for i := nL - 1; i >= 0; i-- {
		dXP := o.stack.dX[i]
		if err := o.backwardDevice(xIn[i], dOutP, cp[i], wp[i], gp[i], dXP); err != nil {
			return 0, err
		}
		o.arena.reset()
		dOutP = dXP
	}
	return dOutP, nil
}
