//go:build windows

package devicemath

import (
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

type residentStackBuffers struct {
	xIn []driver.DevicePtr
	cp  []layerCachePtrs
	dX  []driver.DevicePtr
}

// runStackDevice executes the whole pre-norm stack inside the current
// session: forward over all layers (activation caches resident), tail on the
// final pre-norm hidden states, then backward over all layers writing weight
// gradients into gp. It returns the device pointer of the embedding-input
// gradient and retains the compiled stack buffers for its session.
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
			if err = o.s.fill([]deviceSlot{{&o.stack.xIn[i+1], nil, seqHidden}, {&o.stack.dX[i], nil, seqHidden}}); err != nil {
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
