//go:build windows

package devicemath

import (
	"fmt"
	"slices"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// LayerBackwardResult bundles a layer's input-gradient and weight-gradients
// (densecausal layout), returned by the whole-stack resident driver.
type LayerBackwardResult struct {
	DX                     []float32
	DWInLN, DWPostLN       []float32
	DWQ, DWK, DWV, DWO     []float32
	DQBias, DKBias, DVBias []float32
	DWGate, DWUp, DWDown   []float32
}

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

func (w LayerForwardWeights) shapeOK(d layerDims) bool {
	biasOK := w.QBias == nil && w.KBias == nil && w.VBias == nil ||
		len(w.QBias) == d.width && len(w.KBias) == d.kvWidth && len(w.VBias) == d.kvWidth
	return biasOK && len(w.InLN) == d.hidden && len(w.PostLN) == d.hidden &&
		len(w.Q) == d.width*d.hidden && len(w.K) == d.kvWidth*d.hidden && len(w.V) == d.kvWidth*d.hidden &&
		len(w.O) == d.hidden*d.width && len(w.Gate) == d.inter*d.hidden && len(w.Up) == d.inter*d.hidden &&
		len(w.Down) == d.hidden*d.inter
}

// StackForwardBackwardResident runs the WHOLE pre-norm layer stack forward then
// backward inside ONE cudaBLAS session. Every layer's weights upload once and
// serve both its forward and its backward; every layer's forward activation cache
// stays device-resident and is consumed by that layer's backward with NO host
// download/reupload in between. The only host round-trips are: the final pre-norm
// stream out (for the host head/norm/softmax-CE tail) and the tail's returned
// output-gradient back in. `tail(final)` computes loss/logits/head+norm backward
// on host and returns dOut for the last layer. Returns the embedding-input
// gradient (dX after layer 0) and each layer's weight grads. Attention bias is
// not supported (reject upstream). Same math as the per-layer resident path.
func StackForwardBackwardResident(
	worker *device.Worker,
	embeds []float32,
	layers []LayerForwardWeights,
	invFreq []float32,
	seq, hidden, heads, kvHeads, hd, inter int,
	rmsEps float64,
	tail func(final []float32) ([]float32, error),
) ([]float32, []LayerBackwardResult, error) {
	d := newLayerDims(seq, hidden, heads, kvHeads, hd, inter, rmsEps)
	nL := len(layers)
	if nL == 0 || heads%kvHeads != 0 || len(embeds) != seq*hidden || len(invFreq) != hd/2 {
		return nil, nil, fmt.Errorf("StackForwardBackwardResident: shape mismatch (layers=%d seq=%d hidden=%d heads=%d kv=%d hd=%d)", nL, seq, hidden, heads, kvHeads, hd)
	}
	for i, w := range layers {
		if !w.shapeOK(d) {
			return nil, nil, fmt.Errorf("StackForwardBackwardResident: layer %d weight shape mismatch", i)
		}
	}

	grads := make([]LayerBackwardResult, nL)
	for i := range grads {
		grads[i].DWInLN = make([]float32, hidden)
		grads[i].DWPostLN = make([]float32, hidden)
		grads[i].DWQ = make([]float32, d.width*hidden)
		grads[i].DWK = make([]float32, d.kvWidth*hidden)
		grads[i].DWV = make([]float32, d.kvWidth*hidden)
		if layers[i].QBias != nil {
			grads[i].DQBias = make([]float32, d.width)
			grads[i].DKBias = make([]float32, d.kvWidth)
			grads[i].DVBias = make([]float32, d.kvWidth)
		}
		grads[i].DWO = make([]float32, hidden*d.width)
		grads[i].DWGate = make([]float32, inter*hidden)
		grads[i].DWUp = make([]float32, inter*hidden)
		grads[i].DWDown = make([]float32, hidden*inter)
	}
	dxEmbed := make([]float32, seq*hidden)

	err := withCUDABLAS(worker, func(s *cudaBLAS) error {
		ops, err := newLayerOps(s, stackFnNames, d, invFreq)
		if err != nil {
			return err
		}
		// Weights upload once, grad buffers allocate in-session; both feed runStack.
		wp := make([]layerWeightPtrs, nL)
		gp := make([]layerGradPtrs, nL)
		for i := range nL {
			if wp[i], err = uploadLayerWeights(s.cudaScope, layers[i]); err != nil {
				return err
			}
			if gp[i], err = allocLayerGrads(s.cudaScope, d); err != nil {
				return err
			}
			if err = allocLayerBiasGrads(s.cudaScope, &gp[i], d, layers[i].QBias != nil); err != nil {
				return err
			}
		}
		dOutP, err := ops.runStack(embeds, wp, gp, tail)
		if err != nil {
			return err
		}
		downloads := make([]cudaDownload, 0, nL*9+1)
		for i := range nL {
			downloads = append(downloads,
				cudaDownload{grads[i].DWInLN, gp[i].dInLN}, cudaDownload{grads[i].DWPostLN, gp[i].dPostLN},
				cudaDownload{grads[i].DWQ, gp[i].dQ}, cudaDownload{grads[i].DWK, gp[i].dK}, cudaDownload{grads[i].DWV, gp[i].dV},
				cudaDownload{grads[i].DWO, gp[i].dO}, cudaDownload{grads[i].DWGate, gp[i].dGate},
				cudaDownload{grads[i].DWUp, gp[i].dUp}, cudaDownload{grads[i].DWDown, gp[i].dDown},
			)
			if layers[i].QBias != nil {
				downloads = append(downloads,
					cudaDownload{grads[i].DQBias, gp[i].dQBias}, cudaDownload{grads[i].DKBias, gp[i].dKBias}, cudaDownload{grads[i].DVBias, gp[i].dVBias})
			}
		}
		downloads = append(downloads, cudaDownload{dxEmbed, dOutP})
		return s.finish(downloads...)
	})
	if err != nil {
		return nil, nil, err
	}
	return dxEmbed, grads, nil
}

// runStack executes the whole pre-norm stack inside the current session: forward
// over all layers (activation caches resident), a host round-trip through tail on
// the streamed-out final pre-norm, then backward over all layers writing weight
// grads into gp. The weight pointers wp and grad pointers gp are supplied by the
// caller -- session-local (StackForwardBackwardResident) or persistent-across-steps
// (StackForwardBackwardResidentWeights) -- so this one driver owns the fwd/bwd
// composition regardless of where the buffers live. Returns the device pointer for
// dX after layer 0 (the embedding-input gradient); the caller downloads what it
// needs. The layer math is forwardDevice/backwardDevice (single owner).
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
