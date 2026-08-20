package hostmath

// HybridStackTrace: opaque forward state consumed by the stack VJP.
type HybridStackTrace struct {
	inputs [][]float32
	caches []hybridLayerCache
}

// HybridStackForward returns output and VJP trace.
func HybridStackForward(x []float32, weights []HybridLayerWeights, dims []HybridLayerDims, states [][]float32) (out []float32, trace HybridStackTrace) {
	trace.inputs = make([][]float32, len(weights))
	trace.caches = make([]hybridLayerCache, len(weights))
	cur := x
	for i := range weights {
		trace.inputs[i] = cur
		o, c := HybridDecoderLayerForward(cur, weights[i], dims[i], states[i])
		trace.caches[i] = c
		cur = o
	}
	return cur, trace
}

// HybridStackBackwardStreamed walks the trace in reverse, handing each layer's
// gradients to visit as they complete and releasing that layer's trace before
// moving inward, so gradient residency is bounded by one layer, never the
// stack. Update ordering contract: a layer's DX and weight gradients are both
// computed here from its PRE-step weights BEFORE visit runs, so a visitor that
// steps the layer's weights in place cannot perturb the remaining backward —
// layers below consume only DX and their own saved activations, never this
// layer's weights — and the trajectory equals the full-slab lane's exactly.
func HybridStackBackwardStreamed(trace *HybridStackTrace, weights []HybridLayerWeights, dims []HybridLayerDims, states [][]float32, dTop []float32, visit func(layer int, g HybridDecoderLayerGrads) error) ([]float32, error) {
	dOut := dTop
	for i := len(weights) - 1; i >= 0; i-- {
		g := HybridDecoderLayerBackward(trace.inputs[i], weights[i], dims[i], states[i], dOut, trace.caches[i])
		trace.inputs[i] = nil
		trace.caches[i] = hybridLayerCache{}
		if err := visit(i, g); err != nil {
			return nil, err
		}
		dOut = g.DX
	}
	return dOut, nil
}

// HybridStackBackward reuses the caller-owned gradient index.
func HybridStackBackward(trace HybridStackTrace, weights []HybridLayerWeights, dims []HybridLayerDims, states [][]float32, dTop []float32, grads []HybridDecoderLayerGrads) ([]HybridDecoderLayerGrads, []float32) {
	if cap(grads) < len(weights) {
		grads = make([]HybridDecoderLayerGrads, len(weights))
	} else {
		grads = grads[:len(weights)]
	}
	dOut := dTop
	for i := len(weights) - 1; i >= 0; i-- {
		g := HybridDecoderLayerBackward(trace.inputs[i], weights[i], dims[i], states[i], dOut, trace.caches[i])
		grads[i] = g
		dOut = g.DX
	}
	return grads, dOut
}
