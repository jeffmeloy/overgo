package hostmath

// Multi-layer composition of the qwen3.5 hybrid decoder layer (hybrid_layer.go):
// an N-layer stack forward and its VJP, chaining the single-layer
// HybridDecoderLayerForward/Backward. The per-layer forward caches are opaque to
// callers (hybridLayerCache is package-private) -- HybridStackForward returns them
// and HybridStackBackward consumes them, so a caller threads them straight through
// without naming the type. This is the host reference the device resident hybrid
// training loop is verified against.

// HybridStackForward runs the N-layer hybrid stack from input x, returning the final
// layer output, the per-layer inputs (input[i] is layer i's input = layer i-1's
// output, input[0]=x) and the per-layer forward caches for the backward. states[i]
// is layer i's GDN input state (nil for full_attention layers).
func HybridStackForward(x []float32, weights []HybridLayerWeights, dims []HybridLayerDims, states [][]float32) (out []float32, inputs [][]float32, caches []hybridLayerCache) {
	inputs = make([][]float32, len(weights))
	caches = make([]hybridLayerCache, len(weights))
	cur := x
	for i := range weights {
		inputs[i] = cur
		o, c := HybridDecoderLayerForward(cur, weights[i], dims[i], states[i])
		caches[i] = c
		cur = o
	}
	return cur, inputs, caches
}

// HybridStackBackward is the VJP of HybridStackForward: given the top cotangent
// dTop (d loss / d final output) and the forward inputs/caches, it walks the layers
// in reverse, feeding each layer's DX up as the next (lower) layer's output
// cotangent. Returns the per-layer grads (grads[i] for layer i) and dX, the
// cotangent w.r.t. the stack input x.
func HybridStackBackward(inputs [][]float32, weights []HybridLayerWeights, dims []HybridLayerDims, states [][]float32, dTop []float32, caches []hybridLayerCache) (grads []HybridDecoderLayerGrads, dX []float32) {
	grads = make([]HybridDecoderLayerGrads, len(weights))
	dOut := dTop
	for i := len(weights) - 1; i >= 0; i-- {
		g := HybridDecoderLayerBackward(inputs[i], weights[i], dims[i], states[i], dOut, caches[i])
		grads[i] = g
		dOut = g.DX
	}
	return grads, dOut
}
