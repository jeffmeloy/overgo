// Text-fusion graph. Reference and CUDA share topology.
// Input: [text_hidden,text_layers*text_tokens], layer-minor per token.
// Output: [hidden,text_tokens]. Geometry comes from TransformerSpec.

package latentimage

import (
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// FusionProgram: the compiled text-fusion graph for one prompt length. Weight
// inputs are named exactly as FusionTensorShapes; rank-2 projection weights take
// matmulType storage (BF16 for the device tensor-core path), norms/biases and the
// projector vector stay F32.
type FusionProgram struct {
	T          TransformerSpec
	Eps        float32
	TextSeq    int
	MatmulType dtype.Type

	// InEncoder (TextHidden, TextLayers*TextSeq): the selected hiddens, layer-minor
	// within each token (column = tok*TextLayers + l). Host-fed f32.
	InEncoder *tensor.Tensor
	keyBias   *tensor.Tensor
	keyData   []float32

	weightInputs tensor.WeightBindings

	// Fused (Hidden, TextSeq): the text conditioning feeding the denoiser text
	// stream (image-token co-attention).
	Fused *tensor.Tensor
}

// FusionTensorShapes returns the name -> torch shape ([out,in] for Linear
// weights) manifest the fusion forward consumes: the layerwise + refiner blocks
// at TextHidden, the layer projector, and txt_in. Derived entirely from spec.
// This is the capability-retention contract for the text-fusion stream (a subset
// of DenoiserTensorShapes -- the same blocks, minus the 28 image co-attention
// blocks and the timestep/img_in/final tensors the denoiser step owns).
func FusionTensorShapes(t TransformerSpec) map[string][]int {
	h := t.Hidden
	th := t.TextHidden
	shapes := map[string][]int{
		"text_fusion.projector.weight": {tensor.SingletonExtent, t.TextLayers},
		"txt_in.norm.weight":           {th},
		"txt_in.linear_1.weight":       {h, th},
		"txt_in.linear_1.bias":         {h},
		"txt_in.linear_2.weight":       {h, h},
		"txt_in.linear_2.bias":         {h},
	}
	addFusionBlocks := func(kind string, n int) {
		for i := range n {
			p := fmt.Sprintf("text_fusion.%s.%d.", kind, i)
			addAttnFF(shapes, p, th, t.TextHeads*t.HeadDim, t.TextKVHeads*t.HeadDim, t.HeadDim, t.TextIntermediate, false)
		}
	}
	addFusionBlocks("layerwise_blocks", t.LayerwiseTextBlocks)
	addFusionBlocks("refiner_blocks", t.RefinerTextBlocks)
	return shapes
}

// CompileFusionProgram builds the text-fusion graph for a sequence of textSeq
// prompt tokens. matmulType selects rank-2 weight storage (dtype.F32 exact
// reference/CUDA parity path, dtype.BF16 device resident path).
func CompileFusionProgram(t TransformerSpec, eps float32, textMask []bool, matmulType dtype.Type) (*FusionProgram, error) {
	textSeq := len(textMask)
	if !checked.PositiveFinite32(eps) {
		return nil, fmt.Errorf("fusion program: eps must be positive, got %g", eps)
	}
	if !checked.PositiveInts(textSeq) {
		return nil, fmt.Errorf("fusion program: textSeq must be positive, got %d", textSeq)
	}
	if !checked.PositiveInts(t.TextLayers, t.TextHidden, t.LayerwiseTextBlocks, t.RefinerTextBlocks) {
		return nil, fmt.Errorf("fusion program: text-fusion geometry not derived (layers=%d hidden=%d layerwise=%d refiner=%d)",
			t.TextLayers, t.TextHidden, t.LayerwiseTextBlocks, t.RefinerTextBlocks)
	}
	if !slices.Contains([]dtype.Type{dtype.F32, dtype.BF16}, matmulType) {
		return nil, fmt.Errorf("fusion program: matmul weight type %s unsupported", matmulType)
	}

	L := uint64(t.TextLayers)
	th := uint64(t.TextHidden)
	h := uint64(t.Hidden)
	headDim := uint64(t.HeadDim)
	qHeads := uint64(t.TextHeads)
	kvHeads := uint64(t.TextKVHeads)
	qDim := qHeads * headDim
	kvDim := kvHeads * headDim
	inter := uint64(t.TextIntermediate)
	ts := uint64(textSeq)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	p := &FusionProgram{
		T: t, Eps: eps, TextSeq: textSeq, MatmulType: matmulType,
	}
	b := tensor.NewBuilder()
	setBuilderMatmulCompute(b, matmulType)
	bind := tensor.WeightInputs{Builder: b, Bindings: &p.weightInputs, MatrixType: matmulType}

	p.InEncoder = b.Input("encoder_hidden", dtype.F32, tensor.MustShape(th, L*ts))
	for _, attended := range textMask {
		if !attended {
			p.keyBias = b.Input("fusion_key_bias", dtype.F32, tensor.MustShape(ts))
			p.keyData = make([]float32, textSeq)
			for index, active := range textMask {
				if !active {
					p.keyData[index] = padKeyBias
				}
			}
			break
		}
	}

	// --- layerwise blocks: attend across the tapped-layer axis, per token ---
	// sequence = TextLayers, batch = textSeq (rank-4 attention).
	hidden := p.InEncoder
	for i := range t.LayerwiseTextBlocks {
		prefix := fmt.Sprintf("text_fusion.layerwise_blocks.%d.", i)
		hidden = fusionAttnFF(b, bind, prefix, hidden, nil, eps, scale, th, headDim, qHeads, kvHeads, qDim, kvDim, inter, L, ts)
	}

	// --- projector: collapse the layer axis, Linear(TextLayers->1) weight [1,L] ---
	// view [th, L*textSeq] as [L*th, textSeq]; layer l occupies the contiguous
	// Dims[0] span [l*th, (l+1)*th). Accumulate each layer's [th,textSeq] scaled
	// by the projector weight -- no transpose needed.
	projW := bind.Input("text_fusion.projector.weight", L) // F32 (L)
	viewed := b.Reshape(hidden, L*th, ts)
	var fused *tensor.Tensor
	for l := range L {
		slice := b.Reshape(b.GroupSlice(viewed, l*th, th, tensor.SingletonExtent, th), th, ts) // (th, textSeq)
		term := b.Multiply(slice, b.FlatSlice(projW, l, tensor.SingletonExtent))               // scale by proj[l]
		if fused == nil {
			fused = term
		} else {
			fused = b.Add(fused, term)
		}
	}

	// --- refiner blocks: attend across the token sequence (batch collapses) ---
	for i := range t.RefinerTextBlocks {
		prefix := fmt.Sprintf("text_fusion.refiner_blocks.%d.", i)
		fused = fusionAttnFF(b, bind, prefix, fused, p.keyBias, eps, scale, th, headDim, qHeads, kvHeads, qDim, kvDim, inter, ts, tensor.SingletonExtent)
	}

	// --- txt_in: zero-centered norm -> linear_1 -> gelu(tanh) -> linear_2 ---
	normed := zeroCenteredRMSNorm(b, fused, bind.Input("txt_in.norm.weight", th), eps)
	l1 := b.Add(
		b.MulMat(bind.Input("txt_in.linear_1.weight", th, h), normed),
		bind.Input("txt_in.linear_1.bias", h),
	)
	l1 = b.GELUTanhExact(l1)
	p.Fused = b.Add(
		b.MulMat(bind.Input("txt_in.linear_2.weight", h, h), l1),
		bind.Input("txt_in.linear_2.bias", h),
	)

	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("fusion program graph: %w", err)
	}
	return p, nil
}

// fusionAttnFF runs one Krea2TextFusionBlock (pre-norm, gated GQA attention, no
// RoPE, no modulation, then SwiGLU FF). hidden is [th, seqLen*batch]. When
// batch>1 the attention runs rank-4 [head_dim, heads, seqLen, batch] (per-batch
// bidirectional); when batch==1 it collapses to rank-3. Mirrors host fusionBlock.
func fusionAttnFF(
	b *tensor.Builder, bind tensor.WeightInputs, prefix string, hidden, keyBias *tensor.Tensor,
	eps, scale float32,
	th, headDim, qHeads, kvHeads, qDim, kvDim, inter, seqLen, batch uint64,
) *tensor.Tensor {
	cols := seqLen * batch

	n1 := zeroCenteredRMSNorm(b, hidden, bind.Input(prefix+"norm1.weight", th), eps)
	q := b.MulMat(bind.Input(prefix+"attn.to_q.weight", th, qDim), n1)
	k := b.MulMat(bind.Input(prefix+"attn.to_k.weight", th, kvDim), n1)
	v := b.MulMat(bind.Input(prefix+"attn.to_v.weight", th, kvDim), n1)
	gate := b.MulMat(bind.Input(prefix+"attn.to_gate.weight", th, th), n1)

	// per-head zero-centered q/k RMSNorm over head_dim.
	q = b.Reshape(q, headDim, qHeads, cols)
	k = b.Reshape(k, headDim, kvHeads, cols)
	v = b.Reshape(v, headDim, kvHeads, cols)
	q = zeroCenteredRMSNorm(b, q, bind.Input(prefix+"attn.norm_q.weight", headDim), eps)
	k = zeroCenteredRMSNorm(b, k, bind.Input(prefix+"attn.norm_k.weight", headDim), eps)
	// NO RoPE, NO AdaLN modulation.

	if checked.Multiple(int(batch)) {
		q = b.Reshape(q, headDim, qHeads, seqLen, batch)
		k = b.Reshape(k, headDim, kvHeads, seqLen, batch)
		v = b.Reshape(v, headDim, kvHeads, seqLen, batch)
	}
	q, k, v = roundAttentionForStorage(b, bind.MatrixType, q, k, v)
	var attn *tensor.Tensor
	if keyBias != nil {
		attn = b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{KeyBias: keyBias, Scale: scale, Causal: false})
	} else {
		attn = b.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: scale, Causal: false})
	}
	attn = b.Reshape(attn, th, cols)
	attn = b.Multiply(attn, b.Sigmoid(gate)) // sigmoid output gate
	attn = b.MulMat(bind.Input(prefix+"attn.to_out.0.weight", th, th), attn)
	hidden = b.Add(hidden, attn)

	n2 := zeroCenteredRMSNorm(b, hidden, bind.Input(prefix+"norm2.weight", th), eps)
	g := b.MulMat(bind.Input(prefix+"ff.gate.weight", th, inter), n2)
	u := b.MulMat(bind.Input(prefix+"ff.up.weight", th, inter), n2)
	ff := b.MulMat(bind.Input(prefix+"ff.down.weight", inter, th), b.SwiGLU(g, u))
	return b.Add(hidden, ff)
}

// FusionFeed is a name->F32 weight lookup for the host-feed reference/CUDA path.
type FusionFeed func(name string) ([]float32, error)

// RunHostFeed executes the fusion graph through run with F32 weights supplied by
// weightAt and the selected hiddens encoderHidden ([textSeq*TextLayers*TextHidden],
// the SelectedHiddenStates.Data layout, token-major then layer then feature). This
// is the exact-parity path (F32 weights): reference.Execute gives the host golden,
// the CUDA generic executor gives the device match. Returns the fused conditioning
// [textSeq*Hidden] (token-major), the layout the denoiser text stream consumes.
func (p *FusionProgram) RunHostFeed(run graphruntime.Runner, weightAt FusionFeed, encoderHidden []float32) ([]float32, error) {
	if p.MatmulType != dtype.F32 {
		return nil, fmt.Errorf("fusion RunHostFeed: needs F32 weights, program compiled %s", p.MatmulType)
	}
	if want := p.TextSeq * p.T.TextLayers * p.T.TextHidden; len(encoderHidden) != want {
		return nil, fmt.Errorf("fusion RunHostFeed: encoder hidden len=%d want %d", len(encoderHidden), want)
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(p.weightInputs)+2)
	if err := graphruntime.AddHostWeights(feeds, p.weightInputs, weightAt); err != nil {
		return nil, fmt.Errorf("fusion RunHostFeed: %w", err)
	}
	feeds[p.InEncoder] = reference.Value{Shape: p.InEncoder.Shape, Data: encoderHidden}
	if p.keyBias != nil {
		feeds[p.keyBias] = reference.Value{Shape: p.keyBias.Shape, Data: p.keyData}
	}

	results, err := run([]*tensor.Tensor{p.Fused}, feeds)
	if err != nil {
		return nil, fmt.Errorf("fusion RunHostFeed: %w", err)
	}
	return p.assemble(results)
}

// assemble extracts the fused conditioning [textSeq*Hidden] (token-major) from the
// graph outputs.
func (p *FusionProgram) assemble(results map[*tensor.Tensor]reference.Value) ([]float32, error) {
	val, ok := results[p.Fused]
	if !ok {
		return nil, fmt.Errorf("fusion assemble: missing fused output")
	}
	if want := p.TextSeq * p.T.Hidden; len(val.Data) != want {
		return nil, fmt.Errorf("fusion assemble: fused len=%d want %d", len(val.Data), want)
	}
	return append([]float32(nil), val.Data...), nil
}
