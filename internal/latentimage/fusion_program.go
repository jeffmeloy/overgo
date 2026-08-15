// Krea2 text-fusion stream (text_fusion + txt_in) routed through the shared
// tensor graph. The SAME graph definition executes on both the reference backend
// (host golden) and the CUDA generic executor, so THIS file is the device port of
// the fusion -- the block arithmetic is defined once, from cataloged ops, and
// mirrors the host reference Denoiser.textConditioning (denoiser.go) op-for-op.
// This is the device-text-conditioning brick 3/3: the encoder (encoder_program.go)
// produces the selected hidden states [textSeq, TextLayers, TextHidden]; this
// program fuses them into the [textSeq, Hidden] conditioning that feeds the
// denoiser text stream (txt_in boundary in denoiser_program.go).
//
// Layout convention (matches the denoiser + encoder: every activation is
// [feature, tokens] with feature contiguous). The encoder output is fed as
// InEncoder shape (TextHidden, TextLayers*textSeq), layer-minor within each
// token (column c = tok*TextLayers + l), exactly the storage
// SelectedHiddenStates.Data already carries.
//
// Op mapping (host textConditioning -> cataloged op):
//   - layerwise blocks attend ACROSS the tapped-layer axis, batched over tokens:
//     a rank-4 attention [head_dim, heads, TextLayers, textSeq] (Dims[3] is the
//     per-token batch). No RoPE, no AdaLN modulation, sigmoid output gate.
//   - projector collapses the layer axis with Linear(TextLayers->1) weight [1,L]:
//     view [th, L*textSeq] as [L*th, textSeq], GroupSlice each layer's [th,textSeq]
//     and accumulate scaled by the layer weight -- transpose-free, all cataloged.
//   - refiner blocks attend ACROSS the token sequence (rank-3, batch collapses).
//   - txt_in: zero-centered RMSNorm -> linear_1 -> gelu(tanh) -> linear_2.
//
// Every norm is the DiT ZERO-CENTERED (1+w) RMSNorm (zeroCenteredRMSNorm, shared
// with the denoiser), NOT the encoder's standard WeightedRMSNorm. Geometry is
// DERIVED from TransformerSpec (config-cross-checked in verify.go); no magics.
package latentimage

import (
	"fmt"
	"math"

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

	weightInputs map[string]*tensor.Tensor

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
		"text_fusion.projector.weight": {1, t.TextLayers},
		"txt_in.norm.weight":           {th},
		"txt_in.linear_1.weight":       {h, th},
		"txt_in.linear_1.bias":         {h},
		"txt_in.linear_2.weight":       {h, h},
		"txt_in.linear_2.bias":         {h},
	}
	addFusionBlocks := func(kind string, n int) {
		for i := 0; i < n; i++ {
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
	if eps <= 0 {
		return nil, fmt.Errorf("fusion program: eps must be positive, got %g", eps)
	}
	if textSeq <= 0 {
		return nil, fmt.Errorf("fusion program: textSeq must be positive, got %d", textSeq)
	}
	if t.TextLayers <= 0 || t.TextHidden <= 0 || t.LayerwiseTextBlocks <= 0 || t.RefinerTextBlocks <= 0 {
		return nil, fmt.Errorf("fusion program: text-fusion geometry not derived (layers=%d hidden=%d layerwise=%d refiner=%d)",
			t.TextLayers, t.TextHidden, t.LayerwiseTextBlocks, t.RefinerTextBlocks)
	}
	if matmulType != dtype.F32 && matmulType != dtype.BF16 {
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
		weightInputs: make(map[string]*tensor.Tensor),
	}
	b := tensor.NewBuilder()
	setBuilderMatmulCompute(b, matmulType)
	bind := tensor.WeightInputs{Builder: b, Inputs: p.weightInputs, MatrixType: matmulType}

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
	for i := 0; i < t.LayerwiseTextBlocks; i++ {
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
	for l := uint64(0); l < L; l++ {
		slice := b.Reshape(b.GroupSlice(viewed, l*th, th, 1, th), th, ts) // (th, textSeq)
		term := b.Multiply(slice, b.FlatSlice(projW, l, 1))               // scale by proj[l]
		if fused == nil {
			fused = term
		} else {
			fused = b.Add(fused, term)
		}
	}

	// --- refiner blocks: attend across the token sequence (batch collapses) ---
	for i := 0; i < t.RefinerTextBlocks; i++ {
		prefix := fmt.Sprintf("text_fusion.refiner_blocks.%d.", i)
		fused = fusionAttnFF(b, bind, prefix, fused, p.keyBias, eps, scale, th, headDim, qHeads, kvHeads, qDim, kvDim, inter, ts, 1)
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

	if batch > 1 {
		q = b.Reshape(q, headDim, qHeads, seqLen, batch)
		k = b.Reshape(k, headDim, kvHeads, seqLen, batch)
		v = b.Reshape(v, headDim, kvHeads, seqLen, batch)
	}
	q, k, v = roundAttentionForStorage(b, bind.MatrixType, q, k, v)
	var attn *tensor.Tensor
	if keyBias != nil {
		attn = b.AttentionWithKeyBias(q, k, v, keyBias, scale, false)
	} else {
		attn = b.Attention(q, k, v, scale, false)
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
func (p *FusionProgram) RunHostFeed(run GraphRunner, weightAt FusionFeed, encoderHidden []float32) ([]float32, error) {
	if p.MatmulType != dtype.F32 {
		return nil, fmt.Errorf("fusion RunHostFeed: needs F32 weights, program compiled %s", p.MatmulType)
	}
	if want := p.TextSeq * p.T.TextLayers * p.T.TextHidden; len(encoderHidden) != want {
		return nil, fmt.Errorf("fusion RunHostFeed: encoder hidden len=%d want %d", len(encoderHidden), want)
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(p.weightInputs)+2)
	for name, node := range p.weightInputs {
		data, err := weightAt(name)
		if err != nil {
			return nil, fmt.Errorf("fusion RunHostFeed: weight %s: %w", name, err)
		}
		elements, _ := node.Shape.Elements()
		if uint64(len(data)) != elements {
			return nil, fmt.Errorf("fusion RunHostFeed: weight %s len=%d want %d", name, len(data), elements)
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: data}
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
