// Qwen3-VL selected-layer text encoder routed through the shared tensor graph.
// The SAME graph definition executes on both the reference backend (host golden)
// and the CUDA generic executor, so THIS file is the device port of the encoder
// -- the block arithmetic is defined once, from cataloged ops, and mirrors the
// host reference encodeSelected/encLayerForward (textencoder.go) op-for-op. This
// is the device-text-conditioning brick 2/3: textencoder.go is a HOST f64
// telemetry oracle; the golden serve runs the encoder device bf16, and this
// program is that device path.
//
// Layout convention (matches the denoiser + MulMat: left[in,out] row-major torch
// [out,in], right[in,tokens] token-major): every activation is [feature, tokens]
// with token-major data, so WeightedRMSNorm over Dims[0] is per-token, and a q/k
// tensor reshaped to [head_dim, heads, tokens] gets per-head normalization for
// free.
//
// Op mapping (host encLayerForward -> cataloged op), all verified against the
// reference backend:
//   - STANDARD RMSNorm x/sqrt(mean(x^2)+eps)*w  -> WeightedRMSNorm (NOT the DiT's
//     zero-centered (1+w) form).
//   - per-head q/k RMSNorm over head_dim       -> reshape [head_dim,heads,tok] +
//     WeightedRMSNorm over Dims[0].
//   - rotate-half (split-half) RoPE, theta 5e6 -> RoPENeoX over the full head_dim
//     with sequential token positions (Qwen3-VL mrope collapses to standard rope
//     for a pure-text sequence).
//   - causal GQA, scale 1/sqrt(head_dim)       -> Attention(causal=true) (GQA-
//     native: heads q, kvHeads kv).
//   - SwiGLU down(silu(gate)*up)               -> SwiGLU + MulMat.
//
// The giant embed_tokens table is NOT a graph weight: the caller gathers the
// per-token embedding rows host-side (readEmbedRows) and feeds them as the small
// [Hidden, seq] Embed input, exactly the split the denoiser uses for its latent
// patches. Only the tapped layers run (layers 0..max(SelectLayers)-1).
//
// ATTENTION MASK residual: the dtc-tokenizer (textinput.go) renders a padded
// [prefix][prompt][pad][suffix] sequence whose pad rows are unattended KEYS. The
// cataloged Attention op has causal masking but no arbitrary per-key pad mask, so
// this graph runs plain causal over the fed token rows -- IDENTICAL to the host
// textencoder.go reference, which also runs maskless causal. That makes the
// device==host bf16-band parity below exact and clean, and leaves pad-key
// exclusion (needed only for element-exact match to the adaptive golden) as the
// documented residual for the full e2e SHA; the telemetry oracle stays until then.
package latentimage

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// EncoderProgram: the compiled selected-layer encoder graph for one prompt
// length. Weight inputs are named exactly as EncoderTensorShapes; rank-2
// projection weights take MatmulType storage (BF16 for the device tensor-core
// path), norms stay F32.
type EncoderProgram struct {
	E          TextEncoderSpec
	Eps        float32
	Seq        int
	MatmulType dtype.Type

	Embed *tensor.Tensor // (Hidden, seq)  host-fed per-token embedding rows

	// keyBias, when non-nil, is a graph Input [seq] carrying the additive per-key
	// pad mask (0.0 attended, padKeyBias for a pad KEY). keyBiasData holds
	// its constant per-prompt values, fed automatically by RunHostFeed / the
	// resident Encode. Absent (nil) => the maskless-causal program (unchanged).
	keyBias     *tensor.Tensor
	keyBiasData []float32

	weightInputs map[string]*tensor.Tensor

	// Selected[i] is the residual stream captured after decoder layer
	// SelectLayers[i]-1 ([Hidden, seq], token-major). CaptureAfter[i] is that
	// 0-based decoder layer, for reporting.
	Selected     []*tensor.Tensor
	CaptureAfter []int
}

// encWeightBinder binds named weight inputs; rank-2 projections take matmulType.
type encWeightBinder struct {
	builder    *tensor.Builder
	inputs     map[string]*tensor.Tensor
	matmulType dtype.Type
}

// input declares a weight tensor with the MulMat storage convention: a torch
// Linear weight [out,in] is declared as shape (in,out); rank-1 tensors stay F32.
func (b encWeightBinder) input(name string, dimensions ...uint64) *tensor.Tensor {
	storage := dtype.F32
	if len(dimensions) == 2 {
		storage = b.matmulType
	}
	node := b.builder.Input(name, storage, tensor.MustShape(dimensions...))
	b.inputs[name] = node
	return node
}

// padKeyBias is the additive score bias for a pad KEY: a query's dot with
// a pad key is driven to ~-inf so it underflows to 0 probability. Mirrors the
// adaptive runtime_causal_gqa_masked_bf16 sentinel (-3.402823466e38) so the
// masked softmax result is identical to adaptive's replace-with-lowest form.
const padKeyBias = float32(-3.402823466e38)

// CompileEncoderProgram builds the maskless-causal Qwen3-VL selected-layer
// encoder graph for a sequence of seq tokens (unchanged behavior). matmulType
// selects rank-2 weight storage (dtype.F32 exact reference/CUDA parity path,
// dtype.BF16 device resident path).
func CompileEncoderProgram(e TextEncoderSpec, eps float32, seq int, matmulType dtype.Type) (*EncoderProgram, error) {
	return compileEncoderProgram(e, eps, seq, matmulType, nil)
}

// CompileEncoderProgramMasked builds the encoder graph with an additive per-key
// pad mask applied in the causal GQA: mask[i]==false marks token i as an
// unattended KEY (the Krea [prefix][prompt][pad][suffix] pad region). len(mask)
// must equal seq. This makes the selected hidden states golden-exact against the
// adaptive masked encoder (runtime_causal_gqa_masked_bf16). The same graph runs
// on the reference backend (host oracle) and the CUDA generic executor (device).
func CompileEncoderProgramMasked(e TextEncoderSpec, eps float32, seq int, matmulType dtype.Type, mask []bool) (*EncoderProgram, error) {
	if len(mask) != seq {
		return nil, fmt.Errorf("encoder program: mask len=%d want seq=%d", len(mask), seq)
	}
	return compileEncoderProgram(e, eps, seq, matmulType, mask)
}

// compileEncoderProgram is the shared builder; mask==nil => maskless causal.
func compileEncoderProgram(e TextEncoderSpec, eps float32, seq int, matmulType dtype.Type, mask []bool) (*EncoderProgram, error) {
	if eps <= 0 {
		return nil, fmt.Errorf("encoder program: eps must be positive, got %g", eps)
	}
	if seq <= 0 {
		return nil, fmt.Errorf("encoder program: seq must be positive, got %d", seq)
	}
	if e.Intermediate <= 0 {
		return nil, fmt.Errorf("encoder program: intermediate must be positive, got %d", e.Intermediate)
	}
	if matmulType != dtype.F32 && matmulType != dtype.BF16 {
		return nil, fmt.Errorf("encoder program: matmul weight type %s unsupported", matmulType)
	}
	slots, err := captureSlots(e)
	if err != nil {
		return nil, err
	}

	h := uint64(e.Hidden)
	headDim := uint64(e.HeadDim)
	heads := uint64(e.Heads)
	kvHeads := uint64(e.KVHeads)
	qDim := heads * headDim
	kvDim := kvHeads * headDim
	inter := uint64(e.Intermediate)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	theta := float32(e.RopeTheta)

	// sequential token positions -- Qwen3-VL mrope collapses to standard rope for
	// a pure-text sequence (all three position sections share the token index).
	positions := make([]uint32, seq)
	for i := range positions {
		positions[i] = uint32(i)
	}

	p := &EncoderProgram{
		E: e, Eps: eps, Seq: seq, MatmulType: matmulType,
		weightInputs: make(map[string]*tensor.Tensor),
	}
	b := tensor.NewBuilder()
	bind := encWeightBinder{builder: b, inputs: p.weightInputs, matmulType: matmulType}

	p.Embed = b.Input("encoder_embed", dtype.F32, tensor.MustShape(h, uint64(seq)))
	hidden := p.Embed

	// Optional additive per-key pad mask: build the [seq] bias input + its
	// constant values (0.0 attended, padKeyBias for a pad key).
	if mask != nil {
		p.keyBias = b.Input("encoder_key_bias", dtype.F32, tensor.MustShape(uint64(seq)))
		p.keyBiasData = make([]float32, seq)
		for i, attended := range mask {
			if !attended {
				p.keyBiasData[i] = padKeyBias
			}
		}
	}

	// selected outputs, ordered by their capture slot.
	p.Selected = make([]*tensor.Tensor, len(e.SelectLayers))
	p.CaptureAfter = make([]int, len(e.SelectLayers))
	maxNeeded := e.SelectLayers[len(e.SelectLayers)-1]

	for l := 0; l < e.HiddenLayers && l < maxNeeded; l++ {
		prefix := fmt.Sprintf("%slayers.%d.", textEncoderPrefix, l)

		// --- causal GQA self-attention (standard RMSNorm, per-head q/k norm) ---
		n1 := b.WeightedRMSNorm(hidden, bind.input(prefix+"input_layernorm.weight", h), eps)
		q := b.MulMat(bind.input(prefix+"self_attn.q_proj.weight", h, qDim), n1)
		k := b.MulMat(bind.input(prefix+"self_attn.k_proj.weight", h, kvDim), n1)
		v := b.MulMat(bind.input(prefix+"self_attn.v_proj.weight", h, kvDim), n1)

		q = b.Reshape(q, headDim, heads, uint64(seq))
		k = b.Reshape(k, headDim, kvHeads, uint64(seq))
		v = b.Reshape(v, headDim, kvHeads, uint64(seq))
		q = b.WeightedRMSNorm(q, bind.input(prefix+"self_attn.q_norm.weight", headDim), eps)
		k = b.WeightedRMSNorm(k, bind.input(prefix+"self_attn.k_norm.weight", headDim), eps)
		q = b.RoPENeoX(q, positions, uint32(headDim), theta)
		k = b.RoPENeoX(k, positions, uint32(headDim), theta)

		var attn *tensor.Tensor
		if p.keyBias != nil {
			attn = b.AttentionWithKeyBias(q, k, v, p.keyBias, scale, true) // causal GQA + pad-key mask
		} else {
			attn = b.Attention(q, k, v, scale, true) // causal GQA
		}
		attn = b.Reshape(attn, qDim, uint64(seq))
		attn = b.MulMat(bind.input(prefix+"self_attn.o_proj.weight", qDim, h), attn)
		hidden = b.Add(hidden, attn)

		// --- SwiGLU MLP (standard RMSNorm) ---
		n2 := b.WeightedRMSNorm(hidden, bind.input(prefix+"post_attention_layernorm.weight", h), eps)
		g := b.MulMat(bind.input(prefix+"mlp.gate_proj.weight", h, inter), n2)
		u := b.MulMat(bind.input(prefix+"mlp.up_proj.weight", h, inter), n2)
		ff := b.MulMat(bind.input(prefix+"mlp.down_proj.weight", inter, h), b.SwiGLU(g, u))
		hidden = b.Add(hidden, ff)

		if slot, ok := slots[l+1]; ok {
			p.Selected[slot] = hidden
			p.CaptureAfter[slot] = l
		}
	}
	for i, s := range p.Selected {
		if s == nil {
			return nil, fmt.Errorf("encoder program: selected slot %d never captured", i)
		}
	}
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("encoder program graph: %w", err)
	}
	return p, nil
}

// EncoderFeed is a name->F32 weight lookup for the host-feed reference/CUDA path.
type EncoderFeed func(name string) ([]float32, error)

// RunHostFeed executes the encoder graph through run with F32 weights supplied by
// weightAt and the per-token embedding rows embedRows ([seq*Hidden], token-major
// F32). This is the exact-parity path (F32 weights): reference.Execute gives the
// host golden, the CUDA generic executor gives the device match. Returns the
// tapped hidden states in the [Seq, LayerCount, Hidden] layout Denoiser.
// textConditioning consumes.
func (p *EncoderProgram) RunHostFeed(run GraphRunner, weightAt EncoderFeed, embedRows []float32) (*SelectedHiddenStates, error) {
	if p.MatmulType != dtype.F32 {
		return nil, fmt.Errorf("encoder RunHostFeed: needs F32 weights, program compiled %s", p.MatmulType)
	}
	if len(embedRows) != p.Seq*p.E.Hidden {
		return nil, fmt.Errorf("encoder RunHostFeed: embed rows len=%d want %d", len(embedRows), p.Seq*p.E.Hidden)
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(p.weightInputs)+1)
	for name, node := range p.weightInputs {
		data, err := weightAt(name)
		if err != nil {
			return nil, fmt.Errorf("encoder RunHostFeed: weight %s: %w", name, err)
		}
		elements, _ := node.Shape.Elements()
		if uint64(len(data)) != elements {
			return nil, fmt.Errorf("encoder RunHostFeed: weight %s len=%d want %d", name, len(data), elements)
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: data}
	}
	feeds[p.Embed] = reference.Value{Shape: p.Embed.Shape, Data: embedRows}
	if p.keyBias != nil {
		feeds[p.keyBias] = reference.Value{Shape: p.keyBias.Shape, Data: p.keyBiasData}
	}

	results, err := run(append([]*tensor.Tensor(nil), p.Selected...), feeds)
	if err != nil {
		return nil, fmt.Errorf("encoder RunHostFeed: %w", err)
	}
	return p.assemble(results)
}

// assemble gathers the per-slot [Hidden, seq] outputs into a single
// [Seq, LayerCount, Hidden] SelectedHiddenStates (token-major, then tapped-layer,
// then feature) -- the layout Denoiser.textConditioning consumes.
func (p *EncoderProgram) assemble(results map[*tensor.Tensor]reference.Value) (*SelectedHiddenStates, error) {
	h := p.E.Hidden
	L := len(p.Selected)
	out := &SelectedHiddenStates{Seq: p.Seq, LayerCount: L, Hidden: h, Data: make([]float64, p.Seq*L*h)}
	for slot, node := range p.Selected {
		val, ok := results[node]
		if !ok {
			return nil, fmt.Errorf("encoder assemble: missing selected slot %d output", slot)
		}
		if len(val.Data) != p.Seq*h {
			return nil, fmt.Errorf("encoder assemble: slot %d len=%d want %d", slot, len(val.Data), p.Seq*h)
		}
		for tok := 0; tok < p.Seq; tok++ {
			src := val.Data[tok*h : tok*h+h]
			dst := out.Data[(tok*L+slot)*h : (tok*L+slot)*h+h]
			for c := 0; c < h; c++ {
				dst[c] = float64(src[c])
			}
		}
	}
	return out, nil
}
