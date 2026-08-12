// Qwen3-VL text-encoder forward + selected-hidden-layer capture (host port).
// This is the encoder that PRODUCES the [textSeq, TextLayers, TextHidden]
// conditioning tensor consumed by Denoiser.textConditioning (the text_fusion +
// txt_in stream) -- the image-diffusion counterpart of latentvideo's UMT5
// textcond sibling. It closes the "NEEDS-HOOK" gap named in denoiser.go: the
// Qwen3-VL 36-layer encoder's SELECTED hidden states were previously not ported.
//
// LAYER SELECTION (config-cited, NOT literals). The 12 tapped layers are READ
// from model_index.json text_encoder_select_layers. Those values are
// hidden_states indices in the transformers/diffusers convention where
// hidden_states[0] is the token embedding and hidden_states[N] is the residual
// stream AFTER decoder layer N-1; so a selected value N is captured immediately
// after running decoder layer N-1 (valid range [1, HiddenLayers]). This mirrors
// adaptive extmodel encodeSelectedLayerTextCUDA exactly (capture[layer+1]).
// len(SelectLayers) == TransformerSpec.TextLayers is the fusion-boundary
// cross-check already asserted in crossCheckConfig.
//
// ARITHMETIC (Qwen3 decoder layer, mirrors adaptive runtime_bf16_causal_gqa):
//   - STANDARD RMSNorm: x/sqrt(mean(x^2)+eps) * weight -- NOT the DiT's
//     zero-centered (1+weight) form; eps = text_config.rms_norm_eps.
//   - GQA attention: Heads q / KVHeads kv, per-head q/k RMSNorm over head_dim,
//     rotate-half (Llama/NeoX) RoPE with theta = text_config rope_theta, a CAUSAL
//     mask, and scale 1/sqrt(head_dim). Qwen3-VL mrope collapses to standard rope
//     for a pure-text sequence (all three mrope position sections share the
//     sequential token index), so a single rope table over the token position is
//     exact for the text-conditioning path.
//   - SwiGLU MLP: down(silu(gate(x)) * up(x)); no biases anywhere.
//
// Host math accumulates in f64 (reference-oracle convention, matching denoiser/
// vae). Weights STREAM layer-by-layer (only the embedding rows for the actual
// tokens plus one layer's weights are resident at a time) so the ~3.7B-parameter
// text encoder runs CPU-only for structural/finiteness telemetry with bounded
// peak memory.
//
// ORACLE SCOPE: telemetry/structural only. Exact numeric parity against the CUDA
// serving path needs an adaptive-side dump hook (out of scope; see denoiser.go
// TestDenoiserExactG3IsHookGapped). This port asserts the tokenizer, the 12-layer
// SELECTION, encoder forward shapes + finiteness on the real checkpoint, and that
// the produced conditioning geometry matches the DiT text-stream input -- never
// bit-exact values. The Krea conditioner's chat prompt-template wrapping / pad-row
// layout / attention mask (adaptive's text.prompt_template.prefix/suffix +
// text.max_prompt_tokens) is now reproduced EXACTLY by RenderKreaTextInput in
// textinput.go (dtc brick 1/3); this encoder still accepts a raw id slice, so a
// caller wiring the device text-conditioning path renders via RenderKreaTextInput
// first. The remaining exact-parity residual is the DEVICE bf16 encoder forward.
package latentimage

import (
	"fmt"
	"io"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/safetensors"
)

// textEncoderPrefix is the checkpoint namespace for the Qwen3-VL language model.
// The sibling visual.* tower is a SEPARATE capability (vision conditioning), not
// part of the text-conditioning encoder, and language_model.norm.weight feeds
// last_hidden_state only (never a tapped intermediate); neither is consumed here.
const textEncoderPrefix = "language_model."

// SelectedHiddenStates: the encoder output feeding text_fusion. Data is
// [Seq, LayerCount, Hidden] row-major (token-major, then tapped-layer, then
// feature) -- exactly the layout Denoiser.textConditioning consumes.
type SelectedHiddenStates struct {
	Seq        int
	LayerCount int // == TextEncoderSpec.SelectLayers count == TransformerSpec.TextLayers
	Hidden     int // == TextEncoderSpec.Hidden == TransformerSpec.TextHidden
	Data       []float64
}

// EncoderTensorShapes returns the name -> torch shape ([out,in] for Linear
// weights) manifest the encoder forward consumes, derived entirely from spec:
// the token embedding plus every decoder layer's norms, GQA projections, per-head
// q/k norms, and SwiGLU MLP. This is the capability-retention contract for the
// text-conditioning encoder (visual.* and the final norm are out of this path).
func EncoderTensorShapes(e TextEncoderSpec) map[string][]int {
	h := e.Hidden
	qDim := e.Heads * e.HeadDim
	kvDim := e.KVHeads * e.HeadDim
	inter := e.Intermediate
	shapes := map[string][]int{
		textEncoderPrefix + "embed_tokens.weight": {e.VocabSize, h},
	}
	for l := 0; l < e.HiddenLayers; l++ {
		p := fmt.Sprintf("%slayers.%d.", textEncoderPrefix, l)
		shapes[p+"input_layernorm.weight"] = []int{h}
		shapes[p+"post_attention_layernorm.weight"] = []int{h}
		shapes[p+"self_attn.q_proj.weight"] = []int{qDim, h}
		shapes[p+"self_attn.k_proj.weight"] = []int{kvDim, h}
		shapes[p+"self_attn.v_proj.weight"] = []int{kvDim, h}
		shapes[p+"self_attn.o_proj.weight"] = []int{h, qDim}
		shapes[p+"self_attn.q_norm.weight"] = []int{e.HeadDim}
		shapes[p+"self_attn.k_norm.weight"] = []int{e.HeadDim}
		shapes[p+"mlp.gate_proj.weight"] = []int{inter, h}
		shapes[p+"mlp.up_proj.weight"] = []int{inter, h}
		shapes[p+"mlp.down_proj.weight"] = []int{h, inter}
	}
	return shapes
}

// captureSlots maps a selected hidden_states value to its output slot, and
// validates the selection against the adaptive convention: strictly increasing
// values in [1, HiddenLayers] (index N captured after decoder layer N-1).
func captureSlots(e TextEncoderSpec) (map[int]int, error) {
	slots := make(map[int]int, len(e.SelectLayers))
	last := 0
	for i, v := range e.SelectLayers {
		if v < 1 || v > e.HiddenLayers || (i > 0 && v <= last) {
			return nil, fmt.Errorf("textencoder: select layers %v not strictly increasing in [1,%d]", e.SelectLayers, e.HiddenLayers)
		}
		slots[v] = i
		last = v
	}
	if len(slots) == 0 {
		return nil, fmt.Errorf("textencoder: empty select layers")
	}
	return slots, nil
}

// encLayerWeights: one decoder layer's resident weights (f32). Freed after the
// layer runs so peak host residency is ~one layer.
type encLayerWeights struct {
	inputNorm, postNorm []float32
	qNorm, kNorm        []float32
	qProj, kProj, vProj []float32
	oProj               []float32
	gate, up, down      []float32
}

// ---- host math primitives (f64 accumulation) ------------------------------

// rmsNormStandard applies the standard (Qwen3) RMSNorm over the last axis:
// out = x / sqrt(mean(x^2)+eps) * weight. Returns a new buffer.
func rmsNormStandard(x []float64, weight []float32, rows, d int, eps float64) []float64 {
	out := make([]float64, len(x))
	for r := 0; r < rows; r++ {
		xr := x[r*d : (r+1)*d]
		or := out[r*d : (r+1)*d]
		var ss float64
		for i := 0; i < d; i++ {
			ss += xr[i] * xr[i]
		}
		inv := 1.0 / math.Sqrt(ss/float64(d)+eps)
		for i := 0; i < d; i++ {
			or[i] = xr[i] * inv * float64(weight[i])
		}
	}
	return out
}

// ropeTableLlama builds per-position cos/sin of length headDim in the rotate-half
// (Llama/NeoX) convention: for position p and pair i<half, freq=theta^(-2i/hd),
// angle=p*freq, and cos/sin are DUPLICATED across the two halves so index i and
// index half+i share the same (cos,sin). Matches adaptive runtimeCausalRoPEInto.
func ropeTableLlama(seq, headDim int, theta float64) (cos, sin []float64) {
	half := headDim / 2
	cos = make([]float64, seq*headDim)
	sin = make([]float64, seq*headDim)
	for p := 0; p < seq; p++ {
		for i := 0; i < half; i++ {
			freq := math.Pow(theta, -float64(2*i)/float64(headDim))
			angle := float64(p) * freq
			c, s := math.Cos(angle), math.Sin(angle)
			cos[p*headDim+i], cos[p*headDim+half+i] = c, c
			sin[p*headDim+i], sin[p*headDim+half+i] = s, s
		}
	}
	return cos, sin
}

// encHeadRMSAndRope normalizes each head's q/k over head_dim with the standard
// scale, then applies rotate-half RoPE at the token's position. arr is
// [seq, heads*headDim] token-major.
func encHeadRMSAndRope(arr []float64, normW []float32, seq, heads, headDim int, eps float64, cos, sin []float64) {
	half := headDim / 2
	for r := 0; r < seq; r++ {
		crow := cos[r*headDim : (r+1)*headDim]
		srow := sin[r*headDim : (r+1)*headDim]
		for hh := 0; hh < heads; hh++ {
			base := (r*heads + hh) * headDim
			vec := arr[base : base+headDim]
			var ss float64
			for i := 0; i < headDim; i++ {
				ss += vec[i] * vec[i]
			}
			inv := 1.0 / math.Sqrt(ss/float64(headDim)+eps)
			for i := 0; i < headDim; i++ {
				vec[i] = vec[i] * inv * float64(normW[i])
			}
			for i := 0; i < half; i++ {
				a, b := vec[i], vec[half+i]
				vec[i] = a*crow[i] - b*srow[i]
				vec[half+i] = b*crow[half+i] + a*srow[half+i]
			}
		}
	}
}

// causalGQA runs one causal grouped-query attention over the full sequence:
// each query at position i attends to keys j<=i (lower-triangular), scale
// 1/sqrt(headDim), softmax in f64. keyMask (nil = all-attended, else len seq)
// drops an unattended KEY entirely from every query's softmax -- the f64 oracle
// form of adaptive runtime_causal_gqa_masked_bf16 (attentionKeyAllowed: a masked
// key is excluded from max, sum, and output). Returns [seq, heads*headDim].
func causalGQA(q, k, v []float64, seq, heads, kvHeads, headDim int, keyMask []bool) []float64 {
	group := heads / kvHeads
	scale := 1.0 / math.Sqrt(float64(headDim))
	out := make([]float64, seq*heads*headDim)
	hostmath.ParallelRangeF64(heads, seq*seq*headDim, func(loH, hiH int) {
		scores := make([]float64, seq)
		for hh := loH; hh < hiH; hh++ {
			kvh := hh / group
			for i := 0; i < seq; i++ {
				qvec := q[(i*heads+hh)*headDim : (i*heads+hh)*headDim+headDim]
				mx := math.Inf(-1)
				for j := 0; j <= i; j++ {
					if keyMask != nil && !keyMask[j] {
						continue
					}
					kvec := k[(j*kvHeads+kvh)*headDim : (j*kvHeads+kvh)*headDim+headDim]
					var dot float64
					for c := 0; c < headDim; c++ {
						dot += qvec[c] * kvec[c]
					}
					dot *= scale
					scores[j] = dot
					if dot > mx {
						mx = dot
					}
				}
				var sum float64
				for j := 0; j <= i; j++ {
					if keyMask != nil && !keyMask[j] {
						continue
					}
					e := math.Exp(scores[j] - mx)
					scores[j] = e
					sum += e
				}
				dst := out[(i*heads+hh)*headDim : (i*heads+hh)*headDim+headDim]
				for j := 0; j <= i; j++ {
					if keyMask != nil && !keyMask[j] {
						continue
					}
					p := scores[j] / sum
					vvec := v[(j*kvHeads+kvh)*headDim : (j*kvHeads+kvh)*headDim+headDim]
					for c := 0; c < headDim; c++ {
						dst[c] += p * vvec[c]
					}
				}
			}
		}
	})
	return out
}

// encLayerForward runs one Qwen3 decoder layer in place on x [seq, Hidden],
// returning the mutated residual stream. inter is the MLP intermediate width.
func encLayerForward(e TextEncoderSpec, eps float64, x []float64, seq, inter int, w *encLayerWeights, cos, sin []float64, keyMask []bool) []float64 {
	h := e.Hidden
	qDim := e.Heads * e.HeadDim
	kvDim := e.KVHeads * e.HeadDim

	n1 := rmsNormStandard(x, w.inputNorm, seq, h, eps)
	q := dense(n1, w.qProj, nil, seq, h, qDim)
	k := dense(n1, w.kProj, nil, seq, h, kvDim)
	v := dense(n1, w.vProj, nil, seq, h, kvDim)
	encHeadRMSAndRope(q, w.qNorm, seq, e.Heads, e.HeadDim, eps, cos, sin)
	encHeadRMSAndRope(k, w.kNorm, seq, e.KVHeads, e.HeadDim, eps, cos, sin)
	attn := causalGQA(q, k, v, seq, e.Heads, e.KVHeads, e.HeadDim, keyMask)
	ao := dense(attn, w.oProj, nil, seq, qDim, h)
	for i := range x {
		x[i] += ao[i]
	}
	n2 := rmsNormStandard(x, w.postNorm, seq, h, eps)
	g := dense(n2, w.gate, nil, seq, h, inter)
	u := dense(n2, w.up, nil, seq, h, inter)
	for i := range g {
		g[i] = siluF64(g[i]) * u[i]
	}
	mo := dense(g, w.down, nil, seq, inter, h)
	for i := range x {
		x[i] += mo[i]
	}
	return x
}

// encodeSelected runs the full encoder over the embedded token rows, capturing
// the tapped hidden states. embedRows is [seq, Hidden] f64; layerAt supplies one
// layer's weights on demand (streamed or from a store); inter is the MLP width.
func encodeSelected(e TextEncoderSpec, eps float64, seq, inter int, embedRows []float64, layerAt func(l int) (*encLayerWeights, error), keyMask []bool) (*SelectedHiddenStates, error) {
	slots, err := captureSlots(e)
	if err != nil {
		return nil, err
	}
	if len(embedRows) != seq*e.Hidden {
		return nil, fmt.Errorf("textencoder: embed rows len=%d want %d ([%d,%d])", len(embedRows), seq*e.Hidden, seq, e.Hidden)
	}
	if keyMask != nil && len(keyMask) != seq {
		return nil, fmt.Errorf("textencoder: key mask len=%d want seq=%d", len(keyMask), seq)
	}
	h := e.Hidden
	L := len(e.SelectLayers)
	out := &SelectedHiddenStates{Seq: seq, LayerCount: L, Hidden: h, Data: make([]float64, seq*L*h)}
	cos, sin := ropeTableLlama(seq, e.HeadDim, e.RopeTheta)
	x := embedRows
	captured := 0
	maxNeeded := e.SelectLayers[len(e.SelectLayers)-1] // highest tapped index; layers above it are never captured
	for l := 0; l < e.HiddenLayers && l < maxNeeded; l++ {
		w, err := layerAt(l)
		if err != nil {
			return nil, fmt.Errorf("textencoder: layer %d: %w", l, err)
		}
		x = encLayerForward(e, eps, x, seq, inter, w, cos, sin, keyMask)
		if slot, ok := slots[l+1]; ok {
			for tok := 0; tok < seq; tok++ {
				copy(out.Data[(tok*L+slot)*h:(tok*L+slot)*h+h], x[tok*h:tok*h+h])
			}
			captured++
		}
	}
	if captured != L {
		return nil, fmt.Errorf("textencoder: captured %d/%d selected layers", captured, L)
	}
	return out, nil
}

// ---- real-checkpoint streaming encode -------------------------------------

// EncodeSelectedLayers streams the Qwen3-VL text encoder over promptIDs from the
// real checkpoint under modelDir (never resident beyond one layer + the token
// embedding rows) and returns the tapped hidden states. Telemetry/structural:
// the values are finite and correctly shaped, NOT bit-exact vs the CUDA path.
func EncodeSelectedLayers(modelDir string, spec *Spec, promptIDs []int) (*SelectedHiddenStates, error) {
	return EncodeSelectedLayersMasked(modelDir, spec, promptIDs, nil)
}

// EncodeSelectedLayersMasked is EncodeSelectedLayers with an attention key mask:
// keyMask[i]==false drops token i as an unattended KEY (the Krea pad rows). nil
// keyMask == the maskless path. This is the f64 host oracle for the MASKED
// device encoder (CompileEncoderProgramMasked), so masked host==device parity is
// well-defined.
func EncodeSelectedLayersMasked(modelDir string, spec *Spec, promptIDs []int, keyMask []bool) (*SelectedHiddenStates, error) {
	if len(promptIDs) == 0 {
		return nil, fmt.Errorf("textencoder: empty prompt")
	}
	if keyMask != nil && len(keyMask) != len(promptIDs) {
		return nil, fmt.Errorf("textencoder: key mask len=%d want %d", len(keyMask), len(promptIDs))
	}
	e := spec.TextEncoder
	if e.RMSNormEps <= 0 {
		return nil, fmt.Errorf("textencoder: rms_norm_eps must be positive, got %g", e.RMSNormEps)
	}
	if e.Intermediate <= 0 {
		return nil, fmt.Errorf("textencoder: intermediate_size must be positive, got %d", e.Intermediate)
	}
	src, err := safetensors.OpenSource(modelDir + "/text_encoder")
	if err != nil {
		return nil, fmt.Errorf("textencoder: open text_encoder: %w", err)
	}
	defer src.Close()

	// Cross-check the config MLP width against the real down_proj tensor shape.
	inter, err := encoderCheckpointIntermediate(src)
	if err != nil {
		return nil, err
	}
	if inter != e.Intermediate {
		return nil, fmt.Errorf("textencoder: checkpoint intermediate %d != config %d", inter, e.Intermediate)
	}

	seq := len(promptIDs)
	embedRows, err := readEmbedRows(src, e, promptIDs)
	if err != nil {
		return nil, err
	}
	layerAt := func(l int) (*encLayerWeights, error) {
		return readEncoderLayer(src, e, inter, l)
	}
	return encodeSelected(e, e.RMSNormEps, seq, inter, embedRows, layerAt, keyMask)
}

// encoderCheckpointIntermediate reads the MLP intermediate width from layer 0's
// mlp.down_proj.weight shape ([hidden, intermediate]).
func encoderCheckpointIntermediate(src *safetensors.Source) (int, error) {
	name := textEncoderPrefix + "layers.0.mlp.down_proj.weight"
	t, ok := src.Tensors[name]
	if !ok || len(t.Shape) != 2 {
		return 0, fmt.Errorf("textencoder: missing/short %s", name)
	}
	return int(t.Shape[1]), nil
}

// readEmbedRows reads only the embedding rows for the given token ids from the
// BF16 embed_tokens.weight (selective ReadAt -- the full [vocab,hidden] table is
// never materialized), decoding each row to f64.
func readEmbedRows(src *safetensors.Source, e TextEncoderSpec, ids []int) ([]float64, error) {
	name := textEncoderPrefix + "embed_tokens.weight"
	t, ok := src.Tensors[name]
	if !ok || len(t.Shape) != 2 {
		return nil, fmt.Errorf("textencoder: missing/short %s", name)
	}
	if t.DType != "BF16" {
		return nil, fmt.Errorf("textencoder: %s dtype %q != BF16", name, t.DType)
	}
	vocab, h := int(t.Shape[0]), int(t.Shape[1])
	if h != e.Hidden {
		return nil, fmt.Errorf("textencoder: embed hidden %d != spec %d", h, e.Hidden)
	}
	out := make([]float64, len(ids)*h)
	buf := make([]byte, h*2)
	for i, id := range ids {
		if id < 0 || id >= vocab {
			return nil, fmt.Errorf("textencoder: token id %d out of range [0,%d)", id, vocab)
		}
		if _, err := t.ReadAt(buf, int64(id)*int64(h)*2); err != nil {
			return nil, fmt.Errorf("textencoder: embed row %d: %w", id, err)
		}
		for c := 0; c < h; c++ {
			bits := uint16(buf[2*c]) | uint16(buf[2*c+1])<<8
			out[i*h+c] = float64(math.Float32frombits(uint32(bits) << 16))
		}
	}
	return out, nil
}

// readEncoderLayer reads and decodes one decoder layer's weight set to f32.
func readEncoderLayer(src *safetensors.Source, e TextEncoderSpec, inter, l int) (*encLayerWeights, error) {
	p := fmt.Sprintf("%slayers.%d.", textEncoderPrefix, l)
	h, qDim, kvDim := e.Hidden, e.Heads*e.HeadDim, e.KVHeads*e.HeadDim
	get := func(suffix string, want int) ([]float32, error) {
		return readTensorF32(src, p+suffix, want)
	}
	w := &encLayerWeights{}
	var err error
	for _, spec := range []struct {
		dst  *[]float32
		name string
		n    int
	}{
		{&w.inputNorm, "input_layernorm.weight", h},
		{&w.postNorm, "post_attention_layernorm.weight", h},
		{&w.qNorm, "self_attn.q_norm.weight", e.HeadDim},
		{&w.kNorm, "self_attn.k_norm.weight", e.HeadDim},
		{&w.qProj, "self_attn.q_proj.weight", qDim * h},
		{&w.kProj, "self_attn.k_proj.weight", kvDim * h},
		{&w.vProj, "self_attn.v_proj.weight", kvDim * h},
		{&w.oProj, "self_attn.o_proj.weight", h * qDim},
		{&w.gate, "mlp.gate_proj.weight", inter * h},
		{&w.up, "mlp.up_proj.weight", inter * h},
		{&w.down, "mlp.down_proj.weight", h * inter},
	} {
		if *spec.dst, err = get(spec.name, spec.n); err != nil {
			return nil, err
		}
	}
	return w, nil
}

// readTensorF32 reads a whole tensor payload, promoting BF16/F16/F32 to F32, and
// checks the element count against want.
func readTensorF32(src *safetensors.Source, name string, want int) ([]float32, error) {
	t, ok := src.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("textencoder: missing tensor %s", name)
	}
	if int(t.Elements()) != want {
		return nil, fmt.Errorf("textencoder: tensor %s elements=%d want %d", name, t.Elements(), want)
	}
	reader, err := safetensors.F32Reader(t)
	if err != nil {
		return nil, fmt.Errorf("textencoder: tensor %s: %w", name, err)
	}
	raw := make([]byte, want*4)
	if _, err := io.ReadFull(reader, raw); err != nil {
		return nil, fmt.Errorf("textencoder: tensor %s payload: %w", name, err)
	}
	out := make([]float32, want)
	for i := range out {
		bits := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

// ---- structural checkpoint verification (headers only) --------------------

// EncoderWitness: the structural oracle for the encoder port -- the selection
// indices, the tapped-layer capture map, and every consumed tensor's shape vs the
// real checkpoint.
type EncoderWitness struct {
	Checks       []Check
	SelectLayers []int
	CaptureAfter []int // decoder layer (0-based) each tap is captured after
	Intermediate int
	Tensors      int
}

// Failed reports whether any check disagreed.
func (w EncoderWitness) Failed() bool {
	for _, c := range w.Checks {
		if !c.Equal() {
			return true
		}
	}
	return false
}

// VerifyEncoderCheckpoint opens the text_encoder safetensors HEADER under
// modelDir and asserts every tensor the encoder forward consumes exists with the
// derived shape, plus the selection-index invariants. Never reads any payload.
func VerifyEncoderCheckpoint(modelDir string) (*EncoderWitness, error) {
	spec, err := Derive(modelDir)
	if err != nil {
		return nil, err
	}
	e := spec.TextEncoder
	if _, err := captureSlots(e); err != nil {
		return nil, err
	}
	src, err := safetensors.OpenSource(modelDir + "/text_encoder")
	if err != nil {
		return nil, fmt.Errorf("textencoder verify: open text_encoder: %w", err)
	}
	defer src.Close()

	inter, err := encoderCheckpointIntermediate(src)
	if err != nil {
		return nil, err
	}
	shapes := EncoderTensorShapes(e)
	w := &EncoderWitness{
		SelectLayers: append([]int(nil), e.SelectLayers...),
		Intermediate: inter,
		Tensors:      len(shapes),
	}
	for _, v := range e.SelectLayers {
		w.CaptureAfter = append(w.CaptureAfter, v-1)
	}
	// config MLP width vs the real mlp.down_proj[1].
	w.Checks = append(w.Checks, Check{Name: "e.intermediate", Want: e.Intermediate, Got: inter, Source: "mlp.down_proj.weight[1]"})
	for name, shape := range shapes {
		tensor, ok := src.Tensors[name]
		if !ok {
			w.Checks = append(w.Checks, Check{Name: name, Want: prod(shape), Got: -1, Source: "MISSING"})
			continue
		}
		got := 1
		for _, s := range tensor.Shape {
			got *= int(s)
		}
		w.Checks = append(w.Checks, Check{Name: name, Want: prod(shape), Got: got, Source: name})
	}
	if w.Failed() {
		n := 0
		for _, c := range w.Checks {
			if !c.Equal() {
				n++
			}
		}
		return w, fmt.Errorf("textencoder verify: %d structural check(s) disagreed with checkpoint", n)
	}
	return w, nil
}
