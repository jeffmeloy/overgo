package latentimage

import (
	"fmt"
	"io"
	"math"
	"strings"
	"testing"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

// syntheticEncoderSpec: a tiny but structurally faithful Qwen3-VL text encoder
// (GQA divisibility, selection strictly increasing in [1,HiddenLayers]) so the
// exact forward + capture logic runs CPU-cheaply. The real encoder is ~3.7B params.
func syntheticEncoderSpec() TextEncoderSpec {
	return TextEncoderSpec{
		ModelType:    "fixture-text-encoder",
		HiddenLayers: 6,
		Hidden:       16,
		Intermediate: 24,
		Heads:        4,
		KVHeads:      2,
		HeadDim:      4,
		VocabSize:    40,
		RopeTheta:    5000000,
		RMSNormEps:   1e-6,
		SelectLayers: []int{2, 4, 5}, // hidden_states idx -> captured after layers 1,3,4
	}
}

// encoderStore fills every manifest tensor with small deterministic values so
// the forward stays well-conditioned.
func encoderStore(e TextEncoderSpec) map[string][]float32 {
	shapes := EncoderTensorShapes(e)
	store := make(map[string][]float32, len(shapes))
	state := uint64(0x9e3779b97f4a7c15)
	next := func() float32 {
		state = state*6364136223846793005 + 1442695040888963407
		u := float64(state>>11) / float64(1<<53)
		return float32((u - 0.5) * 0.1)
	}
	for name, shape := range shapes {
		v := make([]float32, prod(shape))
		for i := range v {
			v[i] = next()
		}
		store[name] = v
	}
	return store
}

// storeLayerAt returns a layerAt closure reading one layer's weights from a store.
func storeLayerAt(e TextEncoderSpec, store map[string][]float32) func(int) (*encLayerWeights, error) {
	return func(l int) (*encLayerWeights, error) {
		p := fmt.Sprintf("%slayers.%d.", textEncoderPrefix, l)
		get := func(s string) []float32 { return store[p+s] }
		return &encLayerWeights{
			inputNorm: get("input_layernorm.weight"),
			postNorm:  get("post_attention_layernorm.weight"),
			qNorm:     get("self_attn.q_norm.weight"),
			kNorm:     get("self_attn.k_norm.weight"),
			qProj:     get("self_attn.q_proj.weight"),
			kProj:     get("self_attn.k_proj.weight"),
			vProj:     get("self_attn.v_proj.weight"),
			oProj:     get("self_attn.o_proj.weight"),
			gate:      get("mlp.gate_proj.weight"),
			up:        get("mlp.up_proj.weight"),
			down:      get("mlp.down_proj.weight"),
		}, nil
	}
}

// captureSlots must accept the real Krea selection and reject malformed ones. The
// capture-after mapping (index N -> decoder layer N-1) is the diffusers/adaptive
// convention and is the load-bearing correctness fact for the tapped fusion.
func TestEncoderCaptureSlots(t *testing.T) {
	e := TextEncoderSpec{HiddenLayers: 36, SelectLayers: []int{2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35}}
	slots, err := captureSlots(e)
	if err != nil {
		t.Fatalf("captureSlots: %v", err)
	}
	if len(slots) != 12 {
		t.Fatalf("slots=%d want 12", len(slots))
	}
	// selected value N is captured after decoder layer N-1, into slot = order index.
	for i, v := range e.SelectLayers {
		if slots[v] != i {
			t.Errorf("select %d -> slot %d want %d", v, slots[v], i)
		}
	}
	// rejections: out of range, non-increasing, empty.
	for _, bad := range [][]int{{0, 5}, {5, 5}, {5, 2}, {2, 37}, {}} {
		if _, err := captureSlots(TextEncoderSpec{HiddenLayers: 36, SelectLayers: bad}); err == nil {
			t.Errorf("captureSlots(%v) accepted, want error", bad)
		}
	}
}

// The synthetic encoder forward must produce the [seq, LayerCount, Hidden]
// conditioning tensor, finite and deterministic, capturing exactly the selected
// layers -- the model-free proof of the selection + geometry logic (runs in CI).
func TestEncoderForwardSyntheticFinite(t *testing.T) {
	e := syntheticEncoderSpec()
	store := encoderStore(e)
	seq := 5
	embed := make([]float64, seq*e.Hidden)
	st := uint64(0x2545f4914f6cdd1d)
	for i := range embed {
		st = st*6364136223846793005 + 1442695040888963407
		embed[i] = (float64(st>>11)/float64(1<<53) - 0.5) * 0.2
	}
	out, err := encodeSelected(e, e.RMSNormEps, seq, e.Intermediate, append([]float64(nil), embed...), storeLayerAt(e, store), nil)
	if err != nil {
		t.Fatalf("encodeSelected: %v", err)
	}
	if out.Seq != seq || out.LayerCount != len(e.SelectLayers) || out.Hidden != e.Hidden {
		t.Fatalf("geometry seq=%d layers=%d hidden=%d", out.Seq, out.LayerCount, out.Hidden)
	}
	if want := seq * len(e.SelectLayers) * e.Hidden; len(out.Data) != want {
		t.Fatalf("data len=%d want %d", len(out.Data), want)
	}
	if !allFinite(out.Data) {
		t.Fatal("encoder output not finite")
	}
	// determinism
	out2, err := encodeSelected(e, e.RMSNormEps, seq, e.Intermediate, append([]float64(nil), embed...), storeLayerAt(e, store), nil)
	if err != nil {
		t.Fatalf("encodeSelected(2): %v", err)
	}
	for i := range out.Data {
		if out.Data[i] != out2.Data[i] {
			t.Fatalf("nondeterministic at %d: %g vs %g", i, out.Data[i], out2.Data[i])
		}
	}
	t.Logf("synthetic encoder OK: [%d,%d,%d] finite/deterministic; taps after layers %v",
		out.Seq, out.LayerCount, out.Hidden, captureAfter(e))
}

func captureAfter(e TextEncoderSpec) []int {
	out := make([]int, len(e.SelectLayers))
	for i, v := range e.SelectLayers {
		out[i] = v - 1
	}
	return out
}

// The synthetic encoder output must feed Denoiser.textConditioning (the DiT text
// stream) at the exact geometry boundary: [seq, TextLayers, TextHidden] in ->
// [seq, Hidden] out. This is the model-free proof that the encoder wires to the
// fusion consumer.
func TestEncoderFeedsFusionGeometrySynthetic(t *testing.T) {
	// A transformer whose text stream matches the synthetic encoder's geometry.
	e := syntheticEncoderSpec()
	tr := TransformerSpec{
		Layers: 1, Heads: 4, KVHeads: 2, HeadDim: e.HeadDim,
		Hidden: 4 * e.HeadDim, KVDim: 2 * e.HeadDim, InChannels: 16, Intermediate: 16,
		RopeAxes: [3]int{4, 6, 6}, RopeTheta: 1000, NormEps: 1e-5, TimestepEmbed: 8, ModFields: 6,
		TextLayers: len(e.SelectLayers), TextHidden: e.Hidden, TextIntermediate: 16,
		// text-fusion block invariant: TextHeads*HeadDim == TextHidden (real: 20*128=2560).
		TextHeads: e.Hidden / e.HeadDim, TextKVHeads: 2, LayerwiseTextBlocks: 1, RefinerTextBlocks: 1,
	}
	if tr.TextLayers != len(e.SelectLayers) || tr.TextHidden != e.Hidden {
		t.Fatalf("fusion boundary mismatch: text_layers=%d hidden=%d vs encoder %d/%d",
			tr.TextLayers, tr.TextHidden, len(e.SelectLayers), e.Hidden)
	}
	d := &Denoiser{T: tr, Eps: tr.NormEps, store: syntheticFusionStore(tr)}
	seq := 4
	enc := make([]float64, seq*tr.TextLayers*tr.TextHidden)
	for i := range enc {
		enc[i] = math.Sin(float64(i) * 0.017)
	}
	txt, err := d.textConditioning(enc, seq)
	if err != nil {
		t.Fatalf("textConditioning: %v", err)
	}
	if want := seq * tr.Hidden; len(txt) != want {
		t.Fatalf("fused conditioning len=%d want %d ([%d,%d])", len(txt), want, seq, tr.Hidden)
	}
	if !allFinite(txt) {
		t.Fatal("fused conditioning not finite")
	}
	t.Logf("synthetic fusion OK: encoder [%d,%d,%d] -> DiT text stream [%d,%d]",
		seq, tr.TextLayers, tr.TextHidden, seq, tr.Hidden)
}

// syntheticFusionStore fills only the text_fusion.* + txt_in.* tensors (the ones
// textConditioning reads) with small deterministic values.
func syntheticFusionStore(tr TransformerSpec) map[string][]float32 {
	shapes := DenoiserTensorShapes(tr)
	store := make(map[string][]float32, len(shapes))
	state := uint64(0xc2b2ae3d27d4eb4f)
	next := func() float32 {
		state = state*6364136223846793005 + 1442695040888963407
		return float32((float64(state>>11)/float64(1<<53) - 0.5) * 0.1)
	}
	for name, shape := range shapes {
		v := make([]float32, prod(shape))
		for i := range v {
			v[i] = next()
		}
		store[name] = v
	}
	return store
}

// ---- real-checkpoint structural + telemetry oracle ------------------------

// VerifyEncoderCheckpoint must derive the Qwen3-VL text-encoder geometry from the
// REAL Krea-2-Turbo checkpoint (headers only): every consumed tensor present at
// the exact shape, the 12 SELECTED layers config-read from
// model_index.text_encoder_select_layers, and their capture-after mapping.
func TestEncoderDerivesFromRealCheckpoint(t *testing.T) {
	dir := kreaDirOrSkip(t)
	w, err := VerifyEncoderCheckpoint(dir)
	if err != nil {
		if w != nil {
			for _, c := range w.Checks {
				if !c.Equal() {
					t.Errorf("%s", c.String())
				}
			}
		}
		t.Fatalf("VerifyEncoderCheckpoint: %v", err)
	}
	wantSel := []int{2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35}
	if len(w.SelectLayers) != len(wantSel) {
		t.Fatalf("select layers=%v want %v", w.SelectLayers, wantSel)
	}
	for i := range wantSel {
		if w.SelectLayers[i] != wantSel[i] {
			t.Errorf("select[%d]=%d want %d", i, w.SelectLayers[i], wantSel[i])
		}
		if w.CaptureAfter[i] != wantSel[i]-1 {
			t.Errorf("captureAfter[%d]=%d want %d", i, w.CaptureAfter[i], wantSel[i]-1)
		}
	}
	if w.Intermediate != 9728 {
		t.Errorf("intermediate=%d want 9728", w.Intermediate)
	}
	spec, _ := Derive(dir)
	e := spec.TextEncoder
	for _, tc := range []struct {
		name      string
		got, want int
	}{
		{"hidden_layers", e.HiddenLayers, 36}, {"hidden", e.Hidden, 2560},
		{"heads", e.Heads, 32}, {"kv_heads", e.KVHeads, 8}, {"head_dim", e.HeadDim, 128},
		{"vocab", e.VocabSize, 151936}, {"text_layers", len(e.SelectLayers), spec.Transformer.TextLayers},
	} {
		if tc.got != tc.want {
			t.Errorf("%s=%d want %d", tc.name, tc.got, tc.want)
		}
	}
	t.Logf("encoder geometry verified vs real ckpt: %d tensors consumed, select=%v, captured after layers=%v, rms_eps=%g theta=%g",
		w.Tensors, w.SelectLayers, w.CaptureAfter, e.RMSNormEps, e.RopeTheta)
}

// loadFusionStore reads only the text_fusion.* + txt_in.* transformer tensors
// (the ones Denoiser.textConditioning consumes) so the fusion runs with bounded
// residency -- the full 12.82B-param transformer is never materialized.
func loadFusionStore(dir string) (map[string][]float32, error) {
	src, err := safetensors.OpenSource(dir + `\transformer`)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	store := make(map[string][]float32)
	for name, tensor := range src.Tensors {
		if !strings.HasPrefix(name, "text_fusion.") && !strings.HasPrefix(name, "txt_in.") {
			continue
		}
		v, err := readTensorF32(src, name, int(tensor.Elements()))
		if err != nil {
			return nil, err
		}
		store[name] = v
	}
	return store, nil
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
		w.CaptureAfter = append(w.CaptureAfter, v-tensor.SingletonExtent)
	}
	// config MLP width vs the real mlp.down_proj[1].
	w.Checks = append(w.Checks, Check{Name: "e.intermediate", Want: e.Intermediate, Got: inter, Source: "mlp.down_proj.weight[1]"})
	for name, shape := range shapes {
		checkpointTensor, ok := src.Tensors[name]
		if !ok {
			w.Checks = append(w.Checks, Check{Name: name, Want: prod(shape), Got: checked.UnknownCount(), Source: "MISSING"})
			continue
		}
		got, ok := checked.Int(checkpointTensor.Elements())
		if !ok {
			got = checked.UnknownCount()
		}
		w.Checks = append(w.Checks, Check{Name: name, Want: prod(shape), Got: got, Source: name})
	}
	if failures := failedCheckCount(w.Checks); checked.Nonzero(failures) {
		return w, fmt.Errorf("textencoder verify: %d structural check(s) disagreed with checkpoint", failures)
	}
	return w, nil
}

// EncodeSelectedLayers streams the Qwen3-VL text encoder over promptIDs from the
// real checkpoint under modelDir (never resident beyond one layer + the token
// embedding rows) and returns the tapped hidden states. Telemetry/structural:
// the values are finite and correctly shaped, NOT bit-exact vs the CUDA path.
func EncodeSelectedLayers(modelDir string, spec *Spec, promptIDs []int) (*SelectedHiddenStates, error) {
	return EncodeSelectedLayersMasked(modelDir, spec, promptIDs, nil)
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
	for l := range e.HiddenLayers {
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

// ropeTableLlama builds each token's cos/sin of length headDim in the rotate-half
// (Llama/NeoX) convention: for a token at position p and pair i<half,
// freq=theta^(-2i/hd), angle=p*freq, and cos/sin are DUPLICATED across the two
// halves so index i and index half+i share the same (cos,sin). Matches adaptive
// runtimeCausalRoPEInto.
func ropeTableLlama(positions []uint32, headDim int, theta float64) (cos, sin []float64) {
	half := headDim / 2
	cos = make([]float64, len(positions)*headDim)
	sin = make([]float64, len(positions)*headDim)
	for p, position := range positions {
		for i := range half {
			freq := math.Pow(theta, -float64(2*i)/float64(headDim))
			angle := float64(position) * freq
			c, s := math.Cos(angle), math.Sin(angle)
			cos[p*headDim+i], cos[p*headDim+half+i] = c, c
			sin[p*headDim+i], sin[p*headDim+half+i] = s, s
		}
	}
	return cos, sin
}

// encLayerForward runs one Qwen3 decoder layer in place on x [seq, Hidden],
// returning the mutated residual stream. inter is the MLP intermediate width.
func encLayerForward(e TextEncoderSpec, eps float64, x []float64, seq, inter int, w *encLayerWeights, cos, sin []float64, keyMask []bool) []float64 {
	h := e.Hidden
	qDim := e.Heads * e.HeadDim
	kvDim := e.KVHeads * e.HeadDim

	n1 := hostmath.StandardRMSNormF64(x, w.inputNorm, seq, h, eps)
	q := dense(n1, w.qProj, nil, seq, h, qDim)
	k := dense(n1, w.kProj, nil, seq, h, kvDim)
	v := dense(n1, w.vProj, nil, seq, h, kvDim)
	hostmath.NormalizeHeadsRotaryHalfF64(q, w.qNorm, seq, e.Heads, e.HeadDim, eps, cos, sin)
	hostmath.NormalizeHeadsRotaryHalfF64(k, w.kNorm, seq, e.KVHeads, e.HeadDim, eps, cos, sin)
	attn := hostmath.GroupedCausalAttentionF64(q, k, v, seq, e.Heads, e.KVHeads, e.HeadDim, keyMask)
	ao := dense(attn, w.oProj, nil, seq, qDim, h)
	for i := range x {
		x[i] += ao[i]
	}
	n2 := hostmath.StandardRMSNormF64(x, w.postNorm, seq, h, eps)
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
	if err := checked.Length(embedRows, seq, e.Hidden); err != nil {
		return nil, fmt.Errorf("textencoder: embed rows: %w", err)
	}
	if keyMask != nil {
		if err := checked.Length(keyMask, seq); err != nil {
			return nil, fmt.Errorf("textencoder: key mask: %w", err)
		}
	}
	h := e.Hidden
	L := len(e.SelectLayers)
	out := &SelectedHiddenStates{Seq: seq, LayerCount: L, Hidden: h, Data: make([]float64, seq*L*h)}
	cos, sin := ropeTableLlama(attendedPositions(seq, keyMask), e.HeadDim, e.RopeTheta)
	x := embedRows
	captured := tensor.FirstOffset
	maxNeeded, _ := checked.Last(e.SelectLayers) // captureSlots established a non-empty selection
	for l := range min(e.HiddenLayers, maxNeeded) {
		w, err := layerAt(l)
		if err != nil {
			return nil, fmt.Errorf("textencoder: layer %d: %w", l, err)
		}
		x = encLayerForward(e, eps, x, seq, inter, w, cos, sin, keyMask)
		if slot, ok := slots[l+tensor.SingletonExtent]; ok {
			for tok := range seq {
				copy(out.Data[(tok*L+slot)*h:(tok*L+slot)*h+h], x[tok*h:tok*h+h])
			}
			captured++
		}
	}
	if !checked.Equal(captured, L) {
		return nil, fmt.Errorf("textencoder: captured %d/%d selected layers", captured, L)
	}
	return out, nil
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
	if keyMask != nil && !checked.Equal(len(keyMask), len(promptIDs)) {
		return nil, fmt.Errorf("textencoder: key mask len=%d want %d", len(keyMask), len(promptIDs))
	}
	e := spec.TextEncoder
	if !checked.PositiveFinite64(e.RMSNormEps) {
		return nil, fmt.Errorf("textencoder: rms_norm_eps must be positive, got %g", e.RMSNormEps)
	}
	if !checked.PositiveInts(e.Intermediate) {
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
	if !checked.Equal(inter, e.Intermediate) {
		return nil, fmt.Errorf("textencoder: checkpoint intermediate %d != config %d", inter, e.Intermediate)
	}

	seq := len(promptIDs)
	embedTensor, ok := src.Tensors[textEncoderPrefix+"embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("textencoder: missing embedding tensor")
	}
	embedRows, err := safetensors.TensorRowsF64(embedTensor, e.Hidden, promptIDs)
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
	if !ok {
		return 0, fmt.Errorf("textencoder: missing/short %s", name)
	}
	_, width, err := safetensors.MatrixShape(t)
	if err != nil {
		return 0, fmt.Errorf("textencoder: %s: %w", name, err)
	}
	return int(width), nil
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

// SelectedHiddenStates: the encoder output feeding text_fusion. Data is
// [Seq, LayerCount, Hidden] row-major (token-major, then tapped-layer, then
// feature) -- exactly the layout Denoiser.textConditioning consumes.
type SelectedHiddenStates struct {
	Seq        int
	LayerCount int // == TextEncoderSpec.SelectLayers count == TransformerSpec.TextLayers
	Hidden     int // == TextEncoderSpec.Hidden == TransformerSpec.TextHidden
	Data       []float64
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
