package latentimage

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"overgo/internal/hfbpe"
	"overgo/internal/safetensors"
)

// syntheticEncoderSpec: a tiny but structurally faithful Qwen3-VL text encoder
// (GQA divisibility, selection strictly increasing in [1,HiddenLayers]) so the
// exact forward + capture logic runs CPU-cheaply. The real encoder is ~3.7B params.
func syntheticEncoderSpec() TextEncoderSpec {
	return TextEncoderSpec{
		ModelType:    TextEncoderType,
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
	out, err := encodeSelected(e, e.RMSNormEps, seq, e.Intermediate, append([]float64(nil), embed...), storeLayerAt(e, store))
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
	out2, err := encodeSelected(e, e.RMSNormEps, seq, e.Intermediate, append([]float64(nil), embed...), storeLayerAt(e, store))
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

// The full telemetry oracle on the real checkpoint: Qwen2 tokenizer -> streamed
// Qwen3-VL encoder forward -> 12 tapped hidden states -> text_fusion/txt_in ->
// conditioning tensor whose geometry matches the DiT text-stream input. Asserts
// shapes + finiteness (NOT bit-exact; exact parity needs the adaptive dump hook).
func TestEncoderRealCheckpointTelemetry(t *testing.T) {
	if testing.Short() {
		t.Skip("streams the ~3.7B-param text encoder over the prompt; skipped in -short")
	}
	dir := kreaDirOrSkip(t)
	spec, err := Derive(dir)
	if err != nil {
		t.Fatalf("Derive: %v", err)
	}
	tok, err := hfbpe.Load(dir + `\tokenizer`)
	if err != nil {
		t.Fatalf("load Qwen2 tokenizer: %v", err)
	}
	prompt := "a red fox eating ice cream, studio photograph"
	ids, err := tok.Encode(prompt)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("tokenizer produced 0 ids")
	}
	t.Logf("tokenizer: %q -> %d ids: %v", prompt, len(ids), ids)

	enc, err := EncodeSelectedLayers(dir, spec, ids)
	if err != nil {
		t.Fatalf("EncodeSelectedLayers: %v", err)
	}
	if enc.Seq != len(ids) || enc.LayerCount != spec.Transformer.TextLayers || enc.Hidden != spec.Transformer.TextHidden {
		t.Fatalf("encoder geometry seq=%d layers=%d hidden=%d want %d/%d/%d",
			enc.Seq, enc.LayerCount, enc.Hidden, len(ids), spec.Transformer.TextLayers, spec.Transformer.TextHidden)
	}
	if !allFinite(enc.Data) {
		t.Fatal("real encoder output not finite")
	}
	// per-tap magnitude telemetry (finite, non-degenerate residual stream).
	for l := 0; l < enc.LayerCount; l++ {
		var mn, mx, sa float64 = math.Inf(1), math.Inf(-1), 0
		n := 0
		for tok := 0; tok < enc.Seq; tok++ {
			row := enc.Data[(tok*enc.LayerCount+l)*enc.Hidden : (tok*enc.LayerCount+l)*enc.Hidden+enc.Hidden]
			for _, v := range row {
				mn, mx, sa, n = math.Min(mn, v), math.Max(mx, v), sa+math.Abs(v), n+1
			}
		}
		t.Logf("tap %2d (after layer %2d): min=%+.4f max=%+.4f mean|.|=%.4f", l, spec.TextEncoder.SelectLayers[l]-1, mn, mx, sa/float64(n))
	}

	// fusion geometry on real weights: load only text_fusion.* + txt_in.* from the
	// transformer and run textConditioning -> [seq, Hidden] (the DiT text-stream input).
	store, err := loadFusionStore(dir)
	if err != nil {
		t.Fatalf("load fusion store: %v", err)
	}
	d := &Denoiser{T: spec.Transformer, Eps: spec.Transformer.NormEps, store: store}
	txt, err := d.textConditioning(enc.Data, enc.Seq)
	if err != nil {
		t.Fatalf("textConditioning: %v", err)
	}
	if want := enc.Seq * spec.Transformer.Hidden; len(txt) != want {
		t.Fatalf("fused conditioning len=%d want %d ([%d,%d])", len(txt), want, enc.Seq, spec.Transformer.Hidden)
	}
	if !allFinite(txt) {
		t.Fatal("real fused conditioning not finite")
	}
	t.Logf("TELEMETRY PASS: tokenizer(%d ids) -> Qwen3VL encoder [%d,%d,%d] -> text_fusion/txt_in -> DiT text stream [%d,%d] (== hidden=%d); all finite. Exact values need adaptive dump hook (out of scope).",
		len(ids), enc.Seq, enc.LayerCount, enc.Hidden, enc.Seq, spec.Transformer.Hidden, spec.Transformer.Hidden)
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
