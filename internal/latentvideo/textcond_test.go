package latentvideo

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/pytorchzip"
)

// referenceEncoderPolicy: the two published-config facts the checkpoint
// cannot carry (T5-family relative_attention_max_distance and
// layer_norm_epsilon); geometry derives from tensor shapes.
var referenceEncoderPolicy = EncoderConfig{RelativeMaxDistance: 128, NormEps: 1e-6}

func arithmeticFloat32Sequence(start, step float64, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(start + step*float64(i))
	}
	return out
}

func requireFloat32MaxAbs(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want %d", name, len(got), len(want))
	}
	var maxDiff float64
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > tol {
		t.Fatalf("%s max abs diff=%g > %g\ngot=%v\nwant=%v", name, maxDiff, tol, got, want)
	}
}

// TestEncoderBlockMatchesTorchFixture: fixed-weight block forward against
// the reference's committed torch fixture (adaptive
// TestRelativePositionEncoderEncoderBlockMatchesTorchFixture).
func TestEncoderBlockMatchesTorchFixture(t *testing.T) {
	x := []float32{-0.5, -0.4000000059604645, -0.30000001192092896, -0.19999998807907104, -0.09999999403953552, 0, 0.10000002384185791, 0.19999998807907104, 0.30000001192092896, 0.3999999761581421, 0.5, 0.6000000238418579}
	w := encoderBlockWeights{
		Norm1:        arithmeticFloat32Sequence(-0.08, 0.02, 4),
		Q:            arithmeticFloat32Sequence(-0.06, 0.02, 16),
		K:            arithmeticFloat32Sequence(-0.04, 0.02, 16),
		V:            arithmeticFloat32Sequence(-0.02, 0.02, 16),
		O:            arithmeticFloat32Sequence(0.00, 0.02, 16),
		Norm2:        arithmeticFloat32Sequence(0.02, 0.02, 4),
		Gate:         arithmeticFloat32Sequence(0.04, 0.02, 24),
		FC1:          arithmeticFloat32Sequence(0.06, 0.02, 24),
		FC2:          arithmeticFloat32Sequence(0.08, 0.02, 24),
		PosEmbedding: arithmeticFloat32Sequence(0.10, 0.02, 8),
	}
	if got := []int{
		relativePositionBucket(0, 4, true, 8),
		relativePositionBucket(1, 4, true, 8),
		relativePositionBucket(2, 4, true, 8),
		relativePositionBucket(-1, 4, true, 8),
	}; !reflect.DeepEqual(got, []int{0, 3, 3, 1}) {
		t.Fatalf("relative buckets=%v", got)
	}
	buckets, err := compileRelativePositionBuckets(3, 3, 4, referenceEncoderPolicy.RelativeMaxDistance, true)
	if err != nil {
		t.Fatal(err)
	}
	got, err := encoderBlockForward(x, []int{1, 1, 0}, buckets, 1, 3, 4, 2, 6, 4, w, 1e-6, false)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{-0.4968312084674835, -0.39273759722709656, -0.2886439859867096, -0.18455031514167786, -0.09616664052009583, 0.008644644170999527, 0.11345595866441727, 0.21826721727848053, 0.3046256899833679, 0.40998393297195435, 0.5153422355651855, 0.6207005977630615}
	requireFloat32MaxAbs(t, "encoder block", got, want, 2e-6)
}

func TestCompileEncoderPlanRejectsIncomplete(t *testing.T) {
	plan, err := CompileEncoderPlan([]pytorchzip.TensorMeta{
		{Name: "token_embedding.weight", DType: "BFloat16Storage", Shape: []int64{256384, 4096}, Numel: 256384 * 4096},
	}, referenceEncoderPolicy)
	if err != nil {
		t.Fatal(err)
	}
	if plan.OK || len(plan.Missing) == 0 {
		t.Fatalf("expected incomplete plan to fail, got %+v", plan)
	}
}

func TestEncoderBlockTensorNames(t *testing.T) {
	names := encoderBlockTensorNames(3)
	if len(names) != blockTensorCount || names[0] != "blocks.3.norm1.weight" || names[len(names)-1] != "blocks.3.pos_embedding.embedding.weight" {
		t.Fatalf("bad block names: %v", names)
	}
	if err := (EncoderPlan{OK: true, Config: EncoderConfig{VocabSize: 256384, Layers: 24, Dim: 4096, Heads: 64, FFNDim: 10240, RelativeBuckets: 32, RelativeMaxDistance: 128, NormEps: 1e-6}}).validateRuntimeBindings(); err == nil {
		t.Fatal("incomplete runtime bindings accepted")
	}
}

func writeSyntheticTokenizer(t *testing.T, dir string) {
	t.Helper()
	raw := map[string]any{
		"model": map[string]any{
			"type":          "Unigram",
			"unk_id":        3,
			"byte_fallback": false,
			"vocab": []any{
				[]any{"<pad>", 0.0},
				[]any{"</s>", 0.0},
				[]any{"<s>", 0.0},
				[]any{"<unk>", -100.0},
				[]any{"▁hello", 1.0},
				[]any{"▁world", 1.0},
				[]any{"▁", -1.0},
				[]any{"hello", -1.0},
				[]any{"world", -1.0},
			},
		},
	}
	b, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "special_tokens_map.json"), []byte(`{"pad_token":"<pad>","eos_token":"</s>","unk_token":"<unk>"}`), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestFixedUnigramTokenizerSyntheticEncodeWithMask(t *testing.T) {
	dir := t.TempDir()
	writeSyntheticTokenizer(t, dir)
	tok, err := loadFixedUnigramTokenizer(dir, 5)
	if err != nil {
		t.Fatal(err)
	}
	ids, mask, err := tok.EncodeWithMask("  hello   world  ")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ids, []int{4, 5, 1, 0, 0}) || !reflect.DeepEqual(mask, []int{1, 1, 1, 0, 0}) {
		t.Fatalf("ids=%v mask=%v", ids, mask)
	}
}

func TestNormalizeEscapedWidthWhitespace(t *testing.T) {
	// Fullwidth punctuation folds to ASCII; ideographic space collapses.
	if got := normalizeEscapedWidthWhitespace("ａ，　 b"); got != "a, b" {
		t.Fatalf("normalized=%q", got)
	}
	if got := normalizeEscapedWidthWhitespace("&amp;amp;  x "); got != "& x" {
		t.Fatalf("normalized=%q", got)
	}
}
