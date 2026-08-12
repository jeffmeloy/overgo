package diffusionimage

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testutil"
)

// Parity gates: the reference port's committed tolerances (adaptive
// uvit_test.go) — 1e-5 tight / 1e-4 loose per component, 2e-4 for the full
// tiny-model gradient, 5e-4 for the real-checkpoint forward.
const (
	tolTight = 1e-5
	tolLoose = 1e-4
	tolGrad  = 2e-4
	tolReal  = 5e-4
)

// loadGolden: rung-9 goldens are committed IN-REPO (fixtures/simplediffusion);
// absence is a defect, not an environment gap.
func loadGolden[T any](t *testing.T, name string) T {
	t.Helper()
	path := testutil.FixturePath(t, "simplediffusion", name+".json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("missing required golden %s: %v", path, err)
	}
	var out T
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal %s: %v", path, err)
	}
	return out
}

func maxAbsDiff(got, want []float32) float64 {
	worst := 0.0
	for i := range got {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > worst {
			worst = d
		}
	}
	return worst
}

// requireWithin logs the measured diff verbatim and fails past the gate.
func requireWithin(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != golden %d", name, len(got), len(want))
	}
	diff := maxAbsDiff(got, want)
	t.Logf("%s: max abs diff %.6e (gate %.0e)", name, diff, tol)
	if diff > tol {
		t.Fatalf("%s diverges: %g > %g", name, diff, tol)
	}
}

// uditTinyFixture: the tiny full-model golden. Threshold fields are the
// generator's record; block types are discovered from tensor presence.
type uditTinyFixture struct {
	B, InChannels, H, W, BaseChannels, PatchSize       int
	NumLevels, EncoderBlocks, DecoderBlocks, MidBlocks int
	Groups, Heads, QRank, KVRank                       int
	NormEps                                            float64
	WeightKeys                                         []string
	WeightShapes                                       map[string][]int
	X, Y                                               []float32
}

// generatedWeights: the fixture's deterministic weight pattern (adaptive
// uvit_test.go generatedWeights — vals[i]=((i+(k+1)*7)%29-14)/128).
func (fx uditTinyFixture) generatedWeights(t *testing.T) map[string][]float32 {
	t.Helper()
	out := make(map[string][]float32, len(fx.WeightKeys))
	for keyIndex, key := range fx.WeightKeys {
		shape, ok := fx.WeightShapes[key]
		if !ok {
			t.Fatalf("missing weight shape for %s", key)
		}
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		vals := make([]float32, n)
		offset := (keyIndex + 1) * 7
		for i := range vals {
			vals[i] = float32(((i+offset)%29)-14) / 128
		}
		out[key] = vals
	}
	return out
}

func compileTiny(t *testing.T, fx uditTinyFixture) *Model {
	t.Helper()
	m, err := Compile(fx.generatedWeights(t))
	if err != nil {
		t.Fatalf("compile tiny model: %v", err)
	}
	want := Config{
		InChannels: fx.InChannels, BaseChannels: fx.BaseChannels, PatchSize: fx.PatchSize,
		NumLevels: fx.NumLevels, MidBlocks: fx.MidBlocks, Groups: fx.Groups,
		Heads: fx.Heads, QRank: fx.QRank, KVRank: fx.KVRank,
		NormEps: fx.NormEps, RopeTheta: vendorRopeTheta,
	}
	if m.Cfg != want {
		t.Fatalf("derived tiny config %+v, want %+v", m.Cfg, want)
	}
	return m
}

// finiteDiffCheck: central-difference spot check against the analytic
// gradient using the f32 forward (rung-8 committed protocol: eps 3e-3,
// relative gate 2e-2 with unit floor).
func finiteDiffCheck(t *testing.T, name string, params, analytic []float32, indices []int, loss func() float64) {
	t.Helper()
	const (
		eps    = 3e-3
		relTol = 2e-2
	)
	for _, i := range indices {
		save := params[i]
		params[i] = save + eps
		lp := loss()
		params[i] = save - eps
		lm := loss()
		params[i] = save
		num := (lp - lm) / (2 * eps)
		den := float64(analytic[i])
		scale := math.Abs(den)
		if scale < 1 {
			scale = 1
		}
		if d := math.Abs(num-den) / scale; d > relTol {
			t.Errorf("%s[%d]: analytic=%.6g finite-diff=%.6g (rel %.2e > %.0e)", name, i, den, num, d, relTol)
		}
	}
}

func spotIndices(n int) []int {
	if n == 1 {
		return []int{0}
	}
	return []int{0, n / 3, n - 1}
}

// Shared once-loaded real artifact.
var (
	modelOnce sync.Once
	modelInst *Model
	modelErr  error
)

func artifactDir(t *testing.T) string {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(roots.Models, "SimpleDiffusion-TensorProductAttentionRope")
}

func loadArtifactModel(t *testing.T) *Model {
	t.Helper()
	if testing.Short() {
		t.Skip("loads the real artifact; skipped in -short")
	}
	dir := artifactDir(t)
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s; parity NOT verified", dir)
	}
	modelOnce.Do(func() { modelInst, modelErr = Load(dir) })
	if modelErr != nil {
		t.Fatal(modelErr)
	}
	return modelInst
}
