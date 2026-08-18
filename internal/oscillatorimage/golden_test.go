package oscillatorimage

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

// Parity gates: the reference port's committed tolerances (adaptive
// conditional_oscillator_image_test.go) — 1e-4 per operator (dynamics, conv
// and block VJPs), 1e-3 for the full generator (four Euler steps plus three
// convolution stages of f32 accumulation reordering).
const (
	tolOperator  = 1e-4
	tolGenerator = 1e-3
)

// referenceFixturePath: the un0 goldens ship in the REFERENCE repo's
// fixtures/ directory (sibling of the models root per the data-root
// contract), not inside the artifact directory; read them where they live,
// never copied.
func referenceFixturePath(t *testing.T, name string) string {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(filepath.Dir(roots.Models), "fixtures", name)
}

// loadReferenceGolden decodes a reference-shipped golden; absence skips LOUDLY.
func loadReferenceGolden(t *testing.T, name string, out any) {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": requires adaptive_new reference goldens")
	}
	raw, err := os.ReadFile(referenceFixturePath(t, name))
	if err != nil {
		t.Skipf("UNAVAILABLE: reference golden %s absent; parity NOT verified", name)
	}
	if err := json.Unmarshal(raw, out); err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
}

func f32of(values []float64) []float32 {
	out := make([]float32, len(values))
	for i, v := range values {
		out[i] = float32(v)
	}
	return out
}

// requireWithin logs the measured diff verbatim and fails past the gate.
func requireWithin(t *testing.T, name string, got []float32, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != golden %d", name, len(got), len(want))
	}
	diff := testutil.MaxAbsDiff(got, want)
	t.Logf("%s: max abs diff %.6e (gate %.0e)", name, diff, tol)
	if diff > tol {
		t.Fatalf("%s diverges: %g > %g", name, diff, tol)
	}
}

func randF32(elements int, seed int64) []float32 {
	random := rand.New(rand.NewSource(seed))
	out := make([]float32, elements)
	for i := range out {
		out[i] = 2*random.Float32() - 1
	}
	return out
}

// finiteDiffGradCheck: shared central-difference VJP check.
func finiteDiffGradCheck(t *testing.T, name string, params, analytic []float32, indices []int, h, absTol float64, loss func() float64) {
	t.Helper()
	for _, i := range indices {
		save := params[i]
		params[i] = save + float32(h)
		lp := loss()
		params[i] = save - float32(h)
		lm := loss()
		params[i] = save
		num := (lp - lm) / (2 * h)
		if d := num - float64(analytic[i]); math.Abs(d) > absTol {
			t.Errorf("%s[%d]: analytic=%.6g finite-diff=%.6g (|err| %.2e > %.2e)", name, i, float64(analytic[i]), num, math.Abs(d), absTol)
		}
	}
}

// Shared once-loaded artifact for the real-model gates.
var (
	modelOnce sync.Once
	modelInst *Model
	modelErr  error
)

func artifactDir(t *testing.T) string {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": requires the Un-0 model artifact")
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(roots.Models, "Un-0")
}

func loadArtifactModel(t *testing.T) *Model {
	t.Helper()
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
