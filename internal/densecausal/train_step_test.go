package densecausal

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
)

// artifactDir resolves a real artifact through the data-root contract;
// absent artifact skips LOUDLY.
func artifactDir(t *testing.T, name string) string {
	t.Helper()
	working, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repo := filepath.Dir(filepath.Dir(working)) // internal/densecausal -> repo root
	roots, err := dataroot.Resolve(repo)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(roots.Models, name)
	// Single-file or sharded-index checkpoints both qualify (OpenSource contract).
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		if _, err := os.Stat(filepath.Join(dir, "model.safetensors.index.json")); err != nil {
			t.Skipf("UNAVAILABLE: %s artifact absent at %s; real-artifact training step NOT verified", name, dir)
		}
	}
	return dir
}

// stepRate: probe step size for the loss-decrease assertion. First-order
// decrease is rate*|g|^2 > 0; small enough that curvature cannot flip the
// sign on a 500M-parameter f32 model.
const stepRate = 1e-3

// runRealArtifactStep loads a real artifact (storage dtype promoted to f32),
// runs one full-parameter backward on a fixed token batch, applies one SGD
// step to EVERY parameter, and asserts the loss on the same batch decreased.
func runRealArtifactStep(t *testing.T, name string) {
	m, err := Load(artifactDir(t, name))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("dims %+v", m.Dims)
	// Fixed token ids (< vocab); no tokenizer dependency for the step gate.
	tokens := []int{1, 4171, 322, 995, 1063, 88, 421, 9822, 15000, 273, 66, 1523, 7, 2094, 511, 2}
	lossBefore, _, grads, err := m.LossAndGrads(tokens)
	if err != nil {
		t.Fatal(err)
	}
	var gradNormSq float64
	for name, g := range grads {
		w, ok := m.Weights[name]
		if !ok || len(w) != len(g) {
			t.Fatalf("grad %q has no matching weight (len %d)", name, len(g))
		}
		for _, v := range g {
			gradNormSq += float64(v) * float64(v)
		}
	}
	t.Logf("loss before step: %.6f  grad tensors: %d  |g|: %.6f", lossBefore, len(grads), math.Sqrt(gradNormSq))
	for name, g := range grads {
		w := m.Weights[name]
		for i := range w {
			w[i] -= float32(stepRate * float64(g[i]))
		}
	}
	lossAfter, _, err := m.Loss(tokens)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loss after step:  %.6f  (delta %.6f)", lossAfter, lossAfter-lossBefore)
	if !(lossAfter < lossBefore) {
		t.Fatalf("loss did not decrease: before %.6f after %.6f", lossBefore, lossAfter)
	}
}

func TestRealArtifactTrainingStepDecreasesLoss(t *testing.T) {
	if testing.Short() {
		t.Skip("loads ~2GB weights and runs a 28-layer host backward; skipped in -short")
	}
	runRealArtifactStep(t, "Carbon-500M")
}
