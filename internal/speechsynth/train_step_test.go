// Training-step gates: one optimizer step over the JOINT trained set with
// the derived lrJoint = baseLR*sqrt(nFlow/nJoint) must decrease the
// teacher-forced loss. The tiny gate runs the ported Muon optimizer over the
// flattened joint plan; the real-artifact gate uses plain SGD — the host
// Newton-Schulz on 1024x4096 backbone matrices costs minutes per matrix,
// which buys no extra verification over the tiny Muon gate.
package speechsynth

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/optimizer"
	"overgo/internal/testevidence"
)

// muonJointStep flattens the joint trained set (sorted names), applies one
// Muon step at rate lrJoint, and scatters the result back into the model.
func muonJointStep(t *testing.T, m *Model, g Grads, lrJoint float64) {
	t.Helper()
	tensors, shapes := m.TrainedTensors(true)
	pack, err := optimizer.NewTensorPack(tensors, optimizer.MatrixGeometry(shapes))
	if err != nil {
		t.Fatal(err)
	}
	if err := pack.GatherGradients(g); err != nil {
		t.Fatal(err)
	}
	opt, err := pack.NewOptimizer(optimizer.Config{
		BaseLearningRate: lrJoint, Momentum: 0.95, Schedule: optimizer.ScheduleConstant,
	})
	if err != nil {
		t.Fatal(err)
	}
	result := opt.Step()
	if result.LearningRate != lrJoint {
		t.Fatalf("step rate %g != derived %g", result.LearningRate, lrJoint)
	}
	pack.Scatter()
}

// TestJointMuonStepTinyDecreasesLoss: derived-lr Muon step on the tiny joint
// model; deterministic noise makes before/after comparable.
func TestJointMuonStepTinyDecreasesLoss(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	m := tinyJointModel(rng)
	const frames, seed = 2, 91
	ids := []int{2, 4, 1}
	z := normVec(rng, frames*m.Dims.LatentDim, 0, 1)

	g := Grads{}
	before, err := m.LossAndGrads(ids, z, frames, seed, g)
	if err != nil {
		t.Fatal(err)
	}
	// Test-only probe rate for the flow-alone base (the derivation is the
	// shipped arithmetic; the base is a policy input).
	const baseLR = 1e-3
	lrJoint, nFlow, nJoint := m.DerivedJointLR(baseLR)
	if nFlow >= nJoint || lrJoint >= baseLR || lrJoint <= 0 {
		t.Fatalf("scale law violated: nFlow=%d nJoint=%d lrJoint=%g", nFlow, nJoint, lrJoint)
	}
	muonJointStep(t, m, g, lrJoint)
	after, err := m.LossAndGrads(ids, z, frames, seed, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("tiny muon: nFlow=%d nJoint=%d lrJoint=%.3e loss %.6f -> %.6f", nFlow, nJoint, lrJoint, before, after)
	if !(after < before) {
		t.Fatalf("loss did not decrease: before %.6f after %.6f", before, after)
	}
}

// TestRealArtifactJointTrainingStepDecreasesLoss: full joint backward on the
// real pocket-tts artifact (private copy — the shared instance stays
// pristine for parity tests), one SGD step at the derived lrJoint, loss
// decreases on the same synthetic teacher-forced batch. The mimi encoder is
// NOT needed: latents are synthetic, exactly as in the reference FD gates.
func TestRealArtifactJointTrainingStepDecreasesLoss(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	dir := artifactDir(t)
	if _, err := os.Stat(filepath.Join(dir, configFileName)); err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s; real training step NOT verified", dir)
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	const frames, seed = 3, 17
	ids := []int{1, 2, 3, 5, 8, 13, 21, 34}
	for _, id := range ids {
		if id >= m.Dims.TextVocab {
			t.Fatalf("fixed id %d outside vocab %d", id, m.Dims.TextVocab)
		}
	}
	zRng := rand.New(rand.NewSource(3))
	z := make([]float32, frames*m.Dims.LatentDim)
	for i := range z { // normalized-latent scale
		z[i] = float32(zRng.NormFloat64())
	}

	g := Grads{}
	before, err := m.LossAndGrads(ids, z, frames, seed, g)
	if err != nil {
		t.Fatal(err)
	}
	const baseLR = 1e-3 // test-only probe for the flow-alone base rate
	lrJoint, nFlow, nJoint := m.DerivedJointLR(baseLR)
	var gradNormSq float64
	tensors, _ := m.TrainedTensors(true)
	for name, grad := range g {
		w, ok := tensors[name]
		if !ok || len(w) != len(grad) {
			t.Fatalf("grad %q has no matching trained tensor", name)
		}
		for _, v := range grad {
			gradNormSq += float64(v) * float64(v)
		}
	}
	t.Logf("scale law: nFlow=%d nJoint=%d -> lrJoint=%.3e", nFlow, nJoint, lrJoint)
	t.Logf("loss before step: %.6f  grad tensors: %d  |g|: %.6f", before, len(g), math.Sqrt(gradNormSq))
	for name, grad := range g {
		w := tensors[name]
		for i := range w {
			w[i] -= float32(lrJoint * float64(grad[i]))
		}
	}
	after, err := m.LossAndGrads(ids, z, frames, seed, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loss after step:  %.6f  (delta %.6f)", after, after-before)
	if !(after < before) {
		t.Fatalf("loss did not decrease: before %.6f after %.6f", before, after)
	}
}
