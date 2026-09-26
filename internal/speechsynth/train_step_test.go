// Training-step gates: compiled joint Muon must lower teacher-forced loss.
package speechsynth

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/optimizer"
	"overgo/internal/testskip"
	"overgo/internal/trainingprogram"
)

func TestJointMuonStepTinyDecreasesLoss(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	m := tinyJointModel(rng)
	const frames, seed = 2, 91
	ids := []int{2, 4, 1}
	z := normVec(rng, frames*m.Dims.LatentDim, 0, 1)
	before, err := m.LossAndGrads(ids, z, frames, seed, nil)
	if err != nil {
		t.Fatal(err)
	}
	const baseLR = 1e-3
	lrJoint, nFlow, nJoint := m.DerivedJointLR(baseLR)
	if nFlow >= nJoint || lrJoint >= baseLR || lrJoint <= 0 {
		t.Fatalf("scale law violated: nFlow=%d nJoint=%d lrJoint=%g", nFlow, nJoint, lrJoint)
	}
	trainer, err := NewJointTrainer(m, optimizer.Config{Steps: 1, BaseLearningRate: baseLR, Momentum: 0.95, Schedule: optimizer.ScheduleConstant}, false)
	if err != nil {
		t.Fatal(err)
	}
	if trainer.Program().Objective() != trainingprogram.ObjectiveLatentSequence {
		t.Fatalf("trainer objective = %q", trainer.Program().Objective())
	}
	defer trainer.Close()
	stepLoss, err := trainer.Step(TrainingExample{TextIDs: ids, Latents: z, Frames: frames, Seed: seed})
	if err != nil {
		t.Fatal(err)
	}
	after, err := m.LossAndGrads(ids, z, frames, seed, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("tiny Muon: nFlow=%d nJoint=%d lrJoint=%.3e loss %.6f -> %.6f", nFlow, nJoint, lrJoint, before, after)
	if stepLoss != before || !(after < before) {
		t.Fatalf("loss did not decrease: before %.6f step %.6f after %.6f", before, stepLoss, after)
	}
}

func TestRealArtifactJointMuonTraining(t *testing.T) {
	testRealArtifactJointMuonTraining(t, false)
}

func TestRealArtifactCPUJointMuonTraining(t *testing.T) {
	testRealArtifactJointMuonTraining(t, true)
}

func testRealArtifactJointMuonTraining(t *testing.T, host bool) {
	t.Helper()
	testskip.Short(t)
	dir := artifactDir(t)
	if _, err := os.Stat(filepath.Join(dir, configFileName)); err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s; real training step NOT verified", dir)
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, err := LoadTokenizer(filepath.Join(dir, "tokenizer.model"))
	if err != nil {
		t.Fatal(err)
	}
	golden := loadFixture[g7Golden](t, "g7_generation.json")
	ids, err := tokenizer.Encode(golden.Text)
	if err != nil {
		t.Fatal(err)
	}
	frames := len(golden.QuantizerCalls)
	latents := make([]float32, 0, frames*m.Dims.LatentDim)
	for _, call := range golden.QuantizerCalls {
		latents = append(latents, f32of(call.In.Values)...)
	}
	if frames == 0 || frames > golden.NQuantizerCalls || len(latents) != frames*m.Dims.LatentDim {
		t.Fatalf("native latent geometry traced=%d generated=%d values=%d want=%d", frames, golden.NQuantizerCalls, len(latents), frames*m.Dims.LatentDim)
	}
	before, err := m.LossAndGrads(ids, latents, frames, golden.Seed, nil)
	if err != nil {
		t.Fatal(err)
	}
	const baseLR = 1e-4
	lrJoint, nFlow, nJoint := m.DerivedJointLR(baseLR)
	tensors, _ := m.TrainedTensors(true)
	probe := tensors["flow_lm.flow_net.final_layer.linear.weight"]
	probeBefore := append([]float32(nil), probe...)
	trainer, err := NewJointTrainer(m, optimizer.Config{Steps: 1, BaseLearningRate: baseLR, Momentum: 0.95, Schedule: optimizer.ScheduleConstant}, host)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	started := time.Now()
	stepLoss, err := trainer.Step(TrainingExample{TextIDs: ids, Latents: latents, Frames: frames, Seed: golden.Seed})
	stepWall := time.Since(started)
	if err != nil {
		t.Fatal(err)
	}
	after, err := m.LossAndGrads(ids, latents, frames, golden.Seed, nil)
	if err != nil {
		t.Fatal(err)
	}
	changed := 0
	for index, value := range probe {
		if value != probeBefore[index] {
			changed++
		}
	}
	t.Logf("real native-latent Muon: host=%t text=%q traced_frames=%d generated_frames=%d nFlow=%d nJoint=%d lrJoint=%.3e loss %.6f -> %.6f changed=%d/%d update_wall=%s; traced frames do not establish full-utterance quality", host, golden.Text, frames, golden.NQuantizerCalls, nFlow, nJoint, lrJoint, before, after, changed, len(probe), stepWall)
	if stepLoss != before || !(after < before) || changed == 0 {
		t.Fatalf("loss did not decrease: before %.6f step %.6f after %.6f changed=%d", before, stepLoss, after, changed)
	}
}
