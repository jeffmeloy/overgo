// Training-step gates: compiled joint Muon must lower teacher-forced loss.
package speechsynth

import (
	"math/rand"
	"testing"

	"overgo/internal/optimizer"
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
