// Training-step gates: compiled joint Muon must lower teacher-forced loss.
package speechsynth

import (
	"maps"
	"math/rand"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/checkpointtest"
	"overgo/internal/optimizer"
	"overgo/internal/testcheck"
	"overgo/internal/trainingprogram"
)

func TestJointMuonStepTinyDecreasesLoss(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	m := tinyJointModel(rng)
	const frames, seed = 2, 91
	ids := []int{2, 4, 1}
	z := normVec(rng, frames*m.Dims.LatentDim, 0, 1)
	before := testcheck.Must(m.LossAndGrads(ids, z, frames, seed, nil))(t)
	const baseLR = 1e-3
	lrJoint, nFlow, nJoint := m.DerivedJointLR(baseLR)
	if nFlow >= nJoint || lrJoint >= baseLR || lrJoint <= 0 {
		t.Fatalf("scale law violated: nFlow=%d nJoint=%d lrJoint=%g", nFlow, nJoint, lrJoint)
	}
	trainer := testcheck.Must(NewJointTrainer(m, optimizer.Config{Steps: 1, BaseLearningRate: baseLR, Momentum: 0.95, Schedule: optimizer.ScheduleConstant}, false))(t)
	if trainer.Program().Objective() != trainingprogram.ObjectiveLatentSequence {
		t.Fatalf("trainer objective = %q", trainer.Program().Objective())
	}
	defer trainer.Close()
	stepLoss := testcheck.Must(trainer.Step(TrainingExample{TextIDs: ids, Latents: z, Frames: frames, Seed: seed}))(t)
	after := testcheck.Must(m.LossAndGrads(ids, z, frames, seed, nil))(t)
	t.Logf("tiny Muon: nFlow=%d nJoint=%d lrJoint=%.3e loss %.6f -> %.6f", nFlow, nJoint, lrJoint, before, after)
	if stepLoss != before || !(after < before) {
		t.Fatalf("loss did not decrease: before %.6f step %.6f after %.6f", before, stepLoss, after)
	}
	state := mustJointState(t, trainer)
	weights := jointWeightSnapshot(m)
	checkpointtest.RequireRoundTrip(t, trainer.Program(), trainer, weights)
	resumedModel := tinyJointModel(rand.New(rand.NewSource(31)))
	resumed := testcheck.Must(NewJointTrainer(resumedModel, optimizer.Config{Steps: 1, BaseLearningRate: baseLR, Momentum: 0.95, Schedule: optimizer.ScheduleConstant}, false))(t)
	defer resumed.Close()
	testcheck.NoError(t, resumed.Restore(weights, state))
	next := TrainingExample{TextIDs: ids, Latents: z, Frames: frames, Seed: seed + 1}
	want := testcheck.Must(trainer.Step(next))(t)
	got, err := resumed.Step(next)
	if err != nil || got != want || !maps.EqualFunc(jointWeightSnapshot(m), jointWeightSnapshot(resumedModel), slices.Equal[[]float32]) ||
		!reflect.DeepEqual(mustJointState(t, trainer), mustJointState(t, resumed)) {
		t.Fatalf("native resume differs: loss=%g want=%g error=%v", got, want, err)
	}
}
