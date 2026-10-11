package densecausal

import (
	"maps"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// rewardGroup is the fixture's pair of answers as one reward group at a
// position of a stream, given the two rewards.
func rewardGroup(t *testing.T, pair trainingdata.PreferencePair, position uint64, chosen, rejected float64) trainingdata.RolloutGroup {
	t.Helper()
	group, err := trainingdata.NewRolloutGroup("group", testutil.ArtifactID(t, artifact.KindEvidence, "reward fixture evaluator"),
		trainingdata.StreamState{Identity: testutil.ArtifactID(t, artifact.KindProfile, "reward fixture stream"), Position: position},
		[]trainingdata.Rollout{{ID: "chosen", Sequence: pair.Chosen, Reward: chosen}, {ID: "rejected", Sequence: pair.Rejected, Reward: rejected}})
	if err != nil {
		t.Fatal(err)
	}
	return group
}

// TestTiedGroupPreservesWeightsMomentumAndStep holds a reward group whose
// answers all stand together to moving nothing.
//
// What it guards against is shown first: after one update on a group with an
// order, the optimizer holds momentum, and a step taken on gradients that are
// all zero moves the weights and counts as a step. So a tied group is given
// no step. After a real update, a tied group leaves the weights and the
// momentum what they were to the bit and the optimizer's step where it was;
// the stream's position passes it, the state counts it skipped, and nothing
// is observed of it. A run resumed from the state after a tied group reaches
// what an uninterrupted run over the same ordered groups reaches, weights,
// momentum and step, with the tied group counted once and not again. And a
// session of nothing but tied groups ends with the weights it began with, no
// step taken, and each group counted.
func TestTiedGroupPreservesWeightsMomentumAndStep(t *testing.T) {
	t.Parallel()
	const momentum = 0.9
	train := func(model *Model, resume *RLState, groups ...trainingdata.RolloutGroup) ([]trainingprogram.GRPOObservation, RLState) {
		t.Helper()
		observations, state, err := model.TrainGRPOGroupsResume(groups, derivedTestLearningRate(t, model), momentum, resume, nil)
		if err != nil {
			t.Fatal(err)
		}
		return observations, state
	}
	same := func(left, right *Model) bool {
		return maps.EqualFunc(left.Weights, right.Weights, func(a, b []float32) bool { return slices.Equal(a, b) })
	}
	once, _, pair := dpoFixture(t)
	ordered := func(position uint64) trainingdata.RolloutGroup { return rewardGroup(t, pair, position, 1, -1) }
	tied := func(position uint64) trainingdata.RolloutGroup { return rewardGroup(t, pair, position, 0.5, 0.5) }

	// One real update, and what a step on zero gradients would then do.
	_, after := train(once, nil, ordered(1))
	if after.Optimizer.Step != 1 || !slices.ContainsFunc(after.Optimizer.Momentum, func(value float64) bool { return value != 0 }) {
		t.Fatalf("one update leaves step %d and no momentum, and the test needs momentum", after.Optimizer.Step)
	}
	stepped, _, _ := dpoFixture(t)
	train(stepped, nil, ordered(1))
	update, err := stepped.NewDenseUpdate(derivedTestLearningRate(t, stepped), momentum, &after.Optimizer)
	if err != nil {
		t.Fatal(err)
	}
	_, zero, err := stepped.grpoLossAndGrads(ordered(2))
	if err != nil {
		t.Fatal(err)
	}
	for _, gradient := range zero {
		clear(gradient)
	}
	if moved, err := update.step(zero); err != nil || moved.Step != 2 || same(stepped, once) {
		t.Fatalf("a step on gradients of zero after momentum = step %d (%v) with the weights unmoved %t; the fault a tied group is kept from is that it moves them", moved.Step, err, same(stepped, once))
	}

	// A tied group after the update: nothing moves but the stream.
	skipping, _, _ := dpoFixture(t)
	observed := 0
	_, state, err := skipping.TrainGRPOGroupsResume([]trainingdata.RolloutGroup{ordered(1), tied(2)}, derivedTestLearningRate(t, skipping), momentum, nil,
		func(trainingprogram.GRPOObservation) { observed++ })
	if err != nil {
		t.Fatal(err)
	}
	if !same(skipping, once) || !slices.Equal(state.Optimizer.Momentum, after.Optimizer.Momentum) || state.Optimizer.Step != 1 ||
		state.Stream.Position != 2 || state.Skipped != 1 || observed != 1 {
		t.Fatalf("after an update and a tied group: weights unmoved %t, momentum unmoved %t, step %d, position %d, %d skipped, %d observed; want the update's own weights and momentum, step 1, position 2, one skipped and one observed",
			same(skipping, once), slices.Equal(state.Optimizer.Momentum, after.Optimizer.Momentum), state.Optimizer.Step, state.Stream.Position, state.Skipped, observed)
	}

	// Resumed from the state after the tie, against a run that never met it.
	_, resumed := train(skipping, &state, ordered(3))
	straight, _, _ := dpoFixture(t)
	_, want := train(straight, nil, ordered(1), ordered(3))
	if !same(skipping, straight) || !slices.Equal(resumed.Optimizer.Momentum, want.Optimizer.Momentum) || resumed.Optimizer.Step != want.Optimizer.Step ||
		resumed.Stream != want.Stream || resumed.Skipped != 1 || want.Skipped != 0 {
		t.Fatalf("resumed past a tied group: weights as uninterrupted %t, momentum %t, step %d against %d, stream %+v against %+v, %d and %d skipped; want the same run with the tied group counted once",
			same(skipping, straight), slices.Equal(resumed.Optimizer.Momentum, want.Optimizer.Momentum), resumed.Optimizer.Step, want.Optimizer.Step, resumed.Stream, want.Stream, resumed.Skipped, want.Skipped)
	}

	// Nothing but tied groups.
	still, _, _ := dpoFixture(t)
	begun, _, _ := dpoFixture(t)
	observations, idle := train(still, nil, tied(1), tied(2))
	if !same(still, begun) || idle.Optimizer.Step != 0 || idle.Skipped != 2 || idle.Stream.Position != 2 || len(observations) != 0 {
		t.Fatalf("a session of tied groups: weights as begun %t, step %d, %d skipped, position %d, %d observed; want nothing moved and both counted", same(still, begun), idle.Optimizer.Step, idle.Skipped, idle.Stream.Position, len(observations))
	}
}
