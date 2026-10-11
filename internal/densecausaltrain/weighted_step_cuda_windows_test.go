//go:build windows

package densecausaltrain

import (
	"maps"
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/densecausal"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
)

// answered is a prompt of five tokens and one answer to it, the answer's
// tokens marked as the completion.
func answered(completion ...int) trainingdata.PreferenceSequence {
	prompt := []int{1, 5, 9, 3, 7}
	mask := make([]bool, len(prompt)+len(completion))
	for index := len(prompt); index < len(mask); index++ {
		mask[index] = true
	}
	return trainingdata.PreferenceSequence{Tokens: append(prompt, completion...), Completion: mask}
}

// apart is the furthest two models' weights stand from one another.
func apart(left, right *densecausal.Model) (furthest float64) {
	for name, values := range left.Weights {
		for index, value := range values {
			furthest = max(furthest, math.Abs(float64(value)-float64(right.Weights[name][index])))
		}
	}
	return furthest
}

// TestOneWeightedStepServesFineTuningAndRewards holds the resident step to
// its three uses, each against the host's own calculation, and to moving
// nothing when it is given nothing.
//
// Fine-tuning. A batch stepped as it always was and the same batch stepped
// as a group of one sequence with the mean written out as weights end at the
// same bytes: weights, momentum and losses. That those are the bytes
// fine-tuning had before the step took weights was checked once, by a digest
// taken on this device before and after.
//
// Rewards. Groups of answers of unequal lengths, weighed by the order of
// their rewards, are stepped on the device and by the host's reward trainer:
// the losses and the weights agree within the trainer's bounds, a shorter
// answer padded with tokens of no weight.
//
// The control. The answers that pass are weighed alike and the rest not at
// all, each passing answer one over their count on every token of its
// completion: the device's step agrees with a step built on the host from
// the masked loss's gradients, a completion's mean times its length. The
// same host gradient is checked against the loss it is the gradient of, for
// weights of both signs, along a direction: the loss a little either side.
//
// Nothing to learn. A group whose rewards are all one, and a group of zero
// weights, take no step: the bytes of the weights and the momentum and the
// count of steps are what they were, and the result counts them skipped. And
// a run resumed after such a group ends where an uninterrupted run does, to
// the byte.
func TestOneWeightedStepServesFineTuningAndRewards(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	const momentum, lossLimit = 0.9, 2e-3
	same := func(left, right *densecausal.Model) bool {
		return maps.EqualFunc(left.Weights, right.Weights, func(a, b []float32) bool { return slices.Equal(a, b) })
	}
	train := func(model *densecausal.Model, resume *densecausal.TrainState, groups ...[]densecausal.WeightedSequence) Result {
		t.Helper()
		result, err := TrainDeviceResidentWeighted(model, worker, groups, derivedTestLearningRate(t, model), momentum, Options{Resume: resume})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}

	// Fine-tuning: the mean written out is the mean.
	batches := [][]int{{1, 5, 9, 3, 7, 2, 11, 4}, {4, 11, 2, 7, 3, 9, 5, 1}, {2, 3, 5, 7, 11, 13, 17, 19}}
	plain, written, hosted := tinyMuonModel(t), tinyMuonModel(t), tinyMuonModel(t)
	before, err := TrainDeviceResident(plain, worker, batches, derivedTestLearningRate(t, plain), momentum, Options{})
	if err != nil {
		t.Fatal(err)
	}
	// How far the device's fine-tuning stands from the host's over as many
	// steps is the trainer's own difference, its step taken in single
	// precision on the device and in double on the host. The weighted uses
	// are held to half as much again, over no more steps.
	if _, _, err := hosted.Train(batches, derivedTestLearningRate(t, hosted), momentum, nil, nil); err != nil {
		t.Fatal(err)
	}
	weightLimit := 1.5 * apart(hosted, plain)
	if weightLimit == 0 || weightLimit > 0.05 {
		t.Fatalf("the device's fine-tuning stands %.2e from the host's after %d steps, which is no bound to hold another use to", weightLimit/1.5, len(batches))
	}
	var meant [][]densecausal.WeightedSequence
	for _, batch := range batches {
		weights := make([]float32, len(batch)-1)
		for index := range weights {
			weights[index] = float32(1) / float32(len(weights))
		}
		meant = append(meant, []densecausal.WeightedSequence{{Tokens: batch, Weights: weights}})
	}
	if after := train(written, nil, meant...); !same(plain, written) || !slices.Equal(before.State.Momentum, after.State.Momentum) || !slices.Equal(before.Losses, after.Losses) || after.Skipped != 0 {
		t.Fatalf("fine-tuning with its mean written out as weights: weights the same %t, momentum the same %t, losses %v against %v", same(plain, written), slices.Equal(before.State.Momentum, after.State.Momentum), after.Losses, before.Losses)
	}

	// Rewards: the device's step against the host's reward trainer.
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "weighted step evaluator")
	stream := testutil.ArtifactID(t, artifact.KindProfile, "weighted step stream")
	rollouts := func(position uint64, rewards ...float64) trainingdata.RolloutGroup {
		t.Helper()
		answers := []trainingdata.PreferenceSequence{answered(2, 11, 4), answered(6, 13), answered(8, 2, 14)}
		members := make([]trainingdata.Rollout, len(rewards))
		for index, reward := range rewards {
			members[index] = trainingdata.Rollout{ID: string(rune('a' + index)), Sequence: answers[index], Reward: reward}
		}
		group, err := trainingdata.NewRolloutGroup("group", evaluator, trainingdata.StreamState{Identity: stream, Position: position}, members)
		if err != nil {
			t.Fatal(err)
		}
		return group
	}
	weighted := func(group trainingdata.RolloutGroup) []densecausal.WeightedSequence {
		t.Helper()
		sequences, err := densecausal.RewardSequences(group)
		if err != nil {
			t.Fatal(err)
		}
		return sequences
	}
	ordered := []trainingdata.RolloutGroup{rollouts(1, 2, 0, 1), rollouts(2, -1, 3, 3), rollouts(3, 0, 5, -2)}
	host, resident := tinyMuonModel(t), tinyMuonModel(t)
	observed, _, err := host.TrainGRPOGroupsResume(ordered, derivedTestLearningRate(t, host), momentum, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	rewarded := train(resident, nil, weighted(ordered[0]), weighted(ordered[1]), weighted(ordered[2]))
	for step, observation := range observed {
		if math.Abs(observation.Loss-rewarded.Losses[step]) > lossLimit {
			t.Errorf("reward group %d: the device's loss %.6f, the host's %.6f", step, rewarded.Losses[step], observation.Loss)
		}
	}
	if furthest := apart(host, resident); furthest > weightLimit || len(rewarded.Losses) != len(observed) || same(resident, tinyMuonModel(t)) {
		t.Fatalf("after %d reward groups the device's weights stand %.2e from the host's, with %d losses; want them within %.1e and moved", len(observed), furthest, len(rewarded.Losses), weightLimit)
	}

	// The control: passing answers weighed alike, against a host step built from the masked loss.
	passing := []trainingdata.PreferenceSequence{answered(2, 11, 4), answered(8, 2, 14)}
	failed := answered(6, 13)
	share := 1 / float64(len(passing))
	control := []densecausal.WeightedSequence{{Tokens: failed.Tokens, Weights: make([]float32, len(failed.Tokens)-1)}}
	reference, controlled := tinyMuonModel(t), tinyMuonModel(t)
	// gradient is the host's gradient of the weights' sum of the completions' losses, and loss that sum.
	gradient := func(model *densecausal.Model, weights []float64) (densecausal.Grads, float64) {
		t.Helper()
		total, loss := densecausal.Grads{}, 0.0
		for index, answer := range passing {
			mean, gradients, err := model.MaskedLossAndGrads(answer.Tokens, answer.Completion)
			if err != nil {
				t.Fatal(err)
			}
			completion := float64(len(answer.Tokens) - 5)
			loss += weights[index] * completion * mean
			for name, values := range gradients {
				if total[name] == nil {
					total[name] = make([]float32, len(values))
				}
				for at, value := range values {
					total[name][at] += float32(weights[index] * completion * float64(value))
				}
			}
		}
		return total, loss
	}
	for _, answer := range passing {
		weights := make([]float32, len(answer.Tokens)-1)
		for token := range weights {
			if answer.Completion[token+1] {
				weights[token] = float32(share)
			}
		}
		control = append(control, densecausal.WeightedSequence{Tokens: answer.Tokens, Weights: weights})
	}
	wanted, wantedLoss := gradient(reference, []float64{share, share})
	update, err := reference.NewDenseUpdate(derivedTestLearningRate(t, reference), momentum, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := update.Pack.GatherGradients(wanted); err != nil {
		t.Fatal(err)
	}
	update.Step()
	update.Pack.Scatter()
	if stepped := train(controlled, nil, control); math.Abs(stepped.Losses[0]-wantedLoss) > lossLimit || apart(reference, controlled) > weightLimit || same(controlled, tinyMuonModel(t)) {
		t.Fatalf("the control's step: loss %.6f against the host's %.6f, weights %.2e from the host's", stepped.Losses[0], wantedLoss, apart(reference, controlled))
	}

	// The host gradient the device is held to, against its own loss, for weights of both signs.
	signed := []float64{0.75, -0.5}
	analytic, _ := gradient(tinyMuonModel(t), signed)
	// The direction is the gradient's own, of unit length, so that what it gives
	// along it is its length; the weights are moved a hundredth of that either way.
	var along float64
	for _, values := range analytic {
		for _, value := range values {
			along += float64(value) * float64(value)
		}
	}
	along = math.Sqrt(along)
	const nudge = 1e-2
	sides := [2]float64{}
	for side, direction := range []float64{nudge, -nudge} {
		moved := tinyMuonModel(t)
		for name, values := range analytic {
			for index, value := range values {
				moved.Weights[name][index] += float32(direction * float64(value) / along)
			}
		}
		_, sides[side] = gradient(moved, signed)
	}
	if measured := (sides[0] - sides[1]) / (2 * nudge); math.Abs(measured-along) > 0.05*along || !(along > 0) {
		t.Fatalf("the host's gradient for signed weights has length %.5f, and the loss a little either side along it rises by %.5f a unit", along, measured)
	}

	// Nothing to learn: no step, and a resume past it as if it had not been there.
	tied, nothing := weighted(rollouts(2, 1, 1, 1)), []densecausal.WeightedSequence{{Tokens: failed.Tokens, Weights: make([]float32, len(failed.Tokens)-1)}}
	once, idle := tinyMuonModel(t), tinyMuonModel(t)
	first := train(once, nil, weighted(ordered[0]))
	passed := train(idle, nil, weighted(ordered[0]), tied, nothing)
	if !same(once, idle) || !slices.Equal(first.State.Momentum, passed.State.Momentum) || passed.State.Step != first.State.Step || passed.Skipped != 2 || len(passed.Losses) != 1 {
		t.Fatalf("after a step and two groups with nothing to learn: weights unmoved %t, momentum unmoved %t, step %d against %d, %d skipped, %d losses",
			same(once, idle), slices.Equal(first.State.Momentum, passed.State.Momentum), passed.State.Step, first.State.Step, passed.Skipped, len(passed.Losses))
	}
	straight := tinyMuonModel(t)
	want := train(straight, nil, weighted(ordered[0]), weighted(ordered[2]))
	if got := train(idle, &passed.State, weighted(ordered[2])); !same(straight, idle) || !slices.Equal(want.State.Momentum, got.State.Momentum) || got.State.Step != want.State.Step {
		t.Fatalf("resumed past groups with nothing to learn: weights as uninterrupted %t, momentum %t, step %d against %d", same(straight, idle), slices.Equal(want.State.Momentum, got.State.Momentum), got.State.Step, want.State.Step)
	}
}
