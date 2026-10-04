package densecausal

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// TestKTOGradientHoldsTheKLEstimateFixed checks the KTO trainer's parameter
// gradient on a two-example microbatch against central finite differences
// of KTO's loss with the mismatched policy scores, and so the KL estimate
// z0, frozen at their unperturbed values; with z0 recomputed at each
// perturbation the differences depart from the trainer's gradient, so the
// check would see a gradient taken through z0. Tolerance rationale: f32
// forward with f64 score accumulation and h = 1e-3 hold the slopes within
// 2e-6 of the gradient on these probes, while the path through z0 moves
// them by up to 3e-3, so 1e-5 separates the two.
func TestKTOGradientHoldsTheKLEstimateFixed(t *testing.T) {
	golden := readTinyGolden(t)
	first, second := modelFromGolden(t, golden), modelFromGolden(t, golden)
	for name, values := range second.Weights {
		for index := range values {
			values[index] += float32(0.05 * math.Sin(float64(index+len(name))))
		}
	}
	examples := ktoExamples(t, golden.Tokens)
	score := func(model *Model, sequence trainingdata.PreferenceSequence) sequencescore.Score {
		t.Helper()
		result, err := model.SequenceLogProb(sequence)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	// The policy is whichever model gives the mismatched sequences more
	// mass than the other, so z0 is positive rather than clamped to zero.
	var kl float64
	for _, example := range examples {
		kl += score(first, example.Mismatched).LogProbability - score(second, example.Mismatched).LogProbability
	}
	policy, reference := first, second
	if kl < 0 {
		policy, reference = second, first
	}
	kto, err := trainingprogram.KTOPolicy(testutil.ArtifactID(t, artifact.KindModel, "reference"), 1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	observation, grads, err := policy.ktoLossAndGrads(reference, examples, kto)
	if err != nil || observation.KL <= 0 {
		t.Fatalf("KTO step %+v, %v; want a positive KL estimate", observation, err)
	}
	frozen := make([]sequencescore.Score, len(examples))
	for index, example := range examples {
		frozen[index] = score(policy, example.Mismatched)
	}
	loss := func(live bool) float64 {
		scores := make([]trainingprogram.UnpairedScores, len(examples))
		for index, example := range examples {
			scores[index] = trainingprogram.UnpairedScores{
				Desirable: example.Desirable, Policy: score(policy, example.Sequence), Reference: score(reference, example.Sequence),
				MismatchedPolicy: frozen[index], MismatchedReference: score(reference, example.Mismatched),
			}
			if live {
				scores[index].MismatchedPolicy = score(policy, example.Mismatched)
			}
		}
		result, err := kto.UnpairedLoss(scores)
		if err != nil {
			t.Fatal(err)
		}
		return result.Loss
	}
	const h, tolerance = 1e-3, 1e-5
	names := DenseLayerTensorNames(0)
	var departed float64
	for _, probe := range []struct {
		name  string
		index int
	}{
		{policy.tensors.finalNorm.name, 5}, {policy.tensors.embedding.name, golden.Tokens[2]*policy.Dims.Hidden + 3},
		{names.Q, 17}, {names.Down, 40}, {DenseLayerTensorNames(1).Up, 9},
	} {
		values := policy.Weights[probe.name]
		original := values[probe.index]
		slope := func(live bool) float64 {
			values[probe.index] = original + h
			plus := loss(live)
			values[probe.index] = original - h
			minus := loss(live)
			values[probe.index] = original
			return (plus - minus) / (2 * h)
		}
		analytic := float64(grads[probe.name][probe.index])
		held, through := slope(false), slope(true)
		if math.Abs(held-analytic) > tolerance*max(1, math.Abs(held)) {
			t.Errorf("%s[%d]: gradient %g, slope with z0 held %g", probe.name, probe.index, analytic, held)
		}
		departed = max(departed, math.Abs(through-analytic))
		t.Logf("%s[%d]: gradient %.6g, slope with z0 held %.6g, through z0 %.6g", probe.name, probe.index, analytic, held, through)
	}
	if departed <= 10*tolerance {
		t.Fatalf("slopes through z0 depart from the gradient by at most %g; the check cannot see a gradient taken through z0", departed)
	}
}

// ktoExamples are a desirable and an undesirable completion over the
// golden tokens, each with the other's completion spliced after its prompt.
func ktoExamples(t *testing.T, tokens []int) []trainingdata.UnpairedExample {
	t.Helper()
	sequence := func(tokens []int, prompt int) trainingdata.PreferenceSequence {
		mask := make([]bool, len(tokens))
		for index := prompt; index < len(mask); index++ {
			mask[index] = true
		}
		return trainingdata.PreferenceSequence{Tokens: slices.Clone(tokens), Completion: mask}
	}
	examples := []trainingdata.UnpairedExample{
		{ID: "desirable", Desirable: true, Sequence: sequence(tokens[:8], 5)},
		{ID: "undesirable", Sequence: sequence(tokens[4:], 4)},
	}
	for index := range examples {
		var err error
		if examples[index].Mismatched, err = trainingdata.SpliceCompletion(examples[index].Sequence, examples[(index+1)%len(examples)].Sequence); err != nil {
			t.Fatal(err)
		}
	}
	return examples
}

// TestScoreOnlyForwardsKeepNoLayerInputs checks that the scores a
// score-only forward selects equal those of the trace a backward pass
// reads, bit for bit, while it keeps none of the layer inputs the trace
// keeps; logits likewise keep only the final stream.
func TestScoreOnlyForwardsKeepNoLayerInputs(t *testing.T) {
	golden := readTinyGolden(t)
	model := modelFromGolden(t, golden)
	for _, example := range ktoExamples(t, golden.Tokens) {
		trace, err := model.sequenceTrace(example.Sequence)
		if err != nil {
			t.Fatal(err)
		}
		scored, err := model.scoreSequence(example.Sequence, false)
		if err != nil {
			t.Fatal(err)
		}
		if scored.score != trace.score || slices.ContainsFunc(scored.states[:model.Dims.Layers], func(state []float32) bool { return state != nil }) ||
			slices.ContainsFunc(trace.states, func(state []float32) bool { return len(state) == 0 }) {
			t.Fatalf("%s: score-only %+v keeps %d layer inputs, trace %+v", example.ID, scored.score, len(scored.states), trace.score)
		}
	}
	states, err := model.forwardStates(golden.Tokens, false)
	if err != nil {
		t.Fatal(err)
	}
	retained, err := model.forwardStates(golden.Tokens, true)
	if err != nil {
		t.Fatal(err)
	}
	if states[0] != nil || !slices.Equal(states[model.Dims.Layers], retained[model.Dims.Layers]) {
		t.Fatal("a score-only forward kept a layer input or reached another final stream")
	}
}

// TestKTOStepDescends takes one KTO update on the tiny model from a
// reference equal to the policy, where z0 and every reward are zero, and
// requires it to observe the microbatch's loss and lower it.
func TestKTOStepDescends(t *testing.T) {
	golden := readTinyGolden(t)
	policy, reference := modelFromGolden(t, golden), modelFromGolden(t, golden)
	examples := ktoExamples(t, golden.Tokens)
	kto, err := trainingprogram.KTOPolicy(testutil.ArtifactID(t, artifact.KindModel, "reference"), 0.1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	before, _, err := policy.ktoLossAndGrads(reference, examples, kto)
	if err != nil {
		t.Fatal(err)
	}
	batch := trainingdata.UnpairedBatch{Examples: examples, State: trainingdata.StreamState{
		Identity: testutil.ArtifactID(t, artifact.KindProfile, "kto-stream"), Position: uint64(len(examples)),
	}}
	observations, _, err := policy.TrainKTOBatchesResume(reference, []trainingdata.UnpairedBatch{batch}, derivedTestLearningRate(t, policy), 0.9, kto, nil)
	if err != nil {
		t.Fatal(err)
	}
	after, _, err := policy.ktoLossAndGrads(reference, examples, kto)
	if err != nil || len(observations) != 1 || observations[0].Loss != before.Loss || before.KL != 0 || !(after.Loss < before.Loss) ||
		!(after.DesirableReward > after.UndesirableReward) {
		t.Fatalf("observed %+v, loss %g -> %g, rewards after %g and %g, %v", observations, before.Loss, after.Loss, after.DesirableReward, after.UndesirableReward, err)
	}
	for name, values := range reference.Weights {
		if !slices.Equal(values, modelFromGolden(t, golden).Weights[name]) {
			t.Fatalf("reference tensor %q changed", name)
		}
	}
}
