package densecausal

import (
	"cmp"
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostmath"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

func dpoFixture(t *testing.T) (*Model, *Model, trainingdata.PreferencePair) {
	t.Helper()
	golden := readTinyGolden(t)
	policy := modelFromGolden(t, golden)
	reference := modelFromGolden(t, golden)
	chosen := slices.Clone(golden.Tokens)
	rejected := slices.Clone(golden.Tokens)
	rejected[len(rejected)-1] = (rejected[len(rejected)-1] + 1) % policy.Dims.Vocab
	completion := make([]bool, len(chosen))
	completion[len(completion)-1] = true
	return policy, reference, trainingdata.PreferencePair{
		ID: "pair", SharedPrefix: len(chosen) - 1,
		Chosen:   trainingdata.PreferenceSequence{Tokens: chosen, Completion: slices.Clone(completion)},
		Rejected: trainingdata.PreferenceSequence{Tokens: rejected, Completion: completion},
	}
}

func TestDPOSelectedScoresMatchMaterialized(t *testing.T) {
	policy, _, pair := dpoFixture(t)
	got, err := policy.SequenceLogProb(pair.Chosen)
	if err != nil {
		t.Fatal(err)
	}
	_, logits, err := policy.Loss(pair.Chosen.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	row := logits[(len(pair.Chosen.Tokens)-2)*policy.Dims.Vocab : (len(pair.Chosen.Tokens)-1)*policy.Dims.Vocab]
	hostmath.SoftmaxInPlace(row)
	want := math.Log(float64(row[pair.Chosen.Tokens[len(pair.Chosen.Tokens)-1]]))
	if math.Abs(got.LogProbability-want) > 1e-6 {
		t.Fatalf("selected score=%g want=%g", got.LogProbability, want)
	}
}

func TestDPOStepImprovesPreferenceMargin(t *testing.T) {
	policy, reference, pair := dpoFixture(t)
	before := dpoMargin(t, policy, pair)
	if _, _, err := policy.TrainDPOBatchesResume(reference, dpoBatches(t, pair, 1), derivedTestLearningRate(t, policy), 0.9, dpoPolicy, nil, nil); err != nil {
		t.Fatal(err)
	}
	after := dpoMargin(t, policy, pair)
	if after <= before {
		t.Fatalf("preference margin before=%g after=%g", before, after)
	}
}

func dpoMargin(t *testing.T, model *Model, pair trainingdata.PreferencePair) float64 {
	t.Helper()
	chosen, err := model.SequenceLogProb(pair.Chosen)
	if err != nil {
		t.Fatal(err)
	}
	rejected, err := model.SequenceLogProb(pair.Rejected)
	if err != nil {
		t.Fatal(err)
	}
	return chosen.LogProbability - rejected.LogProbability
}

func TestDPOReferenceRemainsFrozen(t *testing.T) {
	policy, reference, pair := dpoFixture(t)
	want := make(map[string][]float32, len(reference.Weights))
	for name, values := range reference.Weights {
		want[name] = slices.Clone(values)
	}
	if _, _, err := policy.TrainDPOBatchesResume(reference, dpoBatches(t, pair, 1), derivedTestLearningRate(t, policy), 0.9, dpoPolicy, nil, nil); err != nil {
		t.Fatal(err)
	}
	for name, values := range want {
		if !slices.Equal(values, reference.Weights[name]) {
			t.Fatalf("reference tensor %q changed", name)
		}
	}
}

func TestDPOResumeMatchesUninterrupted(t *testing.T) {
	uninterrupted, reference, pair := dpoFixture(t)
	resumed, _, _ := dpoFixture(t)
	batches := dpoBatches(t, pair, 2)
	_, wantState, err := uninterrupted.TrainDPOBatchesResume(reference, batches, derivedTestLearningRate(t, uninterrupted), 0.9, dpoPolicy, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, firstState, err := resumed.TrainDPOBatchesResume(reference, batches[:1], derivedTestLearningRate(t, resumed), 0.9, dpoPolicy, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, gotState, err := resumed.TrainDPOBatchesResume(reference, batches[1:], derivedTestLearningRate(t, resumed), 0.9, dpoPolicy, &firstState, nil)
	if err != nil {
		t.Fatal(err)
	}
	if gotState.Stream != wantState.Stream || gotState.Optimizer.Step != wantState.Optimizer.Step ||
		gotState.Optimizer.PlanIdentity != wantState.Optimizer.PlanIdentity ||
		!slices.Equal(gotState.Optimizer.Momentum, wantState.Optimizer.Momentum) {
		t.Fatalf("resumed state differs: got=%+v want=%+v", gotState, wantState)
	}
	for name, want := range uninterrupted.Weights {
		if !slices.Equal(resumed.Weights[name], want) {
			t.Fatalf("resumed tensor %q differs", name)
		}
	}
}

func dpoBatches(t *testing.T, pair trainingdata.PreferencePair, count int) []trainingdata.PreferenceBatch {
	t.Helper()
	identity := testutil.ArtifactID(t, artifact.KindProfile, "dpo-stream")
	batches := make([]trainingdata.PreferenceBatch, count)
	for index := range batches {
		batches[index] = trainingdata.PreferenceBatch{
			Pairs: []trainingdata.PreferencePair{pair},
			State: trainingdata.StreamState{Identity: identity, Position: uint64(index + 1)},
		}
	}
	return batches
}

// dpoPolicy is DPO's own loss mode at β = 0.1.
var dpoPolicy = trainingprogram.PreferencePolicy{Scale: 0.1}

// TestPreferenceModesDescend takes one step of the DPO trainer on the tiny
// model in each loss mode, at the scales the papers train with, and
// requires the step to observe that mode's loss and lower it.
func TestPreferenceModesDescend(t *testing.T) {
	for _, policy := range []trainingprogram.PreferencePolicy{
		dpoPolicy,
		{Mode: trainingprogram.PreferenceSimPO, Scale: 2, Margin: 1},
		{Mode: trainingprogram.PreferenceORPO, Scale: 0.1},
	} {
		t.Run(cmp.Or(string(policy.Mode), "dpo"), func(t *testing.T) {
			model, reference, pair := dpoFixture(t)
			before := preferenceLoss(t, model, reference, pair, policy)
			observations, _, err := model.TrainDPOBatchesResume(reference, dpoBatches(t, pair, 1), derivedTestLearningRate(t, model), 0.9, policy, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			after := preferenceLoss(t, model, reference, pair, policy)
			if len(observations) != 1 || observations[0].Loss != before || !(after < before) {
				t.Fatalf("observed %+v, %s loss before=%g after=%g", observations, policy.Mode, before, after)
			}
			t.Logf("loss %.6f -> %.6f", before, after)
		})
	}
}

// preferenceLoss is the pair's loss under policy, with model as the policy
// and reference as the frozen reference.
func preferenceLoss(t *testing.T, model, reference *Model, pair trainingdata.PreferencePair, policy trainingprogram.PreferencePolicy) float64 {
	t.Helper()
	score := func(scorer *Model, sequence trainingdata.PreferenceSequence) sequencescore.Score {
		result, err := scorer.SequenceLogProb(sequence)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	result, err := policy.Loss(trainingprogram.PreferenceScores{
		PolicyChosen: score(model, pair.Chosen), PolicyRejected: score(model, pair.Rejected),
		ReferenceChosen: score(reference, pair.Chosen), ReferenceRejected: score(reference, pair.Rejected),
	})
	if err != nil {
		t.Fatal(err)
	}
	return result.Loss
}
