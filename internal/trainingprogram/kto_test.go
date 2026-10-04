package trainingprogram_test

import (
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/densecausal"
	"overgo/internal/hfbpe"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestKTOModeTrainsOnUnpairedFeedback checks KTO (Ethayarajh et al. 2024)
// as a mode of the DPO objective. Its loss matches values computed by hand,
// with the KL estimate z0 both above zero and clamped to it. Its gradients
// hold z0 fixed: each is the loss's slope in one completion's score with
// the mismatched scores held, although the loss itself moves with them.
// And it trains: a tiny policy, trained through the one DPO workflow and
// recipe on unpaired completions after prompts over a and b, where c is
// desirable and d or dd undesirable, lowers KTO's loss on its training
// records and raises their desirable completions' implied reward
// log πθ(y|x) − log πref(y|x) relative to the undesirable ones'; the
// held-out judge, scoring the records training never read, passes it on
// KTO's held-out loss, and the margin rises there too.
func TestKTOModeTrainsOnUnpairedFeedback(t *testing.T) {
	reference := testutil.ArtifactID(t, artifact.KindModel, "reference")
	t.Run("loss matches hand-computed values", func(t *testing.T) {
		// With β = 1/2, λ_D = 2 and λ_U = 1, one desirable completion's
		// reward sits 2·ln 3 above z0 and one undesirable completion's 2·ln 3
		// below it, so both margins are ln 3, σ(ln 3) = 3/4 and σ(-ln 3) =
		// 1/4. The loss is (2·1/4 + 1·1/4)/2 = 3/8; the gradients are
		// -β·λ_D·(3/4)(1/4)/2 = -3/32 and β·λ_U·(3/4)(1/4)/2 = 3/64. The
		// mismatched rewards 3/4 and 1/4 give z0 = 1/2; -1 and -1/2 give a
		// negative mean, which z0 clamps to 0.
		policy, err := trainingprogram.KTOPolicy(reference, 0.5, 1, 2)
		if err != nil || policy.DesirableWeight != 2 || policy.UndesirableWeight != 1 {
			t.Fatalf("one desirable and two undesirable completions weigh (%g, %g), %v; want (2, 1)", policy.DesirableWeight, policy.UndesirableWeight, err)
		}
		shift := 2 * math.Log(3)
		for _, kl := range []struct {
			name                  string
			z0                    float64
			mismatchedDesirable   [2]float64
			mismatchedUndesirable [2]float64
		}{
			{"positive", 0.5, [2]float64{-4.25, -5}, [2]float64{-5.75, -6}},
			{"clamped", 0, [2]float64{-5, -4}, [2]float64{-6, -5.5}},
		} {
			got, err := policy.UnpairedLoss([]trainingprogram.UnpairedScores{
				{
					Desirable: true, Policy: score(-3 + kl.z0 + shift), Reference: score(-3),
					MismatchedPolicy: score(kl.mismatchedDesirable[0]), MismatchedReference: score(kl.mismatchedDesirable[1]),
				},
				{
					Policy: score(-1 + kl.z0 - shift), Reference: score(-1),
					MismatchedPolicy: score(kl.mismatchedUndesirable[0]), MismatchedReference: score(kl.mismatchedUndesirable[1]),
				},
			})
			const tolerance = 1e-12
			if err != nil || math.Abs(got.Loss-0.375) > tolerance || math.Abs(got.KL-kl.z0) > tolerance ||
				len(got.Gradients) != 2 || math.Abs(got.Gradients[0]+0.09375) > tolerance || math.Abs(got.Gradients[1]-0.046875) > tolerance ||
				got.Desirable != 1 || got.Undesirable != 1 || math.Abs(got.DesirableReward-(kl.z0+shift)) > tolerance ||
				math.Abs(got.UndesirableReward-(kl.z0-shift)) > tolerance {
				t.Fatalf("%s z0: KTO = %+v, %v; want loss 3/8, KL %g, gradients -3/32 and 3/64", kl.name, got, err, kl.z0)
			}
		}
	})
	t.Run("the KL estimate carries no gradient", func(t *testing.T) {
		// A two-example batch whose mismatched rewards average 0.45 > 0.
		policy, err := trainingprogram.KTOPolicy(reference, 0.7, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		scores := []trainingprogram.UnpairedScores{
			{Desirable: true, Policy: score(-2.1), Reference: score(-2.4), MismatchedPolicy: score(-3.3), MismatchedReference: score(-3.9)},
			{Policy: score(-1.6), Reference: score(-1.5), MismatchedPolicy: score(-4.2), MismatchedReference: score(-4.5)},
		}
		result, err := policy.UnpairedLoss(scores)
		if err != nil || result.KL <= 0 {
			t.Fatalf("KTO = %+v, %v; want a positive KL estimate", result, err)
		}
		slope := func(move func([]trainingprogram.UnpairedScores, float64)) float64 {
			const epsilon = 1e-6
			loss := func(delta float64) float64 {
				moved := slices.Clone(scores)
				move(moved, delta)
				value, err := policy.UnpairedLoss(moved)
				if err != nil {
					t.Fatal(err)
				}
				return value.Loss
			}
			return (loss(epsilon) - loss(-epsilon)) / (2 * epsilon)
		}
		var sum float64
		for index, gradient := range result.Gradients {
			sum += gradient
			// The gradient is the slope in the completion's own score with
			// every mismatched score, and so z0, held.
			own := slope(func(moved []trainingprogram.UnpairedScores, delta float64) {
				moved[index].Policy.LogProbability += delta
			})
			if math.Abs(own-gradient) > 1e-8 {
				t.Fatalf("gradient %d = %g, the slope with z0 held %g", index, gradient, own)
			}
		}
		// The loss moves with a mismatched policy score through z0, by
		// -Σ gradients / n, and no gradient is returned for it: the trainer
		// backpropagates only the completions after their own prompts.
		for index := range scores {
			through := slope(func(moved []trainingprogram.UnpairedScores, delta float64) {
				moved[index].MismatchedPolicy.LogProbability += delta
			})
			if want := -sum / float64(len(scores)); math.Abs(through) < 1e-6 || math.Abs(through-want) > 1e-8 {
				t.Fatalf("the loss's slope through z0 in mismatched score %d = %g, want %g and nonzero", index, through, want)
			}
		}
	})
	t.Run("trains on unpaired feedback", func(t *testing.T) {
		fixture := newKTOFixture(t)
		const steps, scale = 12, 0.1
		request := trainingworkflow.Request{
			Repository: fixture.store, Recipe: fixture.recipe, Observations: fixture.store, Host: true,
			ModelDirectory: fixture.policy, ReferenceDirectory: fixture.reference, DatasetPath: fixture.dataset,
			OutputDirectory: filepath.Join(fixture.root, "kto"), Steps: steps,
			ObjectiveScale: scale, PreferenceMode: trainingprogram.PreferenceKTO,
		}
		result, err := trainingworkflow.Execute(t.Context(), request)
		if err != nil || result.Objective != trainingprogram.ObjectiveDPO || len(result.KTO) != steps || len(result.DPO) != 0 {
			t.Fatalf("objective=%s KTO=%d DPO=%d err=%v, want %d KTO steps of the DPO objective", result.Objective, len(result.KTO), len(result.DPO), err, steps)
		}
		for _, observation := range result.KTO {
			if observation.Desirable+observation.Undesirable < 2 || observation.KL < 0 || observation.Step == 0 {
				t.Fatalf("KTO step %+v, want a microbatch of two or more completions and a nonnegative KL estimate", observation)
			}
		}
		verdict, judged, err := trainingworkflow.JudgePreferenceHeldout(t.Context(), fixture.store, request, result)
		if err != nil || !judged || !verdict.Passed {
			t.Fatalf("held-out verdict=%+v judged=%t err=%v, want a passing unpaired verdict", verdict, judged, err)
		}
		base, trained, frozen := loadKTOModel(t, fixture.policy), loadKTOModel(t, request.OutputDirectory), loadKTOModel(t, fixture.reference)
		for _, membership := range []dataset.Membership{fixture.training, fixture.heldout} {
			examples := fixture.examples(t, membership)
			before, after := ktoScores(t, base, frozen, examples), ktoScores(t, trained, frozen, examples)
			desirable, undesirable := 0, 0
			for _, example := range examples {
				if example.Desirable {
					desirable++
				} else {
					undesirable++
				}
			}
			policy, err := trainingprogram.KTOPolicy(reference, scale, desirable, undesirable)
			if err != nil {
				t.Fatal(err)
			}
			beforeLoss, err := policy.UnpairedLoss(before)
			if err != nil {
				t.Fatal(err)
			}
			afterLoss, err := policy.UnpairedLoss(after)
			if err != nil {
				t.Fatal(err)
			}
			beforeMargin := beforeLoss.DesirableReward - beforeLoss.UndesirableReward
			afterMargin := afterLoss.DesirableReward - afterLoss.UndesirableReward
			if !(afterMargin > beforeMargin) || membership.ID == fixture.training.ID && !(afterLoss.Loss < beforeLoss.Loss) {
				t.Fatalf("%s records: KTO loss %g -> %g, desirable-minus-undesirable reward %g -> %g; want the margin to rise and the training loss to fall",
					membership.Partition, beforeLoss.Loss, afterLoss.Loss, beforeMargin, afterMargin)
			}
			t.Logf("%s records (%d desirable, %d undesirable): KTO loss %.6f -> %.6f, reward margin %.6f -> %.6f",
				membership.Partition, desirable, undesirable, beforeLoss.Loss, afterLoss.Loss, beforeMargin, afterMargin)
		}
	})
}

// score is a one-token sequence score.
func score(logProbability float64) sequencescore.Score {
	return sequencescore.Score{LogProbability: logProbability, Tokens: 1}
}

// ktoFixture is the preference fixture over unpaired completions, with each
// record's prompt, completion and label.
type ktoFixture struct {
	preferenceVariantFixture
	documents map[string]map[string]any
}

// newKTOFixture commits one desirable completion c after each prompt, one
// undesirable completion d, and after every third prompt a second
// undesirable dd, grouped by prompt, under an objective that declares the
// unpaired metric.
func newKTOFixture(t *testing.T) ktoFixture {
	t.Helper()
	var documents []map[string]any
	fixture := ktoFixture{documents: map[string]map[string]any{}}
	add := func(id, group, prompt, completion string, desirable bool) {
		document := map[string]any{"id": id, "group": group, "prompt": prompt, "completion": completion, "label": desirable}
		documents = append(documents, document)
		fixture.documents[id] = document
	}
	const secondUndesirableEvery = 3
	for index, prompt := range preferencePrompts {
		group := fmt.Sprintf("group-%02d", index)
		add(fmt.Sprintf("desirable-%02d", index), group, prompt, "c", true)
		add(fmt.Sprintf("undesirable-%02d", index), group, prompt, "d", false)
		if index%secondUndesirableEvery == 0 {
			add(fmt.Sprintf("undesirable-%02d-long", index), group, prompt, "dd", false)
		}
	}
	fixture.preferenceVariantFixture = newPreferenceFixture(t, documents, "overgo/training/preference-token-unpaired/v1", trainingprogram.MetricUnpairedPreferenceLoss)
	return fixture
}

// examples encodes a membership's records with the policy's tokenizer, in
// membership order, each with the next record's completion spliced after
// its prompt.
func (fixture ktoFixture) examples(t *testing.T, membership dataset.Membership) []trainingdata.UnpairedExample {
	t.Helper()
	tokenizer, err := hfbpe.Load(fixture.policy)
	if err != nil {
		t.Fatal(err)
	}
	examples := make([]trainingdata.UnpairedExample, len(membership.Records))
	for index, record := range membership.Records {
		document := fixture.documents[record.ID]
		prompt, err := tokenizer.Encode(document["prompt"].(string))
		if err != nil {
			t.Fatal(err)
		}
		tokens, err := tokenizer.Encode(document["prompt"].(string) + document["completion"].(string))
		if err != nil {
			t.Fatal(err)
		}
		mask := make([]bool, len(tokens))
		for position := len(prompt); position < len(mask); position++ {
			mask[position] = true
		}
		examples[index] = trainingdata.UnpairedExample{
			ID: record.ID, Desirable: document["label"].(bool), Sequence: trainingdata.PreferenceSequence{Tokens: tokens, Completion: mask},
		}
	}
	for index := range examples {
		if examples[index].Mismatched, err = trainingdata.SpliceCompletion(examples[index].Sequence, examples[(index+1)%len(examples)].Sequence); err != nil {
			t.Fatal(err)
		}
	}
	return examples
}

// loadKTOModel loads the dense model at directory.
func loadKTOModel(t *testing.T, directory string) *densecausal.Model {
	t.Helper()
	model, err := densecausal.Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

// ktoScores scores each example and its mismatched sequence under policy
// and reference.
func ktoScores(t *testing.T, policy, reference *densecausal.Model, examples []trainingdata.UnpairedExample) []trainingprogram.UnpairedScores {
	t.Helper()
	sequence := func(model *densecausal.Model, sequence trainingdata.PreferenceSequence) sequencescore.Score {
		result, err := model.SequenceLogProb(sequence)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	scores := make([]trainingprogram.UnpairedScores, len(examples))
	for index, example := range examples {
		scores[index] = trainingprogram.UnpairedScores{
			Desirable: example.Desirable, Policy: sequence(policy, example.Sequence), Reference: sequence(reference, example.Sequence),
			MismatchedPolicy: sequence(policy, example.Mismatched), MismatchedReference: sequence(reference, example.Mismatched),
		}
	}
	return scores
}
