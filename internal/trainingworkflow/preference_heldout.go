package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/sequencescore"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// JudgePreferenceHeldout judges a finished DPO run, in any of its loss
// modes, on the held-out records of its recipe objective's split: the base
// policy the run started from and the trained policy it published each
// score every held-out record against the run's frozen reference, and the
// verdict passes when the trained policy strictly improves the objective's
// declared preference metric. Pairs are scored by DPO's own loss; an
// objective that declares the unpaired metric holds out unpaired
// completions, scored by KTO's. The records are read from the run's dataset
// through the held-out membership alone. judged is false for an objective
// whose split is no committed membership: it holds no held-out records to
// judge on.
func JudgePreferenceHeldout(ctx context.Context, store artifact.Repository, request Request, result Result) (_ HeldoutResult, judged bool, _ error) {
	definition, err := recipe.RequireDefinition(ctx, store, request.Recipe)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	objective, split, err := recipeObjective(ctx, store, definition)
	if err != nil || split == nil {
		return HeldoutResult{}, false, err
	}
	if objective.Kind != trainingprogram.ObjectiveDPO || request.ModelDirectory == "" || !result.Observation.Valid() {
		return HeldoutResult{}, false, errors.New("training workflow: a held-out preference verdict needs an observed DPO run from a base policy")
	}
	training, heldout, err := objectiveMemberships(ctx, store, objective)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	view, err := evaluation.CompileSFTEvaluationView(objective, training, heldout)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	base, err := identifyModel(request.ModelDirectory)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	reference, err := identifyModel(request.ReferenceDirectory)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	compile := pairedHeldout
	if objective.Metric == trainingprogram.MetricUnpairedPreferenceLoss {
		compile = unpairedHeldout
	}
	records, err := compile(ctx, request, objective, view, heldout, reference)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	referenceScores, err := scoreHeldout(request.ReferenceDirectory, records.sequences)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	var reports []evaluation.HeldoutReport
	for _, policy := range []string{request.ModelDirectory, request.OutputDirectory} {
		scores, err := scoreHeldout(policy, records.sequences)
		if err != nil {
			return HeldoutResult{}, false, err
		}
		observations := make([]evaluation.HeldoutObservation, len(scores))
		for index := range scores {
			observations[index] = records.observe(index, scores[index], referenceScores[index])
		}
		report, err := records.plan.Score(observations)
		if err != nil {
			return HeldoutResult{}, false, err
		}
		reports = append(reports, report)
	}
	verdict, err := publishHeldout(ctx, store, objective, view, records.plan, reports[0], reports[1], evaluation.HeldoutTraining{
		Observation: result.Observation, Model: base, Trained: result.Checkpoint, Inputs: records.inputs,
	})
	return verdict, err == nil, err
}

// preferenceHeldout is a preference objective's held-out plan, the
// identity of the held-out stream its records were read from, each view
// record's sequences, in view order, and how a record's observation
// assembles from their scores under a policy and under the reference.
type preferenceHeldout struct {
	plan      evaluation.HeldoutPlan
	inputs    artifact.ID
	sequences [][]trainingdata.PreferenceSequence
	observe   func(index int, policy, reference []sequencescore.Score) evaluation.HeldoutObservation
}

// pairedHeldout reads the held-out pairs, judged by DPO's own loss at the
// run's scale: each record scores its chosen and rejected responses.
func pairedHeldout(ctx context.Context, request Request, objective trainingprogram.ObjectiveDocument, view evaluation.SFTEvaluationView, heldout dataset.Membership, reference artifact.ID) (preferenceHeldout, error) {
	raw, encode, err := heldoutDataset(request)
	if err != nil {
		return preferenceHeldout{}, err
	}
	batches, _, stream, _, err := preferenceBatchesResume(ctx, raw, 0, encode, nil, &heldout)
	if err != nil {
		return preferenceHeldout{}, err
	}
	pairs := make(map[string]trainingdata.PreferencePair, len(batches))
	for _, batch := range batches {
		for _, pair := range batch.Pairs {
			pairs[pair.ID] = pair
		}
	}
	records := preferenceHeldout{inputs: stream.Identity, sequences: make([][]trainingdata.PreferenceSequence, len(view.Records))}
	for index, record := range view.Records {
		pair, found := pairs[record.ID]
		if !found {
			return preferenceHeldout{}, fmt.Errorf("training workflow: held-out pair %q is absent from the dataset", record.ID)
		}
		records.sequences[index] = []trainingdata.PreferenceSequence{pair.Chosen, pair.Rejected}
	}
	records.plan, err = evaluation.CompilePreferenceHeldoutPlan(objective, view, trainingprogram.PreferencePolicy{Reference: reference, Scale: request.ObjectiveScale})
	records.observe = func(index int, policy, reference []sequencescore.Score) evaluation.HeldoutObservation {
		return evaluation.HeldoutObservation{Record: view.Records[index].ID, Preference: trainingprogram.PreferenceScores{
			PolicyChosen: policy[0], PolicyRejected: policy[1], ReferenceChosen: reference[0], ReferenceRejected: reference[1],
		}}
	}
	return records, err
}

// unpairedHeldout reads the held-out unpaired completions, judged by KTO's
// loss at the run's scale: each record scores its completion and its
// prompt with the next view record's completion, cyclically.
func unpairedHeldout(ctx context.Context, request Request, objective trainingprogram.ObjectiveDocument, view evaluation.SFTEvaluationView, heldout dataset.Membership, reference artifact.ID) (preferenceHeldout, error) {
	raw, encode, err := heldoutDataset(request)
	if err != nil {
		return preferenceHeldout{}, err
	}
	read, err := unpairedBatchesResume(ctx, raw, 0, encode, nil, &heldout)
	if err != nil {
		return preferenceHeldout{}, err
	}
	examples := make(map[string]trainingdata.UnpairedExample, len(heldout.Records))
	for _, batch := range read.batches {
		for _, example := range batch.Examples {
			examples[example.ID] = example
		}
	}
	viewed := make([]trainingdata.UnpairedExample, len(view.Records))
	labels := make(map[string]bool, len(view.Records))
	for index, record := range view.Records {
		example, found := examples[record.ID]
		if !found {
			return preferenceHeldout{}, fmt.Errorf("training workflow: held-out completion %q is absent from the dataset", record.ID)
		}
		viewed[index], labels[record.ID] = example, example.Desirable
	}
	records := preferenceHeldout{inputs: read.stream.Identity, sequences: make([][]trainingdata.PreferenceSequence, len(viewed))}
	for index, example := range viewed {
		mismatched, err := trainingdata.SpliceCompletion(example.Sequence, viewed[(index+1)%len(viewed)].Sequence)
		if err != nil {
			return preferenceHeldout{}, err
		}
		records.sequences[index] = []trainingdata.PreferenceSequence{example.Sequence, mismatched}
	}
	records.plan, err = evaluation.CompileUnpairedHeldoutPlan(objective, view, reference, request.ObjectiveScale, labels)
	records.observe = func(index int, policy, reference []sequencescore.Score) evaluation.HeldoutObservation {
		return evaluation.HeldoutObservation{
			Record: view.Records[index].ID, Partner: view.Records[(index+1)%len(view.Records)].ID, Unpaired: trainingprogram.UnpairedScores{
				Desirable: viewed[index].Desirable, Policy: policy[0], Reference: reference[0], MismatchedPolicy: policy[1], MismatchedReference: reference[1],
			},
		}
	}
	return records, err
}

// heldoutDataset is the run's dataset and the base policy's encoder.
func heldoutDataset(request Request) ([]byte, func(string) ([]int, error), error) {
	raw, err := os.ReadFile(request.DatasetPath)
	if err != nil {
		return nil, nil, err
	}
	encode, err := loadTrainableEncoder(request.ModelDirectory)
	return raw, encode, err
}

// scoreHeldout scores each record's sequences under the model at directory.
func scoreHeldout(directory string, sequences [][]trainingdata.PreferenceSequence) ([][]sequencescore.Score, error) {
	model, err := loadTrainableModel(directory)
	if err != nil {
		return nil, err
	}
	scores := make([][]sequencescore.Score, len(sequences))
	for index, record := range sequences {
		for _, sequence := range record {
			score, err := model.SequenceLogProb(sequence)
			if err != nil {
				return nil, err
			}
			scores[index] = append(scores[index], score)
		}
	}
	return scores, nil
}
