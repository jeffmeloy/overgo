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
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// JudgePreferenceHeldout judges a finished DPO run on the held-out pairs of
// its recipe objective's split: the base policy the run started from and
// the trained policy it published each score every held-out pair against
// the run's frozen reference, and the verdict passes when the trained
// policy strictly improves the objective's declared preference metric. The
// pairs are read from the run's dataset through the held-out membership
// alone. judged is false for an objective whose split is no committed
// membership: it holds no held-out pairs to judge on.
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
	training, heldout, err := ObjectiveMemberships(ctx, store, objective)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	view, err := evaluation.CompileSFTEvaluationView(objective, training, heldout)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	pairs, err := heldoutPairs(ctx, request, heldout)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	referenceID, err := identifyModel(request.ReferenceDirectory)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	plan, err := evaluation.CompilePreferenceHeldoutPlan(objective, view, trainingprogram.PreferencePolicy{Reference: referenceID, Scale: request.ObjectiveScale})
	if err != nil {
		return HeldoutResult{}, false, err
	}
	reference, err := scoreHeldoutPairs(request.ReferenceDirectory, view, pairs)
	if err != nil {
		return HeldoutResult{}, false, err
	}
	var reports []evaluation.HeldoutReport
	for _, policy := range []string{request.ModelDirectory, request.OutputDirectory} {
		scored, err := scoreHeldoutPairs(policy, view, pairs)
		if err != nil {
			return HeldoutResult{}, false, err
		}
		observations := make([]evaluation.HeldoutObservation, len(scored))
		for index, scores := range scored {
			scores.ReferenceChosen, scores.ReferenceRejected = reference[index].PolicyChosen, reference[index].PolicyRejected
			observations[index] = evaluation.HeldoutObservation{Record: view.Records[index].ID, Preference: scores}
		}
		report, err := plan.Score(observations)
		if err != nil {
			return HeldoutResult{}, false, err
		}
		reports = append(reports, report)
	}
	verdict, err := publishHeldout(ctx, store, objective, view, plan, reports[0], reports[1], result.Observation)
	return verdict, err == nil, err
}

// heldoutPairs reads the held-out membership's pairs from the run's
// dataset through the path training reads its own membership by, keyed by
// record.
func heldoutPairs(ctx context.Context, request Request, heldout dataset.Membership) (map[string]trainingdata.PreferencePair, error) {
	raw, err := os.ReadFile(request.DatasetPath)
	if err != nil {
		return nil, err
	}
	encode, err := loadTrainableEncoder(request.ModelDirectory)
	if err != nil {
		return nil, err
	}
	batches, _, _, _, err := preferenceBatchesResume(ctx, raw, 0, encode, nil, &heldout)
	if err != nil {
		return nil, err
	}
	pairs := make(map[string]trainingdata.PreferencePair, len(batches))
	for _, batch := range batches {
		for _, pair := range batch.Pairs {
			pairs[pair.ID] = pair
		}
	}
	return pairs, nil
}

// scoreHeldoutPairs scores each view record's chosen and rejected responses
// under the model at directory as the pair's policy, in view order.
func scoreHeldoutPairs(directory string, view evaluation.SFTEvaluationView, pairs map[string]trainingdata.PreferencePair) ([]trainingprogram.PreferenceScores, error) {
	model, err := loadTrainableModel(directory)
	if err != nil {
		return nil, err
	}
	scores := make([]trainingprogram.PreferenceScores, len(view.Records))
	for index, record := range view.Records {
		pair, found := pairs[record.ID]
		if !found {
			return nil, fmt.Errorf("training workflow: held-out pair %q is absent from the dataset", record.ID)
		}
		chosen, err := model.SequenceLogProb(pair.Chosen)
		if err != nil {
			return nil, err
		}
		rejected, err := model.SequenceLogProb(pair.Rejected)
		if err != nil {
			return nil, err
		}
		scores[index] = trainingprogram.PreferenceScores{PolicyChosen: chosen, PolicyRejected: rejected}
	}
	return scores, nil
}
