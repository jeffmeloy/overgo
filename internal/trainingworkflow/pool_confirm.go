package trainingworkflow

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/trainingdata"
)

// poolConfirm is how a text run's trained model answers on the held-out pool an acceptance fixed, refused before anything
// trains when the pool was not fixed for this objective and this base. The answer reads the trained model, then the base
// loaded again, on the promotion part's documents: both readings come after the charge, so nothing of the part is read
// unpaid for, at the cost of holding two models while it is read.
func poolConfirm(
	ctx context.Context,
	store *overgodb.Store,
	acceptance, objective, base artifact.ID,
	trained *densecausal.Model,
	loadBase func() (*densecausal.Model, error),
	records map[string]textRecord,
) (func(context.Context, artifact.ID) (evaluation.PoolConfirmation, uint64, error), error) {
	pool, err := evaluation.RequireHeldoutPool(ctx, store, acceptance)
	if err != nil {
		return nil, err
	}
	if fixed := pool.Acceptance; fixed.Objective != objective || fixed.Base != base {
		return nil, fmt.Errorf("training workflow: pool %s was fixed for objective %s over base %s, not objective %s over base %s, so it cannot answer for this run",
			acceptance, fixed.Objective, fixed.Base, objective, base)
	}
	sequence := int(pool.Acceptance.Sequence)
	return func(ctx context.Context, candidate artifact.ID) (evaluation.PoolConfirmation, uint64, error) {
		return evaluation.ConfirmOnPool(ctx, store, acceptance, candidate, func(documents []string) ([]float64, error) {
			differences, err := documentLikelihoods(trained, documents, records, sequence)
			if err != nil {
				return nil, err
			}
			model, err := loadBase()
			if err != nil {
				return nil, err
			}
			baseline, err := documentLikelihoods(model, documents, records, sequence)
			if err != nil {
				return nil, err
			}
			for index := range differences {
				differences[index] -= baseline[index]
			}
			return differences, nil
		})
	}, nil
}

// documentLikelihoods is a pool's measure of each document under a model: the mean log-likelihood of its tokens, read as
// far as sequence tokens, the first predicted by nothing.
func documentLikelihoods(model *densecausal.Model, documents []string, records map[string]textRecord, sequence int) ([]float64, error) {
	measures := make([]float64, len(documents))
	for index, document := range documents {
		record, found := records[document]
		if !found {
			return nil, fmt.Errorf("training workflow: pool document %q is no record of the objective", document)
		}
		tokens := record.tokens[:min(len(record.tokens), sequence)]
		completion := make([]bool, len(tokens))
		for at := 1; at < len(completion); at++ {
			completion[at] = true
		}
		score, err := model.SequenceLogProb(trainingdata.PreferenceSequence{Tokens: tokens, Completion: completion})
		if err != nil {
			return nil, fmt.Errorf("training workflow: pool document %q: %w", document, err)
		}
		measures[index] = score.LogProbability / float64(score.Tokens)
	}
	return measures, nil
}
