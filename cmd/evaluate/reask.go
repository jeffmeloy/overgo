package main

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
)

// reaskDefaultTokens is -reask-tokens' default: the chat-template protocol's own budget, so a re-ask with no variant
// named asks as the reading did.
const reaskDefaultTokens = 8

// reaskBudgetEnded asks the budget-ended answers of the chat-template reading record names again through the model
// file at modelPath under variant (answer-format/reread), and says how many now name an option letter.
func reaskBudgetEnded(ctx context.Context, repository, modelPath, record string, variant evaluation.AnswerVariant) (evaluation.Reasked, error) {
	if modelPath == "" {
		return evaluation.Reasked{}, errors.New("evaluate: -reask names the model file the reading asked with -model-path")
	}
	id, err := artifact.ParseID(record)
	if err != nil {
		return evaluation.Reasked{}, err
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return evaluation.Reasked{}, err
	}
	defer store.Close()
	runner, err := modelcli.OpenRunner(ctx, repository, modelPath, inference.OpenOptions{})
	if err != nil {
		return evaluation.Reasked{}, err
	}
	reasked, err := evaluation.ReaskBudgetEnded(ctx, store, id, chatShapedRuntime{runner}, variant)
	return reasked, errors.Join(err, runner.Close())
}
