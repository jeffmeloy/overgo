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
// file at modelPath under each of variants (answer-format/reread), and says for each how many now name an option
// letter. The record and its cases are admitted, and the cases published as development evidence, before the model
// opens (evaluation.PrepareReask); the model opens once for every variant.
func reaskBudgetEnded(ctx context.Context, repository, modelPath, record string, variants []evaluation.AnswerVariant) ([]evaluation.Reasked, error) {
	if modelPath == "" {
		return nil, errors.New("evaluate: -reask names the model file the reading asked with -model-path")
	}
	id, err := artifact.ParseID(record)
	if err != nil {
		return nil, err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return nil, err
	}
	prepared, err := evaluation.PrepareReask(ctx, store, id, variants)
	if err = errors.Join(err, store.Close()); err != nil {
		return nil, err
	}
	runner, err := modelcli.OpenRunner(ctx, repository, modelPath, inference.OpenOptions{})
	if err != nil {
		return nil, err
	}
	asked := make([]evaluation.Reasked, len(variants))
	for index, variant := range variants {
		if asked[index], err = prepared.Ask(ctx, chatShapedRuntime{runner}, variant); err != nil {
			break
		}
	}
	return asked, errors.Join(err, runner.Close())
}
