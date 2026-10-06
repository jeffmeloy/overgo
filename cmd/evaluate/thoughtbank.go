package main

import (
	"context"
	"errors"

	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/textgeneration"
	"overgo/internal/thoughtbank"
	"overgo/internal/tokenizer"
)

// thoughtBankRuntime is the Fractale generator as an evaluation runtime: its
// greedy generation answers a generation suite, and its sequence likelihood
// answers a sequence-scoring suite.
type thoughtBankRuntime struct {
	generator *thoughtbank.Generator
}

// Generate decodes up to the options' bound of new tokens greedily.
func (r thoughtBankRuntime) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	generation, err := r.generator.Generate(textgeneration.Request{Text: prompt, MaxTokens: options.MaxNewTokens})
	if err != nil {
		return nil, "", err
	}
	tokens := make([]tokenizer.TokenID, len(generation.Tokens))
	for index, token := range generation.Tokens {
		tokens[index] = tokenizer.TokenID(token)
	}
	return tokens, generation.Text, nil
}

// ScoreContinuations scores each candidate as one sequence from the empty bank.
func (r thoughtBankRuntime) ScoreContinuations(ctx context.Context, prompt string, candidates []string) ([]sequencescore.Score, error) {
	return r.generator.ScoreContinuations(ctx, prompt, candidates)
}

// thoughtBankSession evaluates a Fractale model under its active generation
// recipe: the records it publishes bind that recipe, so the scorecard reads
// them as the generation recipe's held-out quality.
type thoughtBankSession struct {
	store     *overgodb.Store
	generator *thoughtbank.Generator
	campaign  *evaluation.Campaign
}

// openThoughtBankSession opens the model at path when it is a Fractale
// generation model, and reports false, opening nothing, when it is not one.
func openThoughtBankSession(ctx context.Context, store *overgodb.Store, value manifest, path string) (evaluationSession, bool, error) {
	source, err := mediacapability.Catalog[recipe.TaskGeneration].Resolve(path)
	if err != nil {
		return nil, false, nil
	}
	modelID := source.Inventory.Manifest.ID
	activation, _, err := modelrecipe.ResolveActiveCapability(ctx, store, modelID, recipe.TaskGeneration)
	if err != nil {
		return nil, true, err
	}
	definition := activation.Definition
	// A capability model has no language-model spec to define it: its
	// definition binds the recipe it executes to its model and tensors.
	modelDefinition, err := modelrecipe.PublishTaskModelDefinition(ctx, store, definition.ID)
	if err != nil {
		return nil, true, err
	}
	environment, err := runrecord.CurrentEnvironment("host", "go")
	if err != nil {
		return nil, true, err
	}
	generator, err := thoughtbank.LoadGenerator(path)
	if err != nil {
		return nil, true, err
	}
	campaign, err := evaluation.NewIsolatedCampaign(store, thoughtBankRuntime{generator}, modelrecipe.ProgramIdentity{
		Model: modelID, Definition: modelDefinition, Recipe: definition.ID, RecipeVersion: definition.Version,
		Placement: recipe.PlacementHost, Runtime: modelrecipe.RuntimeInference,
	}, environment, value.CodeCommit)
	if err != nil {
		return nil, true, errors.Join(err, generator.Close(ctx))
	}
	campaign.WithPrompting(evaluation.PromptingRawCompletion)
	return &thoughtBankSession{store: store, generator: generator, campaign: campaign}, true, nil
}

// Evaluate campaigns the suite file under the generation recipe's authorities.
func (s *thoughtBankSession) Evaluate(ctx context.Context, path string) error {
	return evaluateSuiteFile(ctx, s.campaign, path)
}

// Close releases the generator and the session's store.
func (s *thoughtBankSession) Close() error {
	if s == nil {
		return nil
	}
	return errors.Join(s.generator.Close(context.Background()), s.store.Close())
}
