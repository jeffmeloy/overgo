package main

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/seq2seq"
	"overgo/internal/sequencescore"
	"overgo/internal/seriesforecast"
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

// seq2seqRuntime is an encoder-decoder generator as an evaluation runtime: it
// answers a request with the text its decoder selects, and reports no
// likelihoods, so its suites score what it writes.
type seq2seqRuntime struct {
	generator *seq2seq.Generator
}

// Generate decodes up to the options' bound of new tokens.
func (r seq2seqRuntime) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	text, err := r.generator.Generate(textgeneration.Request{Text: prompt, MaxTokens: options.MaxNewTokens})
	return nil, text, err
}

// ScoreContinuations refuses: the suites of an encoder-decoder model score
// the text it generates.
func (seq2seqRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("evaluate: an encoder-decoder evaluation scores generated text, not continuation likelihoods")
}

// forecastRuntime is a forecasting model as an evaluation runtime: it
// forecasts the steps after a series, as its point channel, and generates and
// scores no text.
type forecastRuntime struct {
	model *seriesforecast.Model
}

// Forecast is the point forecast of the steps after series, as many as asked
// of the model's horizon.
func (r forecastRuntime) Forecast(series []float32, steps int) ([]float32, error) {
	if steps <= 0 || steps > r.model.Dims.Horizon {
		return nil, fmt.Errorf("evaluate: %d steps asked of a model that forecasts %d", steps, r.model.Dims.Horizon)
	}
	forecast, err := r.model.Forecast(series)
	if err != nil {
		return nil, err
	}
	points := make([]float32, steps)
	for step := range points {
		points[step] = forecast[step*r.model.Dims.Quantiles]
	}
	return points, nil
}

// Generate refuses: a forecasting model writes no text.
func (forecastRuntime) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("evaluate: a forecasting model generates no text")
}

// ScoreContinuations refuses: a forecasting model scores no text.
func (forecastRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("evaluate: a forecasting model scores no continuation")
}

// capabilitySession evaluates a capability model under its active recipe:
// the records it publishes bind that recipe, so the scorecard reads them as
// the recipe's held-out quality.
type capabilitySession struct {
	store    *overgodb.Store
	release  func(context.Context) error
	campaign *evaluation.Campaign
}

// openCapabilitySession binds a campaign over the runtime load opens to the
// model's active recipe for task, over the definition that binds that recipe
// to the model's manifest.
func openCapabilitySession(
	ctx context.Context, store *overgodb.Store, value manifest, source mediacapability.Source, task recipe.Task,
	load func() (evaluation.Runtime, func(context.Context) error, error),
) (evaluationSession, error) {
	modelID := source.Inventory.Manifest.ID
	activation, _, err := modelrecipe.ResolveActiveCapability(ctx, store, modelID, task)
	if err != nil {
		return nil, err
	}
	definition := activation.Definition
	// A capability model has no language-model spec to define it: its
	// definition binds the recipe it executes to its model and tensors.
	modelDefinition, err := modelrecipe.PublishTaskModelDefinition(ctx, store, definition.ID)
	if err != nil {
		return nil, err
	}
	environment, err := runrecord.CurrentEnvironment("host", "go")
	if err != nil {
		return nil, err
	}
	runtime, release, err := load()
	if err != nil {
		return nil, err
	}
	campaign, err := evaluation.NewIsolatedCampaign(store, runtime, modelrecipe.ProgramIdentity{
		Model: modelID, Definition: modelDefinition, Recipe: definition.ID, RecipeVersion: definition.Version,
		Placement: recipe.PlacementHost, Runtime: modelrecipe.RuntimeInference,
	}, environment, value.CodeCommit)
	if err != nil {
		return nil, errors.Join(err, release(ctx))
	}
	campaign.WithPrompting(evaluation.PromptingRawCompletion)
	return &capabilitySession{store: store, release: release, campaign: campaign}, nil
}

// capabilitySessions opens a model a manifest names by its task, under the
// model's active recipe for that task.
var capabilitySessions = map[string]func(context.Context, *overgodb.Store, manifest, string) (evaluationSession, error){
	string(recipe.TaskSeq2Seq):  openSeq2SeqSession,
	string(recipe.TaskForecast): openForecastSession,
}

// openThoughtBankSession opens the model at path when it is a Fractale
// generation model, and reports false, opening nothing, when it is not one.
func openThoughtBankSession(ctx context.Context, store *overgodb.Store, value manifest, path string) (evaluationSession, bool, error) {
	source, err := mediacapability.Catalog[recipe.TaskGeneration].Resolve(path)
	if err != nil {
		return nil, false, nil
	}
	session, err := openCapabilitySession(ctx, store, value, source, recipe.TaskGeneration, func() (evaluation.Runtime, func(context.Context) error, error) {
		generator, err := thoughtbank.LoadGenerator(path)
		return thoughtBankRuntime{generator}, generator.Close, err
	})
	return session, true, err
}

// openSeq2SeqSession opens the encoder-decoder model at path under its active
// seq2seq recipe.
func openSeq2SeqSession(ctx context.Context, store *overgodb.Store, value manifest, path string) (evaluationSession, error) {
	source, err := mediacapability.Catalog[recipe.TaskSeq2Seq].Resolve(path)
	if err != nil {
		return nil, fmt.Errorf("evaluate: %s is no seq2seq model: %w", path, err)
	}
	return openCapabilitySession(ctx, store, value, source, recipe.TaskSeq2Seq, func() (evaluation.Runtime, func(context.Context) error, error) {
		generator, err := seq2seq.LoadGenerator(path)
		return seq2seqRuntime{generator}, func(context.Context) error { return nil }, err
	})
}

// openForecastSession opens the forecasting model at path under its active
// forecast recipe.
func openForecastSession(ctx context.Context, store *overgodb.Store, value manifest, path string) (evaluationSession, error) {
	source, err := mediacapability.Catalog[recipe.TaskForecast].Resolve(path)
	if err != nil {
		return nil, fmt.Errorf("evaluate: %s is no forecast model: %w", path, err)
	}
	return openCapabilitySession(ctx, store, value, source, recipe.TaskForecast, func() (evaluation.Runtime, func(context.Context) error, error) {
		model, err := seriesforecast.Load(path)
		return forecastRuntime{model}, func(context.Context) error { return nil }, err
	})
}

// Evaluate campaigns the suite file under the recipe's authorities.
func (s *capabilitySession) Evaluate(ctx context.Context, path string) error {
	return evaluateSuiteFile(ctx, s.campaign, path)
}

// Close releases the runtime and the session's store.
func (s *capabilitySession) Close() error {
	if s == nil {
		return nil
	}
	return errors.Join(s.release(context.Background()), s.store.Close())
}
