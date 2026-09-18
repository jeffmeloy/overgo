package main

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay"
	"overgo/internal/sequencescore"
)

// hostedListingLimit bounds the declared hosted models a reference
// resolves against.
const hostedListingLimit = 256

// hostedRuntime: the relay as an evaluation runtime. The prompt travels
// as the user message (a hosted API owns its template, so the answer
// opener rides that message); no likelihood is observable, so suites
// score generatively only.
type hostedRuntime struct {
	*remoterelay.Generator
}

// ShapeChatPrompt returns the prompt itself: the provider shapes it.
func (hostedRuntime) ShapeChatPrompt(prompt string) (string, error) { return prompt, nil }

// ScoreContinuations refuses: a hosted completion exposes no likelihoods.
func (hostedRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("evaluate: a hosted model exposes no continuation likelihoods; its suites score through the generated-letter protocol")
}

// hostedSession evaluates a declared hosted model through the relay: the
// key gates it, the remote environment marks every record as not
// reproducible, and the model's definition binds the provider profile
// and the relay recipe where a local model's architecture would stand.
type hostedSession struct {
	store    *overgodb.Store
	campaign *evaluation.Campaign
	model    artifact.ID
}

func openHostedSession(ctx context.Context, store *overgodb.Store, value manifest, reference string) (evaluationSession, error) {
	provider, modelID, remote, err := remoteprovider.Reference(ctx, store, hostedListingLimit, reference)
	if err != nil {
		return nil, err
	}
	if !remote {
		return nil, fmt.Errorf("evaluate: %s names no declared hosted model", reference)
	}
	if _, err := remoteprovider.Key(provider); err != nil {
		return nil, fmt.Errorf("evaluate: %s is gated on its key: %w", reference, err)
	}
	definition, err := modelrecipe.RemoteInferenceDefinition(modelID, provider.ID)
	if err != nil {
		return nil, err
	}
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskInference)
	if err != nil {
		return nil, err
	}
	if !active || activation.Definition.ID != definition.ID {
		return nil, fmt.Errorf("evaluate: %s has no active relay recipe", reference)
	}
	generator, err := remoterelay.New(provider, definition, nil)
	if err != nil {
		return nil, err
	}
	environment, err := remoteprovider.Environment(provider)
	if err != nil {
		return nil, err
	}
	modelDefinition, err := modelrecipe.PublishRemoteModelDefinition(ctx, store, definition.ID)
	if err != nil {
		return nil, err
	}
	campaign, err := evaluation.NewIsolatedCampaign(store, hostedRuntime{generator}, modelrecipe.ProgramIdentity{
		Model: modelID, Profile: provider.ID, Definition: modelDefinition, Recipe: definition.ID, RecipeVersion: definition.Version,
		Placement: recipe.PlacementHost, Runtime: modelrecipe.RuntimeInference,
	}, environment, value.CodeCommit)
	if err != nil {
		return nil, err
	}
	campaign.WithPrompting(evaluation.PromptingHostedChat)
	return &hostedSession{store: store, campaign: campaign, model: modelID}, nil
}

// Evaluate campaigns the suite file under the hosted authorities.
func (s *hostedSession) Evaluate(ctx context.Context, path string) error {
	return evaluateSuiteFile(ctx, s.campaign, path)
}

// EvaluateDerived campaigns the store's multiple-choice suites (the ones
// the generated-letter protocol scores); every other kind is named and
// not taken. No prompt template is published and no long-form admission
// runs: neither belongs to a model without local weights.
func (s *hostedSession) EvaluateDerived(ctx context.Context, family string) error {
	suites, skipped, err := evaluation.DeriveStoreSuiteFamily(ctx, s.store, s.campaign.Authorities(), family)
	if err != nil {
		return err
	}
	domains, declared, err := modelrecipe.EvalDomains(ctx, s.store, s.model)
	if err != nil {
		return err
	}
	suites = evaluation.FilterSuitesForDomains(suites, domains, declared)
	if len(suites) == 0 {
		fmt.Printf("model domains %v admit no derived suite; nothing to evaluate\n", domains)
		return nil
	}
	selected, err := campaignDerivedSuites(ctx, s.campaign, suites, skipped, family, func(descriptor evaluation.SuiteDescriptor) string {
		if descriptor.Kind != evaluation.MultipleChoiceKind {
			return "a hosted model scores multiple choice through the generated-letter protocol only"
		}
		return ""
	})
	if err != nil {
		return err
	}
	return familyFilterOutcome(selected, declared, domains, family)
}

// Close releases the session's store.
func (s *hostedSession) Close() error {
	if s == nil {
		return nil
	}
	return s.store.Close()
}
