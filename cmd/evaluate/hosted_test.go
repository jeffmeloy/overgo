package main

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay/relaytest"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
)

const hostedTestCommit = "0123456789abcdef0123456789abcdef01234567"

// declareHostedFixture: a fake provider on loopback declared in a fresh
// store under the named key variable; returns the store path and the
// declaration.
func declareHostedFixture(t *testing.T, keyEnvironment string, answers []string) (string, remoteprovider.Declaration) {
	t.Helper()
	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	fake, _ := relaytest.Serve(t, "", "hosted-key", answers)
	t.Setenv(keyEnvironment, "hosted-key")
	declaration, err := remoteprovider.Declare(t.Context(), store, remoteprovider.Provider{
		Name: "fake", Endpoint: fake.URL, KeyEnvironment: keyEnvironment, Model: "vendor/model",
	}, hostedTestCommit)
	if closeErr := store.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if err != nil {
		t.Fatal(err)
	}
	return repository, declaration
}

// TestHostedSessionScoresThroughRelay: a declared hosted model evaluates
// through the relay under the hosted-chat protocol; the question travels
// as the user message with the opener, the record's environment is the
// remote one (not reproducible), the plan binds the remote model
// definition; without the key the session is refused.
func TestHostedSessionScoresThroughRelay(t *testing.T) {
	repository, declaration := declareHostedFixture(t, "OVERGO_EVALUATE_HOSTED_TEST_KEY", []string{"B"})
	session, err := openEvaluationSession(t.Context(), manifest{Repository: repository, CodeCommit: hostedTestCommit},
		modelRequest{Path: declaration.Location, Suites: []string{"derived"}})
	if err != nil {
		t.Fatal(err)
	}
	hosted, ok := session.(*hostedSession)
	if !ok {
		t.Fatalf("session = %T", session)
	}
	source, err := json.Marshal(evaluation.MultipleChoiceSuite{
		Kind: evaluation.MultipleChoiceKind, Schema: "test/hosted/v1", Source: "store/hosted",
		Normalization: sequencescore.NormalizationSum, Aggregation: evaluation.AggregationAccuracy,
		Cases: []evaluation.MultipleChoiceCase{
			{Name: "one", Prompt: "Q1\nAnswer:", Candidates: []string{" A", " B"}, Answer: 1},
			{Name: "two", Prompt: "Q2\nAnswer:", Candidates: []string{" A", " B"}, Answer: 0},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := evaluation.CompileSuite(source, hosted.campaign.Authorities())
	if err != nil {
		t.Fatal(err)
	}
	result, err := hosted.campaign.Evaluate(t.Context(), compiled)
	if err != nil {
		t.Fatal(err)
	}
	var accuracy float64
	for _, metric := range result.Metrics {
		if metric.Name == "accuracy" {
			accuracy = metric.Value
		}
	}
	if accuracy != 0.5 {
		t.Fatalf("metrics = %+v", result.Metrics)
	}
	policy, err := evaluation.ReadExecutionPolicy(t.Context(), hosted.store, compiled.Plan().Execution())
	if err != nil || policy.Prompting != evaluation.PromptingHostedChat {
		t.Fatalf("execution policy = %+v, %v", policy, err)
	}
	evidence, err := evaluation.RequireEvaluationEvidence(t.Context(), hosted.store, result.Evidence)
	if err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.RequireEnvironment(t.Context(), hosted.store, evidence.Environment)
	if err != nil || environment.Reproducible() || environment.Backend != runrecord.BackendRemote {
		t.Fatalf("environment = %+v, %v", environment, err)
	}
	if err := modelrecipe.RequireModelDefinitionBinding(t.Context(), hosted.store, evidence.ModelDefinition, declaration.Model, declaration.Recipe.ID); err != nil {
		t.Fatalf("remote model definition binding: %v", err)
	}
	if _, err := (hostedRuntime{}).ScoreContinuations(t.Context(), "", nil); err == nil || !strings.Contains(err.Error(), "no continuation likelihoods") {
		t.Fatalf("hosted likelihoods: %v", err)
	}
	// The session's store closes before the keyless attempt opens its own.
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OVERGO_EVALUATE_HOSTED_TEST_KEY", "")
	if _, err := openEvaluationSession(t.Context(), manifest{Repository: repository, CodeCommit: hostedTestCommit},
		modelRequest{Path: declaration.Location, Suites: []string{"derived"}}); err == nil || !strings.Contains(err.Error(), "gated on its key") {
		t.Fatalf("keyless hosted session: %v", err)
	}
}

// TestServableModelsListHostedModels: a keyed hosted declaration is an
// evaluation target without bytes on disk and without long-form
// admission.
func TestServableModelsListHostedModels(t *testing.T) {
	repository, declaration := declareHostedFixture(t, "OVERGO_EVALUATE_HOSTED_LIST_KEY", []string{"A"})
	targets, err := servableModels(t.Context(), repository, 16)
	if err != nil || len(targets) != 1 || targets[0].path != declaration.Location || !targets[0].remote || targets[0].text {
		t.Fatalf("targets = %+v, %v", targets, err)
	}
}
