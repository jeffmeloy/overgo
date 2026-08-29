package evaluation

import (
	"context"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

const runtimeLedgerCommit = "0123456789abcdef0123456789abcdef01234567"

type observedRuntime struct {
	generateCalls int
}

func (runtime *observedRuntime) Generate(
	ctx context.Context,
	prompt string,
	options inference.GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	runtime.generateCalls++
	return (exactGenerator{pieces: []string{"o", "k"}}).Generate(ctx, prompt, options)
}

func (*observedRuntime) ScoreContinuations(
	context.Context,
	string,
	[]string,
) ([]sequencescore.Score, error) {
	return nil, nil
}

func TestRecipeEvaluationRuntimeAndLedger(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	identity := modelrecipe.ProgramIdentity{
		Model:      planID(t, artifact.KindModel, "model"),
		Definition: planID(t, artifact.KindModelDefinition, "definition"),
		Recipe:     planID(t, artifact.KindRecipe, "active recipe"),
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/active-recipe",
		Artifacts: []artifact.Descriptor{
			{ID: identity.Model}, {ID: identity.Definition}, {ID: identity.Recipe},
		},
	}); err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture",
		Backend: "fixture", Driver: "fixture",
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime := &observedRuntime{}
	campaign, err := NewCampaign(store, runtime, identity, environment, runtimeLedgerCommit)
	if err != nil {
		t.Fatal(err)
	}
	source, err := json.Marshal(exactFixture())
	if err != nil {
		t.Fatal(err)
	}
	suite, err := CompileSuite(source, campaign.Authorities())
	if err != nil {
		t.Fatal(err)
	}
	result, err := campaign.Evaluate(ctx, suite)
	if err != nil {
		t.Fatal(err)
	}
	history, err := campaign.History(ctx, []CompiledSuite{suite}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.generateCalls != len(exactFixture().Cases) || len(history) != 1 {
		t.Fatalf("runtime calls/history = %d/%d", runtime.generateCalls, len(history))
	}
	entry := history[0]
	if entry.Run != result.Run || entry.Evaluation != result.Evaluation || entry.Report != result.Report ||
		entry.Recipe != identity.Recipe || entry.Dataset != suite.Descriptor().Dataset ||
		entry.Outcome != runrecord.OutcomeSucceeded || len(entry.Metrics) != len(result.Metrics) {
		t.Fatalf("indexed history = %+v; result = %+v", entry, result)
	}
}
