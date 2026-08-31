package gate

import (
	"encoding/json"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const strategyAttemptTestCommit = "0123456789abcdef0123456789abcdef01234567"

func TestGateAttemptBindsStrategyIdentity(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	worker, err := recipe.NewAgentDefinition(recipe.AgentDefinition{
		Name: "gate-strategy", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"),
		ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "model"),
		Policies:    []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "policy")},
	})
	if err != nil {
		t.Fatal(err)
	}
	strategy, err := loop.NewStrategy(worker, testutil.ArtifactID(t, artifact.KindProfile, "catalog"), loop.Config{MaxAttemptsPerStep: 2, MaxInvocations: 5})
	if err != nil {
		t.Fatal(err)
	}
	content, err := strategy.Content()
	if err != nil {
		t.Fatal(err)
	}
	strategyBatch, err := artifact.NewDocumentBatch("fixture/gate-strategy", []artifact.Content{content}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), strategyBatch); err != nil {
		t.Fatal(err)
	}

	t.Setenv(loop.StrategyEnvironment, "display-label")
	t.Setenv(loop.StrategyIDEnvironment, strategy.ID.String())
	gate := &gateContext{planRef: "resource-coverage/strategy-identity-binding", start: time.Now().Add(-time.Second)}
	gate.resolveAttemptStrategy(store)
	var batch artifact.Batch
	if err := gate.appendAttemptRecord(&batch, strategyAttemptTestCommit, testutil.ArtifactID(t, artifact.KindRecipe, "gate-recipe"), testutil.ArtifactID(t, artifact.KindEvidence, "gate-result"), runrecord.OutcomeSucceeded, ""); err != nil {
		t.Fatal(err)
	}
	attempt := attemptFromBatch(t, batch)
	if attempt.StrategyID != strategy.ID || attempt.Strategy != "display-label" || len(gate.honesty) != 0 {
		t.Fatalf("bound attempt = %+v, honesty=%v", attempt, gate.honesty)
	}
	foundStrategyParent := false
	for _, edge := range batch.Lineage {
		foundStrategyParent = foundStrategyParent || edge.Parent == strategy.ID
	}
	if !foundStrategyParent {
		t.Fatal("attempt lineage omitted its resolved strategy")
	}

	t.Setenv(loop.StrategyIDEnvironment, testutil.ArtifactID(t, artifact.KindProfile, "unpublished").String())
	unresolved := &gateContext{planRef: gate.planRef, start: gate.start}
	unresolved.resolveAttemptStrategy(store)
	batch = artifact.Batch{}
	if err := unresolved.appendAttemptRecord(&batch, strategyAttemptTestCommit, testutil.ArtifactID(t, artifact.KindRecipe, "gate-recipe"), testutil.ArtifactID(t, artifact.KindEvidence, "gate-result"), runrecord.OutcomeSucceeded, ""); err != nil {
		t.Fatalf("unresolved attempt was not recordable: %v", err)
	}
	attempt = attemptFromBatch(t, batch)
	if attempt.StrategyID.Valid() || attempt.Strategy != "display-label" || len(unresolved.honesty) != 1 {
		t.Fatalf("unresolved attempt = %+v, honesty=%v", attempt, unresolved.honesty)
	}

	t.Setenv(loop.StrategyIDEnvironment, "")
	labelOnly := &gateContext{planRef: gate.planRef, start: gate.start}
	labelOnly.resolveAttemptStrategy(store)
	batch = artifact.Batch{}
	if err := labelOnly.appendAttemptRecord(&batch, strategyAttemptTestCommit, testutil.ArtifactID(t, artifact.KindRecipe, "gate-recipe"), testutil.ArtifactID(t, artifact.KindEvidence, "gate-result"), runrecord.OutcomeSucceeded, ""); err != nil {
		t.Fatalf("label-only attempt was not recordable: %v", err)
	}
	if attempt := attemptFromBatch(t, batch); attempt.StrategyID.Valid() || attempt.Strategy != "display-label" {
		t.Fatalf("label-only attempt = %+v", attempt)
	}
}

func attemptFromBatch(t *testing.T, batch artifact.Batch) runrecord.AttemptRecord {
	t.Helper()
	for _, content := range batch.Contents {
		if content.Descriptor.MediaType != runrecord.AttemptMediaType || content.Descriptor.Schema != runrecord.AttemptSchema {
			continue
		}
		var attempt runrecord.AttemptRecord
		if err := json.Unmarshal(content.Data, &attempt); err != nil {
			t.Fatal(err)
		}
		return attempt
	}
	t.Fatal("attempt content is absent")
	return runrecord.AttemptRecord{}
}
