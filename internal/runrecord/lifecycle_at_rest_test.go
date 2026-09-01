package runrecord

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestGateLifecycleAtRestAdmission(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "at-rest-environment")
	result := testutil.ArtifactID(t, artifact.KindEvidence, "at-rest-result")
	testutil.PublishArtifact(t, store, environment)
	prepared, err := NewGatePreparation(
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		environment, time.Unix(100, 0),
	)
	if err != nil {
		t.Fatal(err)
	}
	preparedContent, err := prepared.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "test/at-rest/prepared", Contents: []artifact.Content{preparedContent},
		Lineage: prepared.Lineage(),
		Aliases: []artifact.AliasBinding{{Name: GateLifecycleCurrentAlias, Target: prepared.ID}},
	}); err != nil {
		t.Fatal(err)
	}

	admit := GateLifecycleAtRest(store)
	if err := admit(overgodb.StoreLocalAliasPrefix+"unknown/current", prepared.ID); err == nil ||
		!strings.Contains(err.Error(), "unknown store-local authority") {
		t.Fatalf("unknown authority error = %v", err)
	}
	if err := admit(GateLifecycleCurrentAlias, prepared.ID); err == nil ||
		!strings.Contains(err.Error(), "not at rest") {
		t.Fatalf("prepared admission error = %v", err)
	}

	testutil.PublishArtifact(t, store, result)
	finalized, err := NewGateFinalization(
		prepared, "0123456789abcdef0123456789abcdef01234567", result, OutcomeSucceeded,
	)
	if err != nil {
		t.Fatal(err)
	}
	finalizedContent, err := finalized.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "test/at-rest/finalized", Contents: []artifact.Content{finalizedContent},
		Lineage: finalized.Lineage(),
		Aliases: []artifact.AliasBinding{{
			Name: GateLifecycleCurrentAlias, Target: finalized.ID, Previous: artifact.IDPointer(prepared.ID),
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := admit(GateLifecycleCurrentAlias, finalized.ID); err != nil {
		t.Fatalf("finalized admission refused: %v", err)
	}
}
