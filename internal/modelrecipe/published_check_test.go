package modelrecipe

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

// TestStoredDocumentsDecodeUnderCurrentSchema pins the failure the published
// check exists for: a chain whose stored bytes the current readers refuse must
// surface an error, never load partially. The alias here names a document of
// the wrong media type -- the same read-side refusal class as a schema that
// moved under a published document.
func TestStoredDocumentsDecodeUnderCurrentSchema(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "published-chain-model")
	if err := VerifyPublishedChain(ctx, store, modelID, recipe.TaskInference); err == nil {
		t.Fatal("chain with no activation alias verified")
	}
	var registered model.ArchitectureProfile
	found := false
	for _, name := range model.SupportedArchitectures() {
		if registered, found = model.LookupArchitecture(name); found {
			break
		}
	}
	if !found {
		t.Fatal("the embedded catalog registers no resolvable architecture")
	}
	document, err := NewProfileDocument(registered)
	if err != nil {
		t.Fatal(err)
	}
	content, err := profileContent(document)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:      "test/published-chain",
		Contents: []artifact.Content{content},
		Aliases:  []artifact.AliasBinding{{Name: activeAlias(modelID, recipe.TaskInference), Target: document.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	err = VerifyPublishedChain(ctx, store, modelID, recipe.TaskInference)
	if err == nil {
		t.Fatal("chain whose alias names a non-recipe document verified")
	}
	if strings.Contains(err.Error(), "panic") {
		t.Fatalf("chain failure is not a typed refusal: %v", err)
	}
}

func TestParseActiveAliasRoundTrip(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "alias-model")
	name := activeAlias(modelID, recipe.TaskInference)
	parsedModel, parsedTask, ok := ParseActiveAlias(name)
	if !ok || parsedModel != modelID || parsedTask != recipe.TaskInference {
		t.Fatalf("round trip = %v %v %v from %q", parsedModel, parsedTask, ok, name)
	}
	for _, invalid := range []string{
		"recipe.status." + modelID.String(),
		"closure/census/latest",
		activeAliasPrefix + "inference",
	} {
		if _, _, ok := ParseActiveAlias(invalid); ok {
			t.Fatalf("%q parsed as an activation alias", invalid)
		}
	}
}
