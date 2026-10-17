package discovery

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestServableReportsStaleActivationsWithoutFailing closes the wholesale
// -failure finding: a model whose activation cannot be trusted becomes a
// STALE entry naming its defect, and discovery keeps enumerating -- one
// broken activation must not blind the catalog to every healthy model. The
// stale entry is reported, never served.
func TestServableReportsStaleActivationsWithoutFailing(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publish := func(name string, payloads ...string) artifact.ID {
		if len(payloads) == 0 {
			payloads = []string{"discovery-weights-" + name}
		}
		batch := artifact.Batch{Key: "fixture/discovery/facts/" + name}
		var components []artifact.Component
		for ordinal, payload := range payloads {
			component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, []byte(payload))
			location := filepath.Join(t.TempDir(), fmt.Sprintf("%s-%d.gguf", name, ordinal))
			if err := os.WriteFile(location, []byte(payload), 0o600); err != nil {
				t.Fatal(err)
			}
			role := artifact.ComponentWeights
			if ordinal != 0 {
				role = artifact.ComponentWeightsShard
			}
			components = append(components, artifact.Component{Role: role, Ordinal: uint32(ordinal), Name: fmt.Sprint("weights-", ordinal), Artifact: component})
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: component, Size: uint64(len(payload))})
			batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: artifact.Location{
				Artifact: component, Kind: artifact.LocationFile, Value: location,
			}, Action: artifact.LocationAdd})
		}
		if len(components) > 1 {
			// Two manifest references to one shard still name one physical payload.
			components = append(components, artifact.Component{Role: artifact.ComponentWeightsShard, Ordinal: uint32(len(components)), Name: "shared-shard", Artifact: components[1].Artifact})
		}
		manifest, err := artifact.NewManifest(artifact.KindModel, components)
		if err != nil {
			t.Fatal(err)
		}
		batch.Manifests = []artifact.Manifest{manifest}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return manifest.ID
	}
	broken := publish("broken")
	publishLegacyActivation(t, store, broken, "present")
	healthy := publish("healthy")
	publishVerifiedActivation(t, store, healthy, "healthy")
	entries, err := Servable(ctx, store, 10)
	if err != nil {
		t.Fatalf("stale activation failed enumeration: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("entries = %+v, want the stale report beside the healthy model", entries)
	}
	byModel := map[artifact.ID]Entry{entries[0].Model: entries[0], entries[1].Model: entries[1]}
	stale := byModel[broken]
	if stale.Stale == "" || !strings.Contains(stale.Stale, "evidence") {
		t.Fatalf("stale entry = %+v, want the broken model reported with its defect", stale)
	}
	served := byModel[healthy]
	if served.Stale != "" || !served.Present || !served.Recipe.Valid() {
		t.Fatalf("healthy entry = %+v, want a served model unaffected by the stale neighbor", served)
	}
	// The split model's entry file is smaller, but its complete weights are larger.
	shards := []string{"x", strings.Repeat("shard", 16)}
	split := publish("split", shards...)
	publishVerifiedActivation(t, store, split, "split")
	sized, err := LocalServable(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	if len(sized) != 2 || sized[0].Model != healthy || sized[1].Model != split ||
		sized[0].Bytes != int64(len("discovery-weights-healthy")) || sized[1].Bytes != int64(len(shards[0])+len(shards[1])) {
		t.Fatalf("local weights = %+v, want healthy then the complete split model without duplicate shard bytes", sized)
	}

	// Drifting the healthy recipe's runtime-policy alias makes the
	// loader refuse it, so the servable listing must report the entry
	// stale exactly as the capability catalog does.
	truePolicy, supported, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !supported {
		t.Fatalf("inference runtime policy = (%t, %v)", supported, err)
	}
	bogus := testutil.ArtifactID(t, artifact.KindProfile, "servable-bogus-policy")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/servable/policy-drift",
		Artifacts: []artifact.Descriptor{{ID: bogus}},
		Aliases: []artifact.AliasBinding{{
			Name: "runtime-policy/v2/" + served.Recipe.String(), Target: bogus, Previous: &truePolicy.ID,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	drifted, err := Servable(ctx, store, 10)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range drifted {
		if entry.Model == healthy && !strings.Contains(entry.Stale, "not servable") {
			t.Fatalf("drifted entry = %+v, want the loader's refusal reported stale", entry)
		}
	}
}

func publishVerifiedActivation(t *testing.T, store *overgodb.Store, modelID artifact.ID, suffix string) {
	t.Helper()
	profile := testutil.ArtifactID(t, artifact.KindProfile, "discovery-profile-"+suffix)
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "discovery-definition-"+suffix)
	testutil.PublishArtifact(t, store, profile)
	testutil.PublishArtifact(t, store, definitionID)
	definition, err := modelrecipe.InferenceWithModelDefinition(modelID, profile, definitionID,
		recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipetest.PublishActivation(t.Context(), store, "fixture/discovery/"+suffix, definition); err != nil {
		t.Fatal(err)
	}
}

func TestServableSkipsReplacedArtifactActivation(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	oldPayload, currentPayload := []byte("old-weights"), []byte("current-weights")
	oldComponent := testutil.ArtifactBytesID(t, artifact.KindTensorSet, oldPayload)
	location := filepath.Join(t.TempDir(), "weights.gguf")
	if err := os.WriteFile(location, currentPayload, 0o600); err != nil {
		t.Fatal(err)
	}
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: oldComponent,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/replaced",
		Artifacts: []artifact.Descriptor{{ID: oldComponent, Size: uint64(len(oldPayload))}},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{
			Artifact: oldComponent, Kind: artifact.LocationFile, Value: location,
		}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	publishLegacyActivation(t, store, manifest.ID, "replaced")
	entries, err := Servable(ctx, store, 10)
	if err != nil || len(entries) != 0 {
		t.Fatalf("servable replaced artifact = (%+v, %v)", entries, err)
	}
}

func publishLegacyActivation(t *testing.T, store artifact.Repository, modelID artifact.ID, suffix string) {
	t.Helper()
	ctx := t.Context()
	profile := testutil.ArtifactID(t, artifact.KindProfile, "discovery-profile-"+suffix)
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "discovery-definition-"+suffix)
	intent := testutil.ArtifactID(t, artifact.KindEvidence, "discovery-legacy-intent-"+suffix)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/legacy-facts/" + suffix,
		Artifacts: []artifact.Descriptor{{ID: profile}, {ID: definitionID}, {ID: intent}},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		modelID, profile, definitionID, recipe.PlacementHost,
		modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/discovery/legacy-candidate/"+suffix, definition); err != nil {
		t.Fatal(err)
	}
	_, validated, err := modelrecipe.Transition(
		ctx, store, "fixture/discovery/legacy-validated/"+suffix, definition, recipe.StatusValidated, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	active, err := recipe.NewLifecycleEvent(
		definition, recipe.StatusValidated, recipe.StatusActive, &validated.ID, nil, []artifact.ID{intent},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := active.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"fixture/discovery/legacy-active/"+suffix, []artifact.Content{content}, nil,
		[]artifact.AliasBinding{
			{Name: "recipe.status." + definition.ID.String(), Target: active.ID, Previous: &validated.ID},
			{Name: "recipe.active.inference." + modelID.String(), Target: definition.ID},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
}
