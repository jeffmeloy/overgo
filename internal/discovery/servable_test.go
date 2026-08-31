package discovery

import (
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
	publish := func(name string) artifact.ID {
		payload := []byte("discovery-weights-" + name)
		component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
		location := filepath.Join(t.TempDir(), name+".gguf")
		if err := os.WriteFile(location, payload, 0o600); err != nil {
			t.Fatal(err)
		}
		manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
			Role: artifact.ComponentWeights, Name: "weights", Artifact: component,
		}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, artifact.Batch{
			Key:       "fixture/discovery/facts/" + name,
			Artifacts: []artifact.Descriptor{{ID: component, Size: uint64(len(payload))}},
			Manifests: []artifact.Manifest{manifest},
			Locations: []artifact.LocationEvent{{Location: artifact.Location{
				Artifact: component, Kind: artifact.LocationFile, Value: location,
			}, Action: artifact.LocationAdd}},
		}); err != nil {
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

// publishVerifiedActivation walks the trusted lifecycle: candidate,
// validated, verified run evidence, then activation -- the activation
// ActiveRecord accepts.
func publishVerifiedActivation(t *testing.T, store *overgodb.Store, modelID artifact.ID, suffix string) {
	t.Helper()
	ctx := t.Context()
	profile := testutil.ArtifactID(t, artifact.KindProfile, "discovery-profile-"+suffix)
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "discovery-definition-"+suffix)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/verified-facts/" + suffix,
		Artifacts: []artifact.Descriptor{{ID: profile}, {ID: definitionID}},
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
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/discovery/verified-candidate/"+suffix, definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "fixture/discovery/verified-validated/"+suffix, definition, recipe.StatusValidated, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(
		ctx, store, "fixture/discovery/verified-evidence/"+suffix, definition.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "fixture/discovery/verified-active/"+suffix, definition, verification,
		recipe.EvidenceVerified, "discovery fixture activation", nil, nil,
	); err != nil {
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
