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
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestServableReportsStaleActivationsWithoutFailing closes the wholesale
// -failure finding: a model whose activation cannot be trusted becomes a
// STALE entry naming its defect, and discovery keeps enumerating -- one
// broken activation must not blind the catalog to every healthy model. The
// stale entry is reported and never measured.
func TestServableReportsStaleActivationsWithoutFailing(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	publish := func(name string, payloads ...string) artifact.ID {
		if len(payloads) == 0 {
			payloads = []string{"discovery-weights-" + name}
		}
		batch := artifact.Batch{Key: "fixture/discovery/facts/" + name}
		var components []artifact.Component
		for ordinal, payload := range payloads {
			component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, []byte(payload))
			location := filepath.Join(t.TempDir(), fmt.Sprintf("%s-%d.gguf", name, ordinal))
			testcheck.NoError(t, os.WriteFile(location, []byte(payload), 0o600))
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
		testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
	if len(sized) != 2 || sized[0].Model != healthy || sized[1].Model != split ||
		sized[0].Bytes != int64(len("discovery-weights-healthy")) || sized[1].Bytes != int64(len(shards[0])+len(shards[1])) {
		t.Fatalf("local weights = %+v, want healthy then the complete split model without duplicate shard bytes", sized)
	}

	// Drifting the healthy recipe's runtime-policy alias takes it off the
	// measured surface, exactly as the capability catalog reports it.
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
	testcheck.NoError(t, err)
	for _, entry := range drifted {
		if entry.Model == healthy && !strings.Contains(entry.Stale, "not measured") {
			t.Fatalf("drifted entry = %+v, want it reported stale", entry)
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
	testcheck.NoError(t, err)
	testcheck.NoError(t, modelrecipetest.PublishActivation(t.Context(), store, "fixture/discovery/"+suffix, definition))
}

func TestServableSkipsReplacedArtifactActivation(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	oldPayload, currentPayload := []byte("old-weights"), []byte("current-weights")
	oldComponent := testutil.ArtifactBytesID(t, artifact.KindTensorSet, oldPayload)
	location := filepath.Join(t.TempDir(), "weights.gguf")
	testcheck.NoError(t, os.WriteFile(location, currentPayload, 0o600))
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: oldComponent,
	}})
	testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/discovery/legacy-candidate/"+suffix, definition); err != nil {
		t.Fatal(err)
	}
	_, validated, err := modelrecipe.Transition(
		ctx, store, "fixture/discovery/legacy-validated/"+suffix, definition, recipe.StatusValidated, nil, nil,
	)
	testcheck.NoError(t, err)
	active, err := recipe.NewLifecycleEvent(
		definition, recipe.StatusValidated, recipe.StatusActive, &validated.ID, nil, []artifact.ID{intent},
	)
	testcheck.NoError(t, err)
	content, err := active.Content()
	testcheck.NoError(t, err)
	batch, err := artifact.NewDocumentBatch(
		"fixture/discovery/legacy-active/"+suffix, []artifact.Content{content}, nil,
		[]artifact.AliasBinding{
			{Name: "recipe.status." + definition.ID.String(), Target: active.ID, Previous: &validated.ID},
			{Name: "recipe.active.inference." + modelID.String(), Target: definition.ID},
		},
	)
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
}

// TestEveryPresentModelListsWithItsState: the catalog a person reads names a model nothing activated, with its state,
// beside an activated one, and leaves out what is not here and a hosted model with no activation; the measured surface
// still holds only what was activated.
func TestEveryPresentModelListsWithItsState(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	register := func(name string, kind artifact.LocationKind, present bool) artifact.ID {
		payload := []byte("catalog-weights-" + name)
		component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
		location := filepath.Join(t.TempDir(), name+".gguf")
		if kind == artifact.LocationRemote {
			location = "remote://retired/" + name
		}
		if present {
			testcheck.NoError(t, os.WriteFile(location, payload, 0o600))
		}
		manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{
			{Role: artifact.ComponentWeights, Name: "weights", Artifact: component},
		})
		testcheck.NoError(t, err)
		if _, err := store.Commit(ctx, artifact.Batch{
			Key:       "fixture/discovery/states/" + name,
			Artifacts: []artifact.Descriptor{{ID: component, Size: uint64(len(payload))}},
			Locations: []artifact.LocationEvent{{Location: artifact.Location{
				Artifact: component, Kind: kind, Value: location,
			}, Action: artifact.LocationAdd}},
			Manifests: []artifact.Manifest{manifest},
		}); err != nil {
			t.Fatal(err)
		}
		return manifest.ID
	}
	verified := register("verified", artifact.LocationFile, true)
	publishVerifiedActivation(t, store, verified, "states")
	unverified := register("unverified", artifact.LocationFile, true)
	register("elsewhere", artifact.LocationFile, false)
	register("retired", artifact.LocationRemote, false)
	entries, truncated, err := ModelCatalog(ctx, store, nil)
	if err != nil || truncated {
		t.Fatalf("catalog = %v, truncated %t", err, truncated)
	}
	states := map[artifact.ID]modelrecipe.VerificationState{}
	for _, entry := range entries {
		states[entry.Model] = entry.State.Verification
	}
	if len(states) != 2 || states[unverified] != modelrecipe.StateNotVerified || states[verified] == modelrecipe.StateNotVerified || states[verified] == "" {
		t.Fatalf("states = %v", states)
	}
	measured, err := Servable(ctx, store, 10)
	testcheck.NoError(t, err)
	if len(measured) != 1 || measured[0].Model != verified {
		t.Fatalf("the measured surface holds %+v, want only the activated model", measured)
	}
}
