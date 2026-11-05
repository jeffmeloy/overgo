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
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// activatedForecaster registers a model whose only activation is a verified forecast capability,
// its weights a file on disk, and returns its manifest, its definition and the file's path.
func activatedForecaster(t *testing.T, store *overgodb.Store) (artifact.Manifest, recipe.Definition, string) {
	t.Helper()
	ctx := t.Context()
	payload := []byte("capability-catalog-weights")
	component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
	location := filepath.Join(t.TempDir(), "forecaster.bin")
	testcheck.NoError(t, os.WriteFile(location, payload, 0o600))
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: component,
	}})
	testcheck.NoError(t, err)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/capability/facts",
		Artifacts: []artifact.Descriptor{{ID: component, Size: uint64(len(payload))}},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{
			Artifact: component, Kind: artifact.LocationFile, Value: location,
		}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskForecast, manifest.ID)
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/discovery/capability/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "fixture/discovery/capability/validated", definition, recipe.StatusValidated, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(
		ctx, store, "fixture/discovery/capability/evidence", definition.ID,
	)
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "fixture/discovery/capability/active", definition, verification,
		recipe.EvidenceVerified, "capability catalog fixture activation", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	return manifest, definition, location
}

// TestRememberedCapabilityCatalogPersistsWhatItHashed lists the catalog twice over the store's own
// memo: the first pass hashes the registered weights and publishes their identities, so the memo a
// later pass loads (in this process or another) answers the same file from its stat alone; a pass
// that hashed nothing publishes nothing, so the alias does not move.
func TestRememberedCapabilityCatalogPersistsWhatItHashed(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	manifest, _, location := activatedForecaster(t, store)
	if _, bound, err := artifact.ResolveAlias(ctx, store, IdentityEvidenceAlias); err != nil || bound {
		t.Fatalf("a store nothing has listed holds identity evidence (%t, %v)", bound, err)
	}
	entries, _, err := RememberedCapabilityCatalog(ctx, store, 100)
	if err != nil || len(entries) != 1 || entries[0].Model != manifest.ID || !entries[0].Present {
		t.Fatalf("first listing = (%+v, %v), want the forecast model present", entries, err)
	}
	first, bound, err := artifact.ResolveAlias(ctx, store, IdentityEvidenceAlias)
	if err != nil || !bound {
		t.Fatalf("the first listing published no identity evidence (%t, %v)", bound, err)
	}
	info, err := os.Stat(location)
	testcheck.NoError(t, err)
	if _, hit := LoadMemo(ctx, store).lookup(location+"\x00"+artifact.KindTensorSet.String(), info); !hit {
		t.Error("the memo a later pass loads does not answer the weights from their stat")
	}
	if _, _, err := RememberedCapabilityCatalog(ctx, store, 100); err != nil {
		t.Fatal(err)
	}
	if second, _, err := artifact.ResolveAlias(ctx, store, IdentityEvidenceAlias); err != nil || second != first {
		t.Fatalf("a listing that hashed nothing moved the identity evidence from %v to %v (%v)", first, second, err)
	}
}

// TestRememberedCapabilityCatalogAnswersAStoreItCannotWrite holds a listing over a store opened
// for reading to answering: a model registered since the memo was last kept is hashed, the catalog
// names it present, and the pass does not fail because it cannot keep the digest. The scorecard
// reads the store this way, and it read "store is read-only" after every registration.
func TestRememberedCapabilityCatalogAnswersAStoreItCannotWrite(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Create(root)
	testcheck.NoError(t, err)
	manifest, _, _ := activatedForecaster(t, store)
	testcheck.NoError(t, store.Close())
	reader, err := overgodb.OpenReadOnly(root)
	testcheck.NoError(t, err)
	defer reader.Close()
	entries, _, err := RememberedCapabilityCatalog(ctx, reader, 100)
	if err != nil || len(entries) != 1 || entries[0].Model != manifest.ID || !entries[0].Present {
		t.Fatalf("a read-only listing = (%+v, %v), want the forecast model present", entries, err)
	}
	if _, bound, err := artifact.ResolveAlias(ctx, reader, IdentityEvidenceAlias); err != nil || bound {
		t.Fatalf("a read-only listing kept identity evidence (%t, %v)", bound, err)
	}
}

// TestCapabilityCatalogListsNonInferenceActivations closes the "why only
// five servable" finding: a model whose only activation is a capability
// task (forecast here) is invisible to the inference-only Servable query
// but must appear in the capability catalog beside chat models, carrying
// its task and tier.
func TestCapabilityCatalogListsNonInferenceActivations(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	manifest, definition, location := activatedForecaster(t, store)
	servable, err := Servable(ctx, store, 10)
	testcheck.NoError(t, err)
	if len(servable) != 0 {
		t.Fatalf("servable = %+v, want the forecast-only model invisible to the inference slice", servable)
	}
	entries, truncated, err := CapabilityCatalog(ctx, store, 100, nil)
	testcheck.NoError(t, err)
	if truncated {
		t.Fatal("catalog reported truncation inside its bound")
	}
	if len(entries) != 1 || entries[0].Model != manifest.ID {
		t.Fatalf("entries = %+v, want exactly the forecast model", entries)
	}
	entry := entries[0]
	if !entry.Present || entry.Location != location {
		t.Fatalf("entry = %+v, want recorded bytes present at the fixture location", entry)
	}
	if len(entry.Capabilities) != 1 {
		t.Fatalf("capabilities = %+v, want the single forecast activation", entry.Capabilities)
	}
	if entry.State.Verification == modelrecipe.StateNotVerified {
		t.Fatalf("a model verified for forecasting reads %+v", entry.State)
	}
	capability := entry.Capabilities[0]
	if capability.Task != recipe.TaskForecast || capability.Recipe != definition.ID ||
		capability.Tier != recipe.EvidenceVerified || capability.Stale != "" {
		t.Fatalf("capability = %+v, want a trusted forecast activation at its activated tier", capability)
	}

	// A measurement holds to the policy its recipe binds: a broken or absent
	// binding (a recipe activated before policies existed, or a drifted alias)
	// reports the entry stale, though it runs on the catalog's policy, and
	// restoring the binding clears it. Forecast above stays trusted throughout
	// -- tasks outside the policy catalog carry no policy requirement.
	profile := testutil.ArtifactID(t, artifact.KindProfile, "capability-inference-profile")
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "capability-inference-definition")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/capability/inference-facts",
		Artifacts: []artifact.Descriptor{{ID: profile}, {ID: definitionID}},
	}); err != nil {
		t.Fatal(err)
	}
	inference, err := modelrecipe.InferenceWithModelDefinition(
		manifest.ID, profile, definitionID, recipe.PlacementHost,
		modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference,
	)
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/discovery/capability/inference-candidate", inference); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "fixture/discovery/capability/inference-validated", inference, recipe.StatusValidated, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	inferenceVerification, err := modelrecipetest.PublishVerification(
		ctx, store, "fixture/discovery/capability/inference-evidence", inference.ID,
	)
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "fixture/discovery/capability/inference-active", inference, inferenceVerification,
		recipe.EvidenceVerified, "capability catalog inference fixture activation", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	findInference := func(entries []CatalogEntry) Capability {
		t.Helper()
		for _, entry := range entries {
			for _, capability := range entry.Capabilities {
				if capability.Task == recipe.TaskInference {
					return capability
				}
			}
		}
		t.Fatal("inference capability absent from the catalog")
		return Capability{}
	}
	trusted, _, err := CapabilityCatalog(ctx, store, 100, nil)
	testcheck.NoError(t, err)
	if stale := findInference(trusted).Stale; stale != "" {
		t.Fatalf("policy-bound inference stale = %q, want trusted", stale)
	}
	// Point the recipe's policy alias at an artifact the store does not hold.
	policyAlias := "runtime-policy/v2/" + inference.ID.String()
	truePolicy, supported, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !supported {
		t.Fatalf("inference runtime policy = (%+v, %t, %v)", truePolicy, supported, err)
	}
	bogus := testutil.ArtifactID(t, artifact.KindProfile, "capability-bogus-policy")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/capability/policy-drift",
		Artifacts: []artifact.Descriptor{{ID: bogus}},
		Aliases:   []artifact.AliasBinding{{Name: policyAlias, Target: bogus, Previous: &truePolicy.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	drifted, _, err := CapabilityCatalog(ctx, store, 100, nil)
	testcheck.NoError(t, err)
	if stale := findInference(drifted).Stale; !strings.Contains(stale, "not measured") {
		t.Fatalf("drifted inference stale = %q, want the not-measured report", stale)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:     "fixture/discovery/capability/policy-restore",
		Aliases: []artifact.AliasBinding{{Name: policyAlias, Target: truePolicy.ID, Previous: &bogus}},
	}); err != nil {
		t.Fatal(err)
	}
	restored, _, err := CapabilityCatalog(ctx, store, 100, nil)
	testcheck.NoError(t, err)
	if stale := findInference(restored).Stale; stale != "" {
		t.Fatalf("restored inference stale = %q, want trusted", stale)
	}
}

// TestCapabilityCatalogForTasksSkipsUnselectedPresence requires a
// task-specific lane to hash only models it can consume. Large inference and
// generation artifacts must not delay a training matrix before being
// discarded by its caller.
func TestCapabilityCatalogForTasksSkipsUnselectedPresence(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()

	forecastModel, forecastLocation := publishCatalogCapability(t, store, "forecast", recipe.TaskForecast)
	tabularModel, tabularLocation := publishCatalogCapability(t, store, "tabular", recipe.TaskTabular)
	memo := NewMemo()
	entries, truncated, err := CapabilityCatalogForTasks(ctx, store, 100, memo, recipe.TaskTabular)
	testcheck.NoError(t, err)
	if truncated {
		t.Fatal("task-scoped catalog reported truncation inside its bound")
	}
	if len(entries) != 1 || entries[0].Model != tabularModel || len(entries[0].Capabilities) != 1 ||
		entries[0].Capabilities[0].Task != recipe.TaskTabular {
		t.Fatalf("entries = %+v, want only model %s tabular", entries, tabularModel)
	}
	if !entries[0].Present || entries[0].Location != tabularLocation {
		t.Fatalf("tabular entry = %+v, want present bytes at %s", entries[0], tabularLocation)
	}
	if _, ok := memo.entries[forecastLocation+"\x00"+artifact.KindTensorSet.String()]; ok {
		t.Fatalf("task-scoped catalog hashed unrelated model %s at %s", forecastModel, forecastLocation)
	}
	if len(memo.entries) != 1 {
		t.Fatalf("memoized identities = %d, want only the selected tabular model", len(memo.entries))
	}
	if _, ok := memo.entries[tabularLocation+"\x00"+artifact.KindTensorSet.String()]; !ok {
		t.Fatalf("task-scoped catalog did not hash selected model %s at %s", tabularModel, tabularLocation)
	}
	if _, _, err := CapabilityCatalogForTasks(ctx, store, 100, memo, recipe.Task("invalid")); err == nil {
		t.Fatal("task-scoped catalog accepted an invalid task")
	}
}

func publishCatalogCapability(
	t *testing.T,
	store *overgodb.Store,
	name string,
	task recipe.Task,
) (artifact.ID, string) {
	t.Helper()
	ctx := t.Context()
	payload := []byte("capability-catalog-" + name + "-weights")
	component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
	location := filepath.Join(t.TempDir(), name+".bin")
	testcheck.NoError(t, os.WriteFile(location, payload, 0o600))
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: component,
	}})
	testcheck.NoError(t, err)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/discovery/task-catalog/" + name + "/facts",
		Artifacts: []artifact.Descriptor{{ID: component, Size: uint64(len(payload))}},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{
			Artifact: component, Kind: artifact.LocationFile, Value: location,
		}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(task, manifest.ID)
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/discovery/task-catalog/"+name+"/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "fixture/discovery/task-catalog/"+name+"/validated", definition, recipe.StatusValidated, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(
		ctx, store, "fixture/discovery/task-catalog/"+name+"/evidence", definition.ID,
	)
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "fixture/discovery/task-catalog/"+name+"/active", definition, verification,
		recipe.EvidenceVerified, "task-scoped capability catalog fixture", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	return manifest.ID, location
}
