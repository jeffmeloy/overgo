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

// TestCapabilityCatalogListsNonInferenceActivations closes the "why only
// five servable" finding: a model whose only activation is a capability
// task (forecast here) is invisible to the inference-only Servable query
// but must appear in the capability catalog beside chat models, carrying
// its task and tier.
func TestCapabilityCatalogListsNonInferenceActivations(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	payload := []byte("capability-catalog-weights")
	component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
	location := filepath.Join(t.TempDir(), "forecaster.bin")
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
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "fixture/discovery/capability/active", definition, verification,
		recipe.EvidenceVerified, "capability catalog fixture activation", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	servable, err := Servable(ctx, store, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(servable) != 0 {
		t.Fatalf("servable = %+v, want the forecast-only model invisible to the inference slice", servable)
	}
	entries, truncated, err := CapabilityCatalog(ctx, store, 100, nil)
	if err != nil {
		t.Fatal(err)
	}
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
	capability := entry.Capabilities[0]
	if capability.Task != recipe.TaskForecast || capability.Recipe != definition.ID ||
		capability.Tier != recipe.EvidenceVerified || capability.Stale != "" {
		t.Fatalf("capability = %+v, want a trusted forecast activation at its activated tier", capability)
	}

	// An inference activation is gated by the runtime policy the loader
	// requires: a broken or absent binding (recipes activated before
	// policies existed, or a drifted alias) reports the entry stale (not
	// servable) instead of listing it as launchable, and restoring the
	// binding clears it. Forecast above stays trusted throughout --
	// tasks outside the policy catalog carry no policy requirement.
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
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	if stale := findInference(trusted).Stale; stale != "" {
		t.Fatalf("policy-bound inference stale = %q, want trusted", stale)
	}
	// Point the recipe's policy alias at an artifact the store does not
	// hold -- the loader would refuse this recipe, so the catalog must
	// report it instead of listing the model as launchable.
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
	if err != nil {
		t.Fatal(err)
	}
	if stale := findInference(drifted).Stale; !strings.Contains(stale, "not servable") {
		t.Fatalf("drifted inference stale = %q, want the not-servable report", stale)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:     "fixture/discovery/capability/policy-restore",
		Aliases: []artifact.AliasBinding{{Name: policyAlias, Target: truePolicy.ID, Previous: &bogus}},
	}); err != nil {
		t.Fatal(err)
	}
	restored, _, err := CapabilityCatalog(ctx, store, 100, nil)
	if err != nil {
		t.Fatal(err)
	}
	if stale := findInference(restored).Stale; stale != "" {
		t.Fatalf("restored inference stale = %q, want trusted", stale)
	}
}
