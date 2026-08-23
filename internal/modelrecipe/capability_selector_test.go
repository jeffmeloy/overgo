package modelrecipe

import (
	"context"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type capabilitySelectorFixture struct {
	store      *repodb.Store
	alias      string
	model      artifact.ID
	definition recipe.Definition
	bytes      uint64
}

func newCapabilitySelectorFixture(t *testing.T) capabilitySelectorFixture {
	t.Helper()
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	weights := []byte("selector-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "selector/model",
		Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Manifests: []artifact.Manifest{manifest},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := CapabilityDefinition(recipe.TaskGeneration, manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "selector/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification := publishVerification(t, store, definition.ID, "selector/verification")
	if err := ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceParity, "selector fixture",
	); err != nil {
		t.Fatal(err)
	}
	const alias = "capability/generation/default"
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:     "selector/alias",
		Aliases: []artifact.AliasBinding{{Name: alias, Target: manifest.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	return capabilitySelectorFixture{
		store: store, alias: alias, model: manifest.ID, definition: definition, bytes: uint64(len(weights)),
	}
}

func TestCapabilityEvidenceSelector(t *testing.T) {
	ctx := context.Background()
	fixture := newCapabilitySelectorFixture(t)
	identities := make(map[artifact.ID]struct{})
	sessions := []SessionSelection{SessionPin, SessionWarm}
	for _, session := range sessions {
		selected, err := ResolveCapabilityEvidenceSelector(ctx, fixture.store, CapabilityEvidenceSelector{
			Alias: fixture.alias, Task: recipe.TaskGeneration, Session: session,
		})
		if err != nil {
			t.Fatal(err)
		}
		if selected.Activation.Definition.ID != fixture.definition.ID ||
			selected.Program.Definition().ID != fixture.definition.ID ||
			selected.Resources.ArtifactBytes != fixture.bytes ||
			len(selected.Resources.Components) != 1 ||
			selected.Resources.Components[0].Model != fixture.model ||
			selected.Resources.Components[0].Session != recipe.SessionCapacity {
			t.Fatalf("%s selection=%+v", session, selected)
		}
		identities[selected.Identity] = struct{}{}
	}
	if len(identities) != len(sessions) {
		t.Fatalf("selector identities=%d", len(identities))
	}
	if _, err := ResolveCapabilityEvidenceSelector(ctx, fixture.store, CapabilityEvidenceSelector{
		Alias: fixture.alias, Task: recipe.TaskGeneration, Session: SessionSpillover,
	}); err == nil {
		t.Fatal("evidence-free spillover accepted")
	}
	if _, err := ResolveCapabilityEvidenceSelector(ctx, fixture.store, CapabilityEvidenceSelector{
		Alias: "capability/generation/missing", Task: recipe.TaskGeneration, Session: SessionWarm,
	}); err == nil {
		t.Fatal("missing model alias accepted")
	}
}

func TestCapabilityExecutionAuthorityTracksAliasWithoutChangingExecution(t *testing.T) {
	ctx := context.Background()
	fixture := newCapabilitySelectorFixture(t)
	const alternateAlias = "capability/generation/alternate"
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key:     "selector/alternate-alias",
		Aliases: []artifact.AliasBinding{{Name: alternateAlias, Target: fixture.model}},
	}); err != nil {
		t.Fatal(err)
	}
	resolve := func(alias string) CapabilityEvidenceSelection {
		selection, err := ResolveCapabilityEvidenceSelector(ctx, fixture.store, CapabilityEvidenceSelector{
			Alias: alias, Task: recipe.TaskGeneration, Session: SessionWarm,
		})
		if err != nil {
			t.Fatal(err)
		}
		return selection
	}
	primary, alternate := resolve(fixture.alias), resolve(alternateAlias)
	if primary.Identity == alternate.Identity || primary.Scope == alternate.Scope ||
		!sameCapabilityExecution(primary, alternate) {
		t.Fatalf("alias executions = %+v / %+v", primary, alternate)
	}
	previous := fixture.model
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key: "selector/retire-primary-alias",
		Aliases: []artifact.AliasBinding{{
			Name: fixture.alias, Target: fixture.model, Previous: &previous, Remove: true,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := RefreshCapabilityExecution(ctx, fixture.store, primary); err == nil {
		t.Fatal("retired alias authority remained current")
	}
	if current, err := RefreshCapabilityExecution(ctx, fixture.store, alternate); err != nil || current.Identity != alternate.Identity {
		t.Fatalf("alternate authority = %+v, %v", current, err)
	}
}

func TestRemotePeerSpilloverRequiresCompatibilityEvidence(t *testing.T) {
	ctx := context.Background()
	fixture := newCapabilitySelectorFixture(t)
	local := testutil.ArtifactID(t, artifact.KindEvidence, "selector-local-environment")
	peer := testutil.ArtifactID(t, artifact.KindEvidence, "selector-peer-environment")
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key: "selector/environments", Artifacts: []artifact.Descriptor{{ID: local}, {ID: peer}},
	}); err != nil {
		t.Fatal(err)
	}
	observe := func(environment artifact.ID) runrecord.ServingObservation {
		observation, err := runrecord.PublishServingObservation(ctx, fixture.store, runrecord.ServingObservation{
			Model: fixture.model, Recipe: fixture.definition.ID, Environment: environment,
			Task: recipe.TaskGeneration, Outcome: runrecord.OutcomeSucceeded,
			StartedUnixNS: time.Now().UnixNano(), MeasuredNS: uint64(time.Nanosecond),
		})
		if err != nil {
			t.Fatal(err)
		}
		return observation
	}
	localObservation, peerObservation := observe(local), observe(peer)
	localSelection, err := ResolveCapabilityEvidenceSelector(ctx, fixture.store, CapabilityEvidenceSelector{
		Alias: fixture.alias, Task: recipe.TaskGeneration, Session: SessionWarm,
	})
	if err != nil {
		t.Fatal(err)
	}
	resources := localSelection.Resources
	resourceContent, err := resources.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(ctx, artifact.Batch{Key: "selector/resources", Contents: []artifact.Content{resourceContent}}); err != nil {
		t.Fatal(err)
	}
	compatibility, err := remotePeerCompatibilityCodec.New(RemotePeerCompatibility{
		Version: remotePeerCompatibilityVersion,
		Model:   fixture.model, Recipe: fixture.definition.ID, Resources: resources.Identity, Task: recipe.TaskGeneration,
		LocalEnvironment: local, PeerEnvironment: peer,
		LocalObservation: localObservation.ID, PeerObservation: peerObservation.ID,
		Endpoint: "https://peer.example/v1/generation",
	})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := compatibility.batch("selector/peer-compatibility")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	selected, err := ResolveCapabilityEvidenceSelector(ctx, fixture.store, CapabilityEvidenceSelector{
		Alias: fixture.alias, Task: recipe.TaskGeneration, Session: SessionSpillover,
		Compatibility: compatibility.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if selected.Peer.ID != compatibility.ID || selected.Peer.PeerObservation != peerObservation.ID ||
		selected.Resources.Identity != resources.Identity {
		t.Fatalf("spillover selection=%+v", selected)
	}
}
