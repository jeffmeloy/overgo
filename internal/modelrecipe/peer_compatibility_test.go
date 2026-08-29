package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestReferenceAdmissionRequiresSemanticRelevance(t *testing.T) {
	ctx := t.Context()
	fixture := newCapabilitySelectorFixture(t)
	id := func(kind artifact.Kind, label string) artifact.ID {
		return testutil.ArtifactID(t, kind, label)
	}
	localEnvironment := id(artifact.KindEvidence, "reference-local-environment")
	peerEnvironment := id(artifact.KindEvidence, "reference-peer-environment")
	resources := id(artifact.KindProfile, "reference-resources")
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key: "reference/authorities", Artifacts: []artifact.Descriptor{
			{ID: localEnvironment}, {ID: peerEnvironment}, {ID: resources},
		},
	}); err != nil {
		t.Fatal(err)
	}
	observe := func(model, recipeID artifact.ID, task recipe.Task, environment, operation artifact.ID) runrecord.ServingObservation {
		t.Helper()
		observation, err := runrecord.PublishServingObservation(ctx, fixture.store, runrecord.ServingObservation{
			Model: model, Recipe: recipeID, Environment: environment, Operation: operation,
			Task: task, Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 1, MeasuredNS: 1,
		})
		if err != nil {
			t.Fatal(err)
		}
		return observation
	}
	capability := RemotePeerCapability{
		Environment: peerEnvironment, Tasks: []recipe.Task{recipe.TaskGeneration},
		Endpoint: "https://peer.example/v1/reference",
	}
	compatibility := func(recipeID artifact.ID, local, peer runrecord.ServingObservation) RemotePeerCompatibility {
		return RemotePeerCompatibility{
			Model: fixture.model, Recipe: recipeID, Resources: resources, Task: recipe.TaskGeneration,
			LocalEnvironment: localEnvironment, PeerEnvironment: peerEnvironment,
			LocalObservation: local.ID, PeerObservation: peer.ID,
		}
	}

	localOperation := id(artifact.KindEvidence, "reference-local-operation")
	peerOperation := id(artifact.KindEvidence, "reference-peer-operation")
	local := observe(fixture.model, fixture.definition.ID, recipe.TaskGeneration, localEnvironment, localOperation)
	peer := observe(fixture.model, fixture.definition.ID, recipe.TaskGeneration, peerEnvironment, peerOperation)
	for _, operation := range []artifact.ID{localOperation, peerOperation} {
		if _, found, err := fixture.store.Artifact(ctx, operation); err != nil || found {
			t.Fatalf("opaque operation unexpectedly resolved = (%t, %v)", found, err)
		}
	}
	if _, err := PublishRemotePeerAuthority(
		ctx, fixture.store, "reference/exact", capability, compatibility(fixture.definition.ID, local, peer),
	); err != nil {
		t.Fatalf("exact peer references refused: %v", err)
	}

	foreignModel := id(artifact.KindModel, "reference-foreign-model")
	foreignDefinition, err := CapabilityDefinition(recipe.TaskGeneration, foreignModel)
	if err != nil {
		t.Fatal(err)
	}
	foreignDefinitionContent, err := foreignDefinition.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key: "reference/foreign-definition", Artifacts: []artifact.Descriptor{{ID: foreignModel}},
		Contents: []artifact.Content{foreignDefinitionContent},
	}); err != nil {
		t.Fatal(err)
	}
	foreignLocal := observe(
		fixture.model, foreignDefinition.ID, recipe.TaskGeneration, localEnvironment,
		id(artifact.KindEvidence, "reference-foreign-local-operation"),
	)
	foreignPeer := observe(
		fixture.model, foreignDefinition.ID, recipe.TaskGeneration, peerEnvironment,
		id(artifact.KindEvidence, "reference-foreign-peer-operation"),
	)
	if _, err := PublishRemotePeerAuthority(
		ctx, fixture.store, "reference/foreign", capability, compatibility(foreignDefinition.ID, foreignLocal, foreignPeer),
	); err == nil {
		t.Fatal("well-formed foreign recipe definition was admitted for another model")
	}
	foreignTaskDefinition, err := CapabilityDefinition(recipe.TaskForecast, fixture.model)
	if err != nil {
		t.Fatal(err)
	}
	foreignTaskContent, err := foreignTaskDefinition.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key: "reference/foreign-task-definition", Contents: []artifact.Content{foreignTaskContent},
	}); err != nil {
		t.Fatal(err)
	}
	foreignTaskLocal := observe(
		fixture.model, foreignTaskDefinition.ID, recipe.TaskGeneration, localEnvironment,
		id(artifact.KindEvidence, "reference-foreign-task-local-operation"),
	)
	foreignTaskPeer := observe(
		fixture.model, foreignTaskDefinition.ID, recipe.TaskGeneration, peerEnvironment,
		id(artifact.KindEvidence, "reference-foreign-task-peer-operation"),
	)
	if _, err := PublishRemotePeerAuthority(
		ctx, fixture.store, "reference/foreign-task", capability,
		compatibility(foreignTaskDefinition.ID, foreignTaskLocal, foreignTaskPeer),
	); err == nil {
		t.Fatal("well-formed recipe definition was admitted for another task")
	}

	detachedPeer := commitUnboundPeerTestObservation(t, ctx, fixture.store, runrecord.ServingObservation{
		Model: fixture.model, Recipe: fixture.definition.ID, Environment: peerEnvironment,
		Operation: id(artifact.KindEvidence, "reference-detached-peer-operation"),
		Task:      recipe.TaskGeneration, Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 1, MeasuredNS: 1,
	})
	if _, err := PublishRemotePeerAuthority(
		ctx, fixture.store, "reference/operation-mismatch", capability, compatibility(fixture.definition.ID, local, detachedPeer),
	); err == nil {
		t.Fatal("peer observation without its exact operation alias was admitted")
	}

	zeroLocal := observe(fixture.model, fixture.definition.ID, recipe.TaskGeneration, localEnvironment, artifact.ID{})
	zeroPeer := observe(fixture.model, fixture.definition.ID, recipe.TaskGeneration, peerEnvironment, artifact.ID{})
	if _, err := PublishRemotePeerAuthority(
		ctx, fixture.store, "reference/operation-absent", capability, compatibility(fixture.definition.ID, zeroLocal, zeroPeer),
	); err == nil {
		t.Fatal("peer observations without an operation were admitted")
	}
}

func commitUnboundPeerTestObservation(
	t *testing.T,
	ctx context.Context,
	repository artifact.Repository,
	value runrecord.ServingObservation,
) runrecord.ServingObservation {
	t.Helper()
	observation, err := runrecord.NewServingObservation(value)
	if err != nil {
		t.Fatal(err)
	}
	content, err := observation.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"test/peer-observation/"+observation.ID.String(),
		[]artifact.Content{content}, observation.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		t.Fatal(err)
	}
	return observation
}
