package workflowruntime

import (
	"context"
	"errors"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const remoteVerificationCodeCommit = "0123456789abcdef0123456789abcdef01234567"

type remoteTransportFunc func(context.Context, RemoteStageRequest) (RemoteStageResponse, error)

func (execute remoteTransportFunc) ExecuteRemote(ctx context.Context, request RemoteStageRequest) (RemoteStageResponse, error) {
	return execute(ctx, request)
}

type remoteAdapterFixture struct {
	store     *repodb.Store
	selection modelrecipe.CapabilityEvidenceSelection
	output    artifact.Content
}

func TestRemoteStageAttemptLineage(t *testing.T) {
	fixture := newRemoteAdapterFixture(t)
	adapter := RemoteAdapter{Store: fixture.store, Selection: fixture.selection, Transport: remoteTransportFunc(
		func(_ context.Context, request RemoteStageRequest) (RemoteStageResponse, error) {
			if request.Endpoint != fixture.selection.Peer.Capability.Endpoint ||
				request.Compatibility != fixture.selection.Peer.ID {
				t.Fatal("remote transport authority differs")
			}
			return RemoteStageResponse{Outputs: remoteOutputs(fixture.output)}, nil
		},
	)}
	manager, err := operation.NewManager(testToolRetention)
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	runID := testutil.ArtifactID(t, artifact.KindRun, "remote-stage-run")
	operationID, err := manager.Submit(t.Context(), operation.Request{
		Task: fixture.selection.Program.Definition().Task, Recipe: fixture.selection.Program.Definition().ID,
	}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		outputs, executeErr := adapter.Execute(ctx, fixture.request(reporter.OperationID(), reporter))
		ids, _, factErr := externalFacts(outputs, true)
		return operation.Completion{Run: runID, Outputs: ids}, errors.Join(executeErr, factErr)
	})
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(t.Context(), operationID)
	if err != nil || status.State != operation.StateCompleted || len(status.Attempts) != 1 {
		t.Fatalf("remote operation = %+v, %v", status, err)
	}
	observation, err := runrecord.RequireServingObservation(t.Context(), fixture.store, status.Attempts[0])
	if err != nil || observation.Operation != operationID || observation.Compatibility != fixture.selection.Peer.ID ||
		observation.AttemptKind(nil) != runrecord.ServingAttemptSpillover {
		t.Fatalf("remote attempt = %+v, %v", observation, err)
	}
	parents, err := fixture.store.Parents(t.Context(), observation.ID)
	if err != nil || !slices.ContainsFunc(parents, func(edge artifact.Lineage) bool {
		return edge.Parent == fixture.selection.Peer.ID
	}) {
		t.Fatalf("remote attempt parents = %v, %v", parents, err)
	}
}

func TestRemoteCancellation(t *testing.T) {
	fixture := newRemoteAdapterFixture(t)
	started := make(chan struct{})
	adapter := RemoteAdapter{Store: fixture.store, Selection: fixture.selection, Transport: remoteTransportFunc(
		func(ctx context.Context, _ RemoteStageRequest) (RemoteStageResponse, error) {
			close(started)
			<-ctx.Done()
			return RemoteStageResponse{}, ctx.Err()
		},
	)}
	manager, err := operation.NewManager(testToolRetention)
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	runID := testutil.ArtifactID(t, artifact.KindRun, "cancelled-remote-stage-run")
	ctx, cancel := context.WithCancel(context.Background())
	operationID, err := manager.Submit(ctx, operation.Request{
		Task: fixture.selection.Program.Definition().Task, Recipe: fixture.selection.Program.Definition().ID,
	}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		_, executeErr := adapter.Execute(ctx, fixture.request(reporter.OperationID(), reporter))
		return operation.Completion{Run: runID}, executeErr
	})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	cancel()
	status, err := manager.Wait(context.Background(), operationID)
	if err != nil || status.State != operation.StateCancelled || len(status.Attempts) != 1 {
		t.Fatalf("cancelled remote operation = %+v, %v", status, err)
	}
	observation, err := runrecord.RequireServingObservation(t.Context(), fixture.store, status.Attempts[0])
	if err != nil || observation.Outcome != runrecord.OutcomeCancelled {
		t.Fatalf("cancelled remote attempt = %+v, %v", observation, err)
	}
}

func TestRemoteArtifactIdentity(t *testing.T) {
	fixture := newRemoteAdapterFixture(t)
	wrong := fixture.output.Descriptor
	wrong.ID = testutil.ArtifactID(t, artifact.KindOutput, "wrong-remote-output")
	adapter := RemoteAdapter{Store: fixture.store, Selection: fixture.selection, Transport: remoteTransportFunc(
		func(context.Context, RemoteStageRequest) (RemoteStageResponse, error) {
			return RemoteStageResponse{Outputs: map[recipe.PortName]Value{
				"output": {Kind: recipe.DataArtifact, Items: []Datum{{Artifact: wrong, Content: &fixture.output}}},
			}}, nil
		},
	)}
	operationID := testutil.ArtifactID(t, artifact.KindEvidence, "remote-artifact-operation")
	if _, err := adapter.Execute(t.Context(), fixture.request(operationID, nil)); err == nil {
		t.Fatal("remote output identity drift accepted")
	}
}

func newRemoteAdapterFixture(t *testing.T) remoteAdapterFixture {
	t.Helper()
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	weights := []byte("remote-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	model, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "remote/model", Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Manifests: []artifact.Manifest{model},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskGeneration, model.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "remote/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification, err := publishRemoteVerification(ctx, store, definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceParity, "remote fixture"); err != nil {
		t.Fatal(err)
	}
	localSelection, err := modelrecipe.ResolveActiveExecution(ctx, store, model.ID, definition.Task, modelrecipe.SessionWarm)
	if err != nil {
		t.Fatal(err)
	}
	resourceContent, err := localSelection.Resources.Content()
	if err != nil {
		t.Fatal(err)
	}
	local := testutil.ArtifactID(t, artifact.KindEvidence, "remote-local-environment")
	peer := testutil.ArtifactID(t, artifact.KindEvidence, "remote-peer-environment")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "remote/authority", Artifacts: []artifact.Descriptor{{ID: local}, {ID: peer}},
		Contents: []artifact.Content{resourceContent},
	}); err != nil {
		t.Fatal(err)
	}
	observe := func(environment artifact.ID) runrecord.ServingObservation {
		observation, publishErr := runrecord.PublishServingObservation(ctx, store, runrecord.ServingObservation{
			Model: model.ID, Recipe: definition.ID, Environment: environment,
			Task: definition.Task, Outcome: runrecord.OutcomeSucceeded,
			StartedUnixNS: 1, MeasuredNS: 1,
		})
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		return observation
	}
	localObservation, peerObservation := observe(local), observe(peer)
	compatibility, err := modelrecipe.PublishRemotePeerAuthority(ctx, store, "remote/peer", modelrecipe.RemotePeerCapability{
		Environment: peer, Tasks: []recipe.Task{definition.Task}, Endpoint: "https://peer.example/v1/workflow",
	}, modelrecipe.RemotePeerCompatibility{
		Model: model.ID, Recipe: definition.ID, Resources: localSelection.Resources.Identity, Task: definition.Task,
		LocalEnvironment: local, PeerEnvironment: peer,
		LocalObservation: localObservation.ID, PeerObservation: peerObservation.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	const alias = "capability/remote/default"
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "remote/alias", Aliases: []artifact.AliasBinding{{Name: alias, Target: model.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	selection, err := modelrecipe.ResolveCapabilityEvidenceSelector(ctx, store, modelrecipe.CapabilityEvidenceSelector{
		Alias: alias, Task: definition.Task, Session: modelrecipe.SessionSpillover, Compatibility: compatibility.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	output, err := (artifact.DocumentContract{
		Kind: artifact.KindOutput, MediaType: "application/octet-stream", Schema: "overgo/remote-output/v1",
	}).ContentBytes([]byte("remote-output"))
	if err != nil {
		t.Fatal(err)
	}
	return remoteAdapterFixture{store: store, selection: selection, output: output}
}

func publishRemoteVerification(
	ctx context.Context,
	store artifact.Repository,
	recipeID artifact.ID,
) (modelrecipe.Verification, error) {
	environment, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("remote-verification-environment"))
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "remote/verification/environment", Artifacts: []artifact.Descriptor{{ID: environment}},
	}); err != nil {
		return modelrecipe.Verification{}, err
	}
	record, err := runrecord.NewGateRecord(
		recipeID, environment, remoteVerificationCodeCommit, runrecord.OutcomeSucceeded, "", 1,
		[]runrecord.GateStep{{Name: "verify", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepSucceeded, DurationNS: 1}},
	)
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	batch, err := record.Batch("remote/verification/gate")
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return modelrecipe.Verification{}, err
	}
	return modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}, nil
}

func (fixture remoteAdapterFixture) request(operationID artifact.ID, attempts AttemptRecorder) StepRequest {
	definition := fixture.selection.Program.Definition()
	return StepRequest{
		Recipe: definition.ID, Operation: operationID, Model: definition.Model,
		Task: definition.Task, Node: definition.Nodes[0].ID, Attempts: attempts,
	}
}

func remoteOutputs(content artifact.Content) map[recipe.PortName]Value {
	return map[recipe.PortName]Value{
		"output": ArtifactValue(recipe.DataArtifact, slices.Clone(content.Data), content),
	}
}
