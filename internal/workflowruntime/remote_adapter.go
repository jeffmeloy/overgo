package workflowruntime

import (
	"context"
	"errors"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const remoteExecutionFailure = "remote_execution_failed"

// RemoteStageRequest binds transport input to exact recipe authority.
type RemoteStageRequest struct {
	Endpoint      string
	Compatibility artifact.ID
	Recipe        artifact.ID
	Operation     artifact.ID
	Model         artifact.ID
	Task          recipe.Task
	Node          recipe.NodeID
	Inputs        map[recipe.PortName]Value
}

// RemoteStageResponse contains artifact-bound stage outputs.
type RemoteStageResponse struct {
	Outputs map[recipe.PortName]Value
}

// RemoteTransport executes one exact admitted stage request.
type RemoteTransport interface {
	ExecuteRemote(context.Context, RemoteStageRequest) (RemoteStageResponse, error)
}

// RemoteAdapter executes through one evidence-bound spillover selection.
type RemoteAdapter struct {
	Store     artifact.Repository
	Selection modelrecipe.CapabilityEvidenceSelection
	Transport RemoteTransport
}

// Execute refreshes authority, calls the peer, and publishes attempt evidence.
func (adapter RemoteAdapter) Execute(ctx context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
	if ctx == nil || adapter.Store == nil || adapter.Transport == nil ||
		adapter.Selection.Session != modelrecipe.SessionSpillover || !adapter.Selection.Peer.ID.Valid() ||
		adapter.Selection.Peer.Capability.ID != adapter.Selection.Peer.PeerCapability {
		return nil, errors.New("workflow runtime: remote execution authority is absent")
	}
	definition := adapter.Selection.Program.Definition()
	if request.Recipe != definition.ID || request.Model != definition.Model || request.Task != definition.Task ||
		request.Operation.Kind() != artifact.KindEvidence {
		return nil, errors.New("workflow runtime: remote stage authority differs")
	}
	current, err := modelrecipe.RefreshCapabilityExecution(ctx, adapter.Store, adapter.Selection)
	if err != nil || current.Identity != adapter.Selection.Identity {
		return nil, errors.Join(errors.New("workflow runtime: remote capability changed"), err)
	}
	started := time.Now()
	clock := processmeasure.NewStopwatch()
	response, executeErr := adapter.Transport.ExecuteRemote(ctx, RemoteStageRequest{
		Endpoint: current.Peer.Capability.Endpoint, Compatibility: current.Peer.ID,
		Recipe: request.Recipe, Operation: request.Operation, Model: request.Model,
		Task: request.Task, Node: request.Node, Inputs: cloneValues(request.Inputs),
	})
	if executeErr == nil {
		_, _, executeErr = externalFacts(response.Outputs, true)
	}
	outcome, failure := runrecord.OutcomeSucceeded, ""
	if executeErr != nil {
		outcome, failure = runrecord.OutcomeFailed, remoteExecutionFailure
		if errors.Is(executeErr, context.Canceled) || errors.Is(executeErr, context.DeadlineExceeded) {
			outcome, failure = runrecord.OutcomeCancelled, ""
		}
	}
	wall, wallErr := clock.Elapsed()
	if wallErr != nil {
		return nil, errors.Join(executeErr, wallErr)
	}
	observation, publishErr := runrecord.PublishServingObservation(context.WithoutCancel(ctx), adapter.Store, runrecord.ServingObservation{
		Model: request.Model, Recipe: request.Recipe, Environment: current.Peer.PeerEnvironment,
		Operation: request.Operation, Compatibility: current.Peer.ID,
		Task: request.Task, Outcome: outcome, Failure: failure,
		StartedUnixNS: started.UnixNano(), MeasuredNS: wall,
	})
	if publishErr == nil && request.Attempts != nil {
		request.Attempts.Attempt(observation.ID)
	}
	if publishErr != nil {
		return nil, errors.Join(executeErr, publishErr)
	}
	return response.Outputs, executeErr
}

func cloneValues(values map[recipe.PortName]Value) map[recipe.PortName]Value {
	result := make(map[recipe.PortName]Value, len(values))
	for name, value := range values {
		result[name] = cloneValue(value)
	}
	return result
}
