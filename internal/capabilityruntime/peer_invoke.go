package capabilityruntime

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// PeerInvocationNode names the receipt chain one peer capability
// invocation records under.
const PeerInvocationNode = recipe.NodeID("peer.invoke")

// peerInvocationAttempt is the single attempt ordinal this door records:
// a retry is a new invocation under a new operation, never a second
// attempt of the same receipt chain.
const peerInvocationAttempt = 1

// PeerCapabilityManual derives the exact UTCP manual one remote recipe
// capability is invoked through: the selection's placement facts admit the
// derivation, and the manual construction itself stays with its agenttool
// owner.
func PeerCapabilityManual(selection modelrecipe.CapabilityEvidenceSelection, task recipe.Task) (agenttool.Manual, error) {
	var none agenttool.Manual
	if err := validateRemotePeerSelection(selection); err != nil {
		return none, err
	}
	if !slices.Contains(selection.Peer.Capability.Tasks, task) {
		return none, fmt.Errorf("capability runtime: peer capability does not publish task %q", task)
	}
	return agenttool.PeerCapabilityManual(
		selection.Peer.Capability.Endpoint, selection.Peer.Capability.ID, string(task),
	)
}

// RequireExactPeerManual refuses invoking a remote recipe capability through
// anything but the exact manual its selection derives: same identity, same
// endpoint, same target, same protocol -- a stale or hand-edited manual
// cannot reach the peer.
func RequireExactPeerManual(manual agenttool.Manual, selection modelrecipe.CapabilityEvidenceSelection, task recipe.Task) error {
	expected, err := PeerCapabilityManual(selection, task)
	if err != nil {
		return err
	}
	if manual.ID != expected.ID {
		return errors.New("capability runtime: remote recipe capability requires its exact UTCP manual")
	}
	return nil
}

// InvokeRemotePeer executes one manual-gated, receipted peer capability
// invocation: the admitted receipt lands before any bytes leave, the
// transport streams through the compatibility-admitted executor, and the
// terminal receipt records completion or the exact failure with the
// capability and compatibility evidence bound as outputs.
func InvokeRemotePeer(
	ctx context.Context,
	client *http.Client,
	repository artifact.Repository,
	manual agenttool.Manual,
	selection modelrecipe.CapabilityEvidenceSelection,
	task recipe.Task,
	operation artifact.ID,
	input io.Reader,
	output io.Writer,
) (runrecord.StageReceipt, error) {
	if repository == nil || !operation.Valid() {
		return runrecord.StageReceipt{}, errors.New("capability runtime: peer invocation requires a repository and operation")
	}
	if err := RequireExactPeerManual(manual, selection, task); err != nil {
		return runrecord.StageReceipt{}, err
	}
	admitted, err := runrecord.PublishStageReceipt(ctx, repository, runrecord.StageReceipt{
		Recipe: selection.Peer.Recipe, Node: PeerInvocationNode, Operation: operation,
		Attempt: peerInvocationAttempt, State: runrecord.StageAdmitted,
		Inputs: []runrecord.StageBinding{{Port: "capability", Artifacts: []artifact.ID{
			selection.Peer.Capability.ID, selection.Peer.ID, manual.ID,
		}}},
	}, nil, []artifact.Descriptor{
		{ID: selection.Peer.Capability.ID}, {ID: selection.Peer.ID}, {ID: manual.ID},
		{ID: selection.Peer.Recipe}, {ID: operation},
	})
	if err != nil {
		return runrecord.StageReceipt{}, err
	}
	running, err := runrecord.PublishStageReceipt(ctx, repository, runrecord.StageReceipt{
		Recipe: selection.Peer.Recipe, Node: PeerInvocationNode, Operation: operation,
		Attempt: peerInvocationAttempt, State: runrecord.StageRunning, Previous: admitted.ID,
	}, nil, nil)
	if err != nil {
		return runrecord.StageReceipt{}, err
	}
	terminal := runrecord.StageReceipt{
		Recipe: selection.Peer.Recipe, Node: PeerInvocationNode, Operation: operation,
		Attempt: peerInvocationAttempt, Previous: running.ID,
		Outputs: []runrecord.StageBinding{{Port: "capability", Artifacts: []artifact.ID{
			selection.Peer.Capability.ID,
		}}},
	}
	if invokeErr := ExecuteRemotePeer(ctx, client, selection, input, output); invokeErr != nil {
		terminal.State, terminal.Failure = runrecord.StageFailed, invokeErr.Error()
		receipt, publishErr := runrecord.PublishStageReceipt(ctx, repository, terminal, nil, nil)
		return receipt, errors.Join(invokeErr, publishErr)
	}
	terminal.State = runrecord.StageCompleted
	return runrecord.PublishStageReceipt(ctx, repository, terminal, nil, nil)
}

// validateRemotePeerSelection is the shared admission preamble: spillover
// session, consistent capability and compatibility identities, and the
// activation the peer was admitted against.
func validateRemotePeerSelection(selection modelrecipe.CapabilityEvidenceSelection) error {
	if selection.Session != modelrecipe.SessionSpillover || !selection.Peer.ID.Valid() ||
		selection.Peer.Capability.ID != selection.Peer.PeerCapability ||
		selection.Peer.Model != selection.Activation.Definition.Model ||
		selection.Peer.Recipe != selection.Activation.Definition.ID ||
		selection.Peer.Resources != selection.Resources.Identity ||
		selection.Peer.Capability.Endpoint == "" {
		return errors.New("capability runtime: incomplete remote peer selection")
	}
	return nil
}
