package server

import (
	"context"
	"errors"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// peerCapacityProjection reports declared work and current lease availability.
type peerCapacityProjection struct {
	Environment  artifact.ID   `json:"environment"`
	Tasks        []recipe.Task `json:"tasks,omitempty"`
	LeaseExpires int64         `json:"lease_expires_unix_ns,omitzero"`
	Available    bool          `json:"available"`
}

// peerInventoryEntry joins one immutable enrollment to its current aliases.
type peerInventoryEntry struct {
	Peer          artifact.ID                            `json:"peer"`
	Enrollment    runrecord.PeerEnrollment               `json:"enrollment"`
	StateID       artifact.ID                            `json:"state_id"`
	State         runrecord.PeerState                    `json:"state"`
	PublicationID artifact.ID                            `json:"publication_id,omitzero"`
	Publication   *modelrecipe.PeerCapabilityPublication `json:"publication,omitempty"`
	Capability    *modelrecipe.RemotePeerCapability      `json:"capability,omitempty"`
	HeartbeatID   artifact.ID                            `json:"heartbeat_id,omitzero"`
	Heartbeat     *runrecord.PeerHeartbeat               `json:"heartbeat,omitempty"`
	Capacity      peerCapacityProjection                 `json:"capacity"`
	Refusal       string                                 `json:"refusal,omitzero"`
}

// peerInventoryProjection is an explicitly bounded peer control-plane view.
type peerInventoryProjection struct {
	Peers     []peerInventoryEntry `json:"peers"`
	Truncated bool                 `json:"truncated"`
}

// peerEvidenceLog is a payload-free rendering of immutable attempt evidence.
type peerEvidenceLog struct {
	Attempt artifact.ID                  `json:"attempt"`
	Phase   runrecord.PeerReplicaPhase   `json:"phase"`
	Outcome runrecord.PeerReplicaOutcome `json:"outcome"`
	Failure string                       `json:"failure,omitzero"`
}

// peerEvidenceProjection joins operation, attempt, artifact, and log facts.
type peerEvidenceProjection struct {
	Operation *operation.Status                                         `json:"operation,omitempty"`
	Attempts  []operationEvidenceDocument[runrecord.PeerReplicaReceipt] `json:"attempts"`
	Artifacts []artifact.ID                                             `json:"artifacts"`
	Logs      []peerEvidenceLog                                         `json:"logs"`
	Truncated bool                                                      `json:"truncated"`
}

// peerWorkspaceAPI is the typed protocol boundary shared by API and GUI.
type peerWorkspaceAPI interface {
	PeerInventory(context.Context, int) (peerInventoryProjection, error)
	EnrollPeer(context.Context, runrecord.PeerEnrollment) (runrecord.PeerEnrollment, runrecord.PeerState, error)
	PublishPeerCapability(context.Context, artifact.ID, modelrecipe.RemotePeerCapability, int64) (modelrecipe.PeerCapabilityPublication, modelrecipe.RemotePeerCapability, error)
	HeartbeatPeer(context.Context, runrecord.PeerHeartbeat) (runrecord.PeerHeartbeat, error)
	TransitionPeer(context.Context, artifact.ID, runrecord.PeerAdministrativeState, int64) (runrecord.PeerState, error)
	CompilePeerPlacement(context.Context, modelrecipe.PeerPlacementRequest) (modelrecipe.PeerPlacementPlan, error)
	ReconcilePeer(context.Context, *operation.Manager, capabilityruntime.PeerReconcileRequest) (artifact.ID, error)
	PeerEvidence(context.Context, *operation.Manager, artifact.ID, int) (peerEvidenceProjection, error)
}

// peerWorkspace composes peer authorities without owning a second registry or queue.
type peerWorkspace struct {
	store   *overgodb.Store
	backend capabilityruntime.PeerReplicaBackend
	limit   int
	clock   func() time.Time
}

// peerWorkspaceConfig binds peer controls to OvergoDB and the peer transport seam.
type peerWorkspaceConfig struct {
	Store   *overgodb.Store
	Backend capabilityruntime.PeerReplicaBackend
	Limit   int
}

// Open validates and constructs the peer operator boundary.
func (config peerWorkspaceConfig) Open() (*peerWorkspace, error) {
	if config.Store == nil || config.Backend == nil || config.Limit <= 0 {
		return nil, errors.New("peer workspace: repository, backend, and positive projection limit required")
	}
	return &peerWorkspace{store: config.Store, backend: config.Backend, limit: config.Limit, clock: time.Now}, nil
}

// PeerInventory returns bounded enrollment and current-state projections.
func (workspace *peerWorkspace) PeerInventory(ctx context.Context, limit int) (peerInventoryProjection, error) {
	if workspace == nil || ctx == nil || limit <= 0 || limit > workspace.limit {
		return peerInventoryProjection{}, errors.New("peer workspace: invalid inventory bound")
	}
	projection := peerInventoryProjection{Peers: make([]peerInventoryEntry, 0, limit)}
	page, err := overgodb.VisitDecodedDocuments(ctx, workspace.store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.PeerEnrollmentMediaType, Schema: runrecord.PeerEnrollmentSchema,
		}},
		Order: overgodb.DocumentNewestFirst, MaxResults: limit,
	}, runrecord.ParsePeerEnrollment, func(_ overgodb.DocumentView, enrollment runrecord.PeerEnrollment) error {
		entry, entryErr := workspace.peerInventoryEntry(ctx, enrollment)
		if entryErr != nil {
			return entryErr
		}
		projection.Peers = append(projection.Peers, entry)
		return nil
	})
	projection.Truncated = page.Truncated
	return projection, err
}

func (workspace *peerWorkspace) peerInventoryEntry(ctx context.Context, enrollment runrecord.PeerEnrollment) (peerInventoryEntry, error) {
	entry := peerInventoryEntry{Peer: enrollment.ID, Enrollment: enrollment, Capacity: peerCapacityProjection{Environment: enrollment.Environment}}
	state, found, err := runrecord.ResolvePeerState(ctx, workspace.store, enrollment.ID)
	if err != nil {
		return peerInventoryEntry{}, err
	}
	if !found {
		entry.Refusal = "peer administrative state is unavailable"
		return entry, nil
	}
	entry.State = state
	entry.StateID = state.ID
	publication, capability, found, err := modelrecipe.ResolvePeerCapabilityPublication(ctx, workspace.store, enrollment.ID)
	if err != nil {
		return peerInventoryEntry{}, err
	}
	if !found {
		entry.Refusal = "peer capability is unavailable"
		return entry, nil
	}
	entry.Publication, entry.Capability = &publication, &capability
	entry.PublicationID = publication.ID
	entry.Capacity.Tasks = slices.Clone(capability.Tasks)
	heartbeat, found, err := runrecord.ResolvePeerHeartbeat(ctx, workspace.store, enrollment.ID)
	if err != nil {
		return peerInventoryEntry{}, err
	}
	if !found {
		entry.Refusal = "peer heartbeat is unavailable"
		return entry, nil
	}
	entry.Heartbeat = &heartbeat
	entry.HeartbeatID = heartbeat.ID
	entry.Capacity.LeaseExpires = heartbeat.ExpiresUnixNS
	now := workspace.clock().UTC().UnixNano()
	entry.Capacity.Available = state.State == runrecord.PeerActive && heartbeat.Capability == publication.ID &&
		heartbeat.ObservedUnixNS <= now && now < heartbeat.ExpiresUnixNS
	if !entry.Capacity.Available {
		entry.Refusal = "peer state or capability-bound lease is unavailable"
	}
	return entry, nil
}

// EnrollPeer commits approval through the common lifecycle owner.
func (workspace *peerWorkspace) EnrollPeer(ctx context.Context, value runrecord.PeerEnrollment) (runrecord.PeerEnrollment, runrecord.PeerState, error) {
	if value.ApprovedUnixNS == 0 {
		value.ApprovedUnixNS = workspace.clock().UTC().UnixNano()
	}
	return (capabilityruntime.PeerLifecycleAuthority{Repository: workspace.store}).Enroll(ctx, value)
}

// PublishPeerCapability advances the enrolled peer's declared transport.
func (workspace *peerWorkspace) PublishPeerCapability(ctx context.Context, peer artifact.ID, value modelrecipe.RemotePeerCapability, publishedUnixNS int64) (modelrecipe.PeerCapabilityPublication, modelrecipe.RemotePeerCapability, error) {
	return (capabilityruntime.PeerLifecycleAuthority{Repository: workspace.store}).PublishCapability(ctx, peer, value, publishedUnixNS)
}

// HeartbeatPeer advances a capability-bound peer lease.
func (workspace *peerWorkspace) HeartbeatPeer(ctx context.Context, value runrecord.PeerHeartbeat) (runrecord.PeerHeartbeat, error) {
	return (capabilityruntime.PeerLifecycleAuthority{Repository: workspace.store}).Heartbeat(ctx, value)
}

// TransitionPeer advances active, draining, and retired authority.
func (workspace *peerWorkspace) TransitionPeer(ctx context.Context, peer artifact.ID, state runrecord.PeerAdministrativeState, changedUnixNS int64) (runrecord.PeerState, error) {
	if changedUnixNS == 0 {
		changedUnixNS = workspace.clock().UTC().UnixNano()
	}
	return (capabilityruntime.PeerLifecycleAuthority{Repository: workspace.store}).Transition(ctx, peer, state, changedUnixNS)
}

// CompilePeerPlacement resolves current placement and refusal truth.
func (workspace *peerWorkspace) CompilePeerPlacement(ctx context.Context, request modelrecipe.PeerPlacementRequest) (modelrecipe.PeerPlacementPlan, error) {
	if request.NowUnixNS == 0 {
		request.NowUnixNS = workspace.clock().UTC().UnixNano()
	}
	return (capabilityruntime.PeerPlacementAuthority{Repository: workspace.store}).Compile(ctx, request)
}

// ReconcilePeer admits reconciliation through the common operation owner.
func (workspace *peerWorkspace) ReconcilePeer(ctx context.Context, manager *operation.Manager, request capabilityruntime.PeerReconcileRequest) (artifact.ID, error) {
	if request.ChangedUnixNS == 0 {
		request.ChangedUnixNS = workspace.clock().UTC().UnixNano()
	}
	reconciler, err := (capabilityruntime.PeerReplicaReconcilerConfig{
		Repository: workspace.store, Operations: manager, Backend: workspace.backend,
	}).Open()
	if err != nil {
		return artifact.ID{}, err
	}
	return reconciler.Reconcile(ctx, request)
}

// PeerEvidence returns bounded reconciliation facts without backend payloads.
func (workspace *peerWorkspace) PeerEvidence(ctx context.Context, manager *operation.Manager, id artifact.ID, limit int) (peerEvidenceProjection, error) {
	if workspace == nil || ctx == nil || manager == nil || id.Kind() != artifact.KindEvidence || limit <= 0 || limit > workspace.limit {
		return peerEvidenceProjection{}, errors.New("peer workspace: invalid evidence request")
	}
	projection := peerEvidenceProjection{Attempts: make([]operationEvidenceDocument[runrecord.PeerReplicaReceipt], 0, limit)}
	page, err := overgodb.VisitDecodedDocuments(ctx, workspace.store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.PeerReplicaReceiptMediaType, Schema: runrecord.PeerReplicaReceiptSchema,
		}},
		AliasPrefixes: []string{runrecord.PeerReplicaReceiptAliasRoot + id.String() + "/"},
		Order:         overgodb.DocumentNewestFirst, MaxResults: limit,
	}, runrecord.ParsePeerReplicaReceipt, func(view overgodb.DocumentView, receipt runrecord.PeerReplicaReceipt) error {
		projection.Attempts = append(projection.Attempts, operationEvidenceDocument[runrecord.PeerReplicaReceipt]{ID: view.Content.Descriptor.ID, Value: receipt})
		projection.Logs = append(projection.Logs, peerEvidenceLog{Attempt: receipt.ID, Phase: receipt.Phase, Outcome: receipt.Outcome, Failure: receipt.Failure})
		for _, artifactID := range receipt.Artifacts {
			if !slices.Contains(projection.Artifacts, artifactID) {
				projection.Artifacts = append(projection.Artifacts, artifactID)
			}
		}
		return nil
	})
	if err != nil {
		return peerEvidenceProjection{}, err
	}
	if status, found := manager.Status(id); found {
		projection.Operation = &status
	}
	slices.SortFunc(projection.Artifacts, artifact.CompareID)
	projection.Truncated = page.Truncated
	return projection, nil
}
