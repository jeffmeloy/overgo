package capabilityruntime

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/runrecord"
)

// PeerReconcilePolicy bounds backend action attempts without embedding retry policy.
type PeerReconcilePolicy struct {
	MaximumAttempts uint32 `json:"maximum_attempts"`
}

// PeerReconcileRequest carries the newly compiled plan and prior resident replicas.
type PeerReconcileRequest struct {
	Plan          modelrecipe.PeerPlacementPlan      `json:"plan"`
	Previous      []modelrecipe.PeerReplicaPlacement `json:"previous,omitempty"`
	Policy        PeerReconcilePolicy                `json:"policy"`
	ChangedUnixNS int64                              `json:"changed_unix_ns"`
}

// PeerReplicaBackend is the idempotent staging and residency seam owned by a peer transport.
type PeerReplicaBackend interface {
	Stage(context.Context, modelrecipe.PeerReplicaPlacement) error
	Load(context.Context, modelrecipe.PeerReplicaPlacement) error
	ActiveLeases(context.Context, modelrecipe.PeerReplicaPlacement) (int, error)
	Unload(context.Context, modelrecipe.PeerReplicaPlacement) error
}

// PeerReplicaReconcilerConfig binds reconciliation to existing operation and RepoDB owners.
type PeerReplicaReconcilerConfig struct {
	Repository artifact.Repository
	Operations *operation.Manager
	Backend    PeerReplicaBackend
}

// PeerReplicaReconciler reconciles one immutable placement plan through existing operations.
type PeerReplicaReconciler struct {
	repository artifact.Repository
	operations *operation.Manager
	backend    PeerReplicaBackend
}

// Open validates and constructs the reconciliation boundary.
func (config PeerReplicaReconcilerConfig) Open() (*PeerReplicaReconciler, error) {
	if config.Repository == nil || config.Operations == nil || config.Backend == nil {
		return nil, errors.New("capability runtime: incomplete peer replica reconciler")
	}
	return &PeerReplicaReconciler{
		repository: config.Repository, operations: config.Operations, backend: config.Backend,
	}, nil
}

// Reconcile admits one placement reconciliation operation.
func (reconciler *PeerReplicaReconciler) Reconcile(
	ctx context.Context,
	request PeerReconcileRequest,
) (artifact.ID, error) {
	if err := validatePeerReconcileRequest(ctx, reconciler, request); err != nil {
		return artifact.ID{}, err
	}
	return reconciler.operations.Submit(ctx, operation.Request{
		Task: request.Plan.Task, Recipe: request.Plan.Recipe,
	}, reconciler.executor(request))
}

// Recover resumes the same durable action chains under the original operation identity.
func (reconciler *PeerReplicaReconciler) Recover(
	ctx context.Context,
	operationID artifact.ID,
	request PeerReconcileRequest,
) (artifact.ID, error) {
	if err := validatePeerReconcileRequest(ctx, reconciler, request); err != nil {
		return artifact.ID{}, err
	}
	return reconciler.operations.Recover(ctx, operationID, operation.Request{
		Task: request.Plan.Task, Recipe: request.Plan.Recipe,
	}, reconciler.executor(request))
}

func validatePeerReconcileRequest(
	ctx context.Context,
	reconciler *PeerReplicaReconciler,
	request PeerReconcileRequest,
) error {
	if ctx == nil || reconciler == nil || reconciler.repository == nil || reconciler.operations == nil || reconciler.backend == nil ||
		request.Policy.MaximumAttempts == 0 || request.ChangedUnixNS <= 0 || !request.Plan.Task.Valid() || len(request.Plan.Components) == 0 ||
		request.Plan.ValidateIdentity() != nil {
		return errors.New("capability runtime: invalid peer reconcile request")
	}
	seen := make(map[artifact.ID]struct{}, len(request.Previous))
	for _, replica := range request.Previous {
		target, err := peerReplicaTarget(replica)
		if err != nil {
			return err
		}
		if _, found := seen[target]; found {
			return errors.New("capability runtime: duplicate previous replica")
		}
		seen[target] = struct{}{}
	}
	return nil
}

func (reconciler *PeerReplicaReconciler) executor(request PeerReconcileRequest) operation.Executor {
	return func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		run, err := artifact.JSONID(artifact.KindRun, struct {
			Operation artifact.ID `json:"operation"`
			Plan      artifact.ID `json:"plan"`
		}{reporter.OperationID(), request.Plan.Identity})
		if err != nil {
			return operation.Completion{}, err
		}
		completion := operation.Completion{Run: run}
		if _, err := artifact.CommitBatch(ctx, reconciler.repository, artifact.Batch{
			Key: "peer/reconcile/run/" + run.String(), Artifacts: []artifact.Descriptor{{ID: run}},
		}); err != nil {
			return completion, err
		}
		var evidence []artifact.ID
		current := make(map[artifact.ID]modelrecipe.PeerReplicaPlacement, len(request.Plan.Replicas))
		for _, replica := range request.Plan.Replicas {
			target, err := peerReplicaTarget(replica)
			if err != nil {
				return completion, err
			}
			current[target] = replica
		}
		retired := make(map[artifact.ID]struct{})
		for _, replica := range request.Previous {
			target, err := peerReplicaTarget(replica)
			if err != nil {
				return completion, err
			}
			if _, retained := current[target]; retained {
				continue
			}
			leases, err := reconciler.backend.ActiveLeases(ctx, replica)
			if err != nil {
				return completion, err
			}
			if leases != 0 {
				receipt, publishErr := reconciler.publishAttempt(
					ctx, reporter, request.Plan.Identity, target, runrecord.PeerReplicaUnload,
					runrecord.PeerReplicaWaiting, "active_leases", peerReplicaArtifacts(replica),
				)
				if publishErr != nil {
					return completion, publishErr
				}
				return completion, operatoraction.Recoverable(errors.New("capability runtime: peer replica has active leases"), operatoraction.Block{
					Subject: request.Plan.Recipe, Reason: "wait for active peer leases to drain",
					Evidence: []artifact.ID{receipt.ID}, Actions: []operatoraction.Action{{
						Code: "retry-drain", Summary: "Retry peer drain", Argv: []string{"overgo", "peers", "reconcile", request.Plan.Identity.String()},
					}},
				})
			}
			receipt, err := reconciler.runAction(
				ctx, reporter, request, target, runrecord.PeerReplicaUnload,
				peerReplicaArtifacts(replica),
				func(ctx context.Context) error { return reconciler.backend.Unload(ctx, replica) },
			)
			if err != nil {
				return completion, err
			}
			evidence = appendEvidence(evidence, receipt.ID)
			if replica.Peer.Valid() {
				if _, done := retired[replica.Peer]; !done {
					state, found, resolveErr := runrecord.ResolvePeerState(ctx, reconciler.repository, replica.Peer)
					if resolveErr != nil {
						return completion, resolveErr
					}
					if found && state.State == runrecord.PeerDraining {
						if _, transitionErr := runrecord.PublishPeerState(
							ctx, reconciler.repository, replica.Peer, runrecord.PeerRetired, request.ChangedUnixNS,
						); transitionErr != nil {
							return completion, transitionErr
						}
					}
					retired[replica.Peer] = struct{}{}
				}
			}
		}
		previous := make(map[artifact.ID]struct{}, len(request.Previous))
		for _, replica := range request.Previous {
			target, _ := peerReplicaTarget(replica)
			previous[target] = struct{}{}
		}
		for _, replica := range request.Plan.Replicas {
			target, err := peerReplicaTarget(replica)
			if err != nil {
				return completion, err
			}
			if _, retained := previous[target]; retained {
				continue
			}
			stage, err := reconciler.runAction(
				ctx, reporter, request, target, runrecord.PeerReplicaStage,
				peerReplicaArtifacts(replica),
				func(ctx context.Context) error { return reconciler.backend.Stage(ctx, replica) },
			)
			if err != nil {
				return completion, err
			}
			evidence = appendEvidence(evidence, stage.ID)
			loaded, err := reconciler.runAction(
				ctx, reporter, request, target, runrecord.PeerReplicaLoad,
				peerReplicaArtifacts(replica),
				func(ctx context.Context) error { return reconciler.backend.Load(ctx, replica) },
			)
			if err != nil {
				return completion, err
			}
			evidence = appendEvidence(evidence, loaded.ID)
		}
		completion.Outputs = evidence
		return completion, nil
	}
}

func (reconciler *PeerReplicaReconciler) runAction(
	ctx context.Context,
	reporter operation.Reporter,
	request PeerReconcileRequest,
	target artifact.ID,
	phase runrecord.PeerReplicaPhase,
	artifacts []artifact.ID,
	action func(context.Context) error,
) (runrecord.PeerReplicaReceipt, error) {
	current, found, err := runrecord.ResolvePeerReplicaReceipt(ctx, reconciler.repository, reporter.OperationID(), target, phase)
	if err != nil {
		return runrecord.PeerReplicaReceipt{}, err
	}
	if found && current.Outcome == runrecord.PeerReplicaSucceeded {
		return current, nil
	}
	attempt := uint32(artifact.InitialDocumentVersion)
	if found {
		attempt = current.Attempt
		attempt++
	}
	for attempt <= request.Policy.MaximumAttempts {
		if err := ctx.Err(); err != nil {
			return runrecord.PeerReplicaReceipt{}, err
		}
		actionErr := action(ctx)
		outcome, failure := runrecord.PeerReplicaSucceeded, ""
		if actionErr != nil {
			outcome, failure = runrecord.PeerReplicaFailed, string(phase)+"_failed"
		}
		receipt, publishErr := runrecord.PublishPeerReplicaReceipt(ctx, reconciler.repository, runrecord.PeerReplicaReceipt{
			Plan: request.Plan.Identity, Operation: reporter.OperationID(), Target: target,
			Artifacts: artifacts, Phase: phase, Attempt: attempt, Outcome: outcome, Failure: failure,
		})
		if publishErr != nil {
			return runrecord.PeerReplicaReceipt{}, publishErr
		}
		reporter.Attempt(receipt.ID)
		if actionErr == nil {
			return receipt, nil
		}
		attempt++
	}
	return runrecord.PeerReplicaReceipt{}, errors.New("capability runtime: peer reconciliation attempts exhausted")
}

func (reconciler *PeerReplicaReconciler) publishAttempt(
	ctx context.Context,
	reporter operation.Reporter,
	plan, target artifact.ID,
	phase runrecord.PeerReplicaPhase,
	outcome runrecord.PeerReplicaOutcome,
	failure string,
	artifacts []artifact.ID,
) (runrecord.PeerReplicaReceipt, error) {
	current, found, err := runrecord.ResolvePeerReplicaReceipt(ctx, reconciler.repository, reporter.OperationID(), target, phase)
	if err != nil {
		return runrecord.PeerReplicaReceipt{}, err
	}
	attempt := uint32(artifact.InitialDocumentVersion)
	if found {
		attempt = current.Attempt
		attempt++
	}
	receipt, err := runrecord.PublishPeerReplicaReceipt(ctx, reconciler.repository, runrecord.PeerReplicaReceipt{
		Plan: plan, Operation: reporter.OperationID(), Target: target,
		Artifacts: artifacts, Phase: phase, Attempt: attempt, Outcome: outcome, Failure: failure,
	})
	if err == nil {
		reporter.Attempt(receipt.ID)
	}
	return receipt, err
}

func peerReplicaArtifacts(replica modelrecipe.PeerReplicaPlacement) []artifact.ID {
	result := make([]artifact.ID, 0, len(replica.Locality))
	for _, location := range replica.Locality {
		if location.Artifact.Valid() && !slices.Contains(result, location.Artifact) {
			result = append(result, location.Artifact)
		}
	}
	slices.SortFunc(result, artifact.CompareID)
	return result
}

func peerReplicaTarget(replica modelrecipe.PeerReplicaPlacement) (artifact.ID, error) {
	return artifact.JSONID(artifact.KindProfile, struct {
		Peer        artifact.ID         `json:"peer,omitzero"`
		Environment artifact.ID         `json:"environment"`
		Endpoint    string              `json:"endpoint,omitempty"`
		Locality    []artifact.Location `json:"locality"`
	}{replica.Peer, replica.Environment, replica.Endpoint, replica.Locality})
}

func appendEvidence(evidence []artifact.ID, id artifact.ID) []artifact.ID {
	if id.Valid() && !slices.Contains(evidence, id) {
		return append(evidence, id)
	}
	return evidence
}
