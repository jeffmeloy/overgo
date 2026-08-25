package runrecord

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// PeerReplicaReceiptMediaType identifies replica reconciliation evidence.
	PeerReplicaReceiptMediaType = "application/vnd.overgo.peer-replica-receipt+json"
	// PeerReplicaReceiptSchema identifies the replica reconciliation contract.
	PeerReplicaReceiptSchema = "overgo/peer-replica-receipt/v1"
	// PeerReplicaReceiptAliasRoot scopes current action evidence by operation and target.
	PeerReplicaReceiptAliasRoot = "peer/replica-receipt/"
)

// PeerReplicaPhase identifies one idempotent reconciliation action.
type PeerReplicaPhase string

const (
	// PeerReplicaStage proves artifact staging.
	PeerReplicaStage PeerReplicaPhase = "stage"
	// PeerReplicaLoad proves whole-model loading.
	PeerReplicaLoad PeerReplicaPhase = "load"
	// PeerReplicaUnload proves zero-active-lease unloading.
	PeerReplicaUnload PeerReplicaPhase = "unload"
)

// PeerReplicaOutcome records an action attempt result.
type PeerReplicaOutcome string

const (
	// PeerReplicaSucceeded marks durable action completion.
	PeerReplicaSucceeded PeerReplicaOutcome = "succeeded"
	// PeerReplicaFailed marks a retryable backend failure.
	PeerReplicaFailed PeerReplicaOutcome = "failed"
	// PeerReplicaWaiting marks an unload blocked by active leases.
	PeerReplicaWaiting PeerReplicaOutcome = "waiting"
)

// PeerReplicaReceipt records one immutable action attempt.
type PeerReplicaReceipt struct {
	Version   uint16             `json:"version"`
	Plan      artifact.ID        `json:"plan"`
	Operation artifact.ID        `json:"operation"`
	Target    artifact.ID        `json:"target"`
	Artifacts []artifact.ID      `json:"artifacts,omitempty"`
	Phase     PeerReplicaPhase   `json:"phase"`
	Attempt   uint32             `json:"attempt"`
	Outcome   PeerReplicaOutcome `json:"outcome"`
	Failure   string             `json:"failure,omitempty"`
	Previous  artifact.ID        `json:"previous,omitzero"`
	ID        artifact.ID        `json:"-"`
}

var peerReplicaReceiptCodec = artifact.JSONDocumentCodec(
	"peer replica receipt", artifact.KindEvidence, PeerReplicaReceiptMediaType, PeerReplicaReceiptSchema,
	canonicalizePeerReplicaReceipt,
	func(value PeerReplicaReceipt) artifact.ID { return value.ID },
	func(value *PeerReplicaReceipt, id artifact.ID) { value.ID = id },
	func(value PeerReplicaReceipt) PeerReplicaReceipt {
		value.Artifacts = slices.Clone(value.Artifacts)
		return value
	},
)

// ParsePeerReplicaReceipt decodes one exact reconciliation attempt.
func ParsePeerReplicaReceipt(content []byte) (PeerReplicaReceipt, error) {
	return peerReplicaReceiptCodec.Parse(content)
}

// ResolvePeerReplicaReceipt returns current evidence for one operation action.
func ResolvePeerReplicaReceipt(
	ctx context.Context,
	reader artifact.Reader,
	operation, target artifact.ID,
	phase PeerReplicaPhase,
) (PeerReplicaReceipt, bool, error) {
	id, found, err := artifact.ResolveAlias(ctx, reader, peerReplicaReceiptAlias(operation, target, phase))
	if err != nil || !found {
		return PeerReplicaReceipt{}, found, err
	}
	value, err := peerReplicaReceiptCodec.Require(ctx, reader, id)
	return value, err == nil, err
}

// PublishPeerReplicaReceipt advances one exact action attempt chain.
func PublishPeerReplicaReceipt(
	ctx context.Context,
	repository artifact.Repository,
	value PeerReplicaReceipt,
) (PeerReplicaReceipt, error) {
	if ctx == nil || repository == nil || value.Previous.Valid() {
		return PeerReplicaReceipt{}, errors.New("run record: invalid peer replica receipt publication")
	}
	previous, found, err := ResolvePeerReplicaReceipt(ctx, repository, value.Operation, value.Target, value.Phase)
	if err != nil {
		return PeerReplicaReceipt{}, err
	}
	if found {
		next := previous.Attempt
		next++
		if value.Attempt != next || previous.Plan != value.Plan {
			return PeerReplicaReceipt{}, errors.New("run record: peer replica receipt chain differs")
		}
		value.Previous = previous.ID
	} else if value.Attempt != uint32(artifact.InitialDocumentVersion) {
		return PeerReplicaReceipt{}, errors.New("run record: peer replica receipt must start at initial attempt")
	}
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	value, err = peerReplicaReceiptCodec.New(value)
	if err != nil {
		return PeerReplicaReceipt{}, err
	}
	content, err := peerReplicaReceiptCodec.Content(value)
	if err != nil {
		return PeerReplicaReceipt{}, err
	}
	alias := artifact.AliasBinding{
		Name: peerReplicaReceiptAlias(value.Operation, value.Target, value.Phase), Target: value.ID,
	}
	if found {
		alias.Previous = artifact.IDPointer(previous.ID)
	}
	parents := append([]artifact.ID{value.Plan, value.Operation, value.Target}, value.Artifacts...)
	if value.Previous.Valid() {
		parents = append(parents, value.Previous)
	}
	batch, err := artifact.NewDocumentBatch(
		"peer/replica-receipt/"+value.ID.String(), []artifact.Content{content},
		artifact.DependencyLineage(value.ID, parents...), []artifact.AliasBinding{alias},
	)
	if err != nil {
		return PeerReplicaReceipt{}, err
	}
	batch.Artifacts = append(batch.Artifacts,
		artifact.Descriptor{ID: value.Plan}, artifact.Descriptor{ID: value.Operation}, artifact.Descriptor{ID: value.Target},
	)
	for _, id := range value.Artifacts {
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: id})
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return PeerReplicaReceipt{}, err
	}
	return value, nil
}

func canonicalizePeerReplicaReceipt(value *PeerReplicaReceipt) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Plan.Kind() != artifact.KindProfile ||
		value.Operation.Kind() != artifact.KindEvidence || value.Target.Kind() != artifact.KindProfile || value.Attempt == 0 ||
		(value.Phase != PeerReplicaStage && value.Phase != PeerReplicaLoad && value.Phase != PeerReplicaUnload) ||
		(value.Outcome != PeerReplicaSucceeded && value.Outcome != PeerReplicaFailed && value.Outcome != PeerReplicaWaiting) ||
		(value.Outcome == PeerReplicaSucceeded) != (value.Failure == "") ||
		value.Failure != "" && (strings.TrimSpace(value.Failure) != value.Failure || strings.ContainsAny(value.Failure, " \t\x00\r\n")) ||
		value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid peer replica receipt")
	}
	for index, id := range value.Artifacts {
		if !id.Valid() || slices.Contains(value.Artifacts[:index], id) {
			return errors.New("run record: invalid peer replica receipt artifacts")
		}
	}
	slices.SortFunc(value.Artifacts, artifact.CompareID)
	return nil
}

func peerReplicaReceiptAlias(operation, target artifact.ID, phase PeerReplicaPhase) string {
	return PeerReplicaReceiptAliasRoot + operation.String() + "/" + target.String() + "/" + fmt.Sprint(phase)
}
