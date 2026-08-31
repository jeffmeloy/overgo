package runrecord

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
)

// OperatorTimelineEvent is one causally ordered committed fact about an
// operation: the durable artifact that records it, its store introduction
// sequence, and the identities it cites. The timeline derives from canonical
// records only — it duplicates no domain state and grants no authority.
type OperatorTimelineEvent struct {
	Sequence uint64        `json:"sequence"`
	Kind     string        `json:"kind"`
	Artifact artifact.ID   `json:"artifact"`
	Detail   string        `json:"detail,omitzero"`
	Cites    []artifact.ID `json:"cites,omitempty"`
}

// PendingOperatorDecision is one committed approval request still awaiting
// its human decision, with the exact tool identity and argument surface the
// operator grants or declines. Deciding removes the row only because the
// durable decision resolves — the view holds no state of its own.
type PendingOperatorDecision struct {
	Operation artifact.ID `json:"operation"`
	Request   artifact.ID `json:"request"`
	Recipe    artifact.ID `json:"recipe"`
	Tool      string      `json:"tool"`
	Arguments []string    `json:"arguments"`
	Prior     artifact.ID `json:"prior,omitzero"`
}

// DeriveOperatorTimeline projects one causal timeline for an operation from
// committed stage receipts, human decisions, and stimulus follow-ups, ordered
// by exact store introduction sequence.
func DeriveOperatorTimeline(
	ctx context.Context,
	store *overgodb.Store,
	operation artifact.ID,
	limit int,
) ([]OperatorTimelineEvent, error) {
	if store == nil || operation.Kind() != artifact.KindEvidence || limit <= 0 {
		return nil, errors.New("run record: operator timeline requires a store, an operation, and a bound")
	}
	events := []OperatorTimelineEvent{}
	appendEvent := func(kind string, id artifact.ID, detail string, cites ...artifact.ID) error {
		introduction, found, err := store.ArtifactIntroduction(ctx, id)
		if err != nil {
			return err
		}
		if !found {
			return nil
		}
		events = append(events, OperatorTimelineEvent{
			Sequence: introduction.Sequence, Kind: kind, Artifact: id, Detail: detail, Cites: cites,
		})
		return nil
	}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: StageReceiptMediaType, Schema: StageReceiptSchema,
		}}, Order: overgodb.DocumentOldestFirst,
	}, ParseStageReceipt, func(_ overgodb.DocumentView, receipt StageReceipt) error {
		if receipt.Operation != operation {
			return nil
		}
		return appendEvent("stage-receipt", receipt.ID, string(receipt.State), receipt.Operation)
	})
	if err != nil {
		return nil, err
	}
	if decision, found, err := ResolveHumanDecision(ctx, store, operation); err != nil {
		return nil, err
	} else if found {
		if err := appendEvent("human-decision", decision.ID, string(decision.Answer), decision.Request); err != nil {
			return nil, err
		}
	}
	if followup, found, err := (StimulusFollowupAuthority{Repository: store}).Current(ctx, operation); err == nil && found {
		if err := appendEvent("stimulus-followup", followup.ID, "", followup.Boundary); err != nil {
			return nil, err
		}
	}
	slices.SortStableFunc(events, func(left, right OperatorTimelineEvent) int {
		return cmp.Compare(left.Sequence, right.Sequence)
	})
	if len(events) > limit {
		events = events[:limit]
	}
	return events, nil
}

// PendingOperatorDecisions lists committed approval requests whose operation
// still lacks a committed human decision, in stable operation order.
func PendingOperatorDecisions(
	ctx context.Context,
	store *overgodb.Store,
	limit int,
) ([]PendingOperatorDecision, error) {
	if store == nil || limit <= 0 {
		return nil, errors.New("run record: pending decisions require a store and a bound")
	}
	pending := []PendingOperatorDecision{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{operatoraction.ApprovalDocumentContract()},
		Order:     overgodb.DocumentOldestFirst,
	}, operatoraction.ParseApprovalRequest, func(_ overgodb.DocumentView, request operatoraction.ApprovalRequest) error {
		if len(pending) >= limit {
			return nil
		}
		if _, decided, err := ResolveHumanDecision(ctx, store, request.Operation); err != nil {
			return err
		} else if decided {
			return nil
		}
		pending = append(pending, PendingOperatorDecision{
			Operation: request.Operation, Request: request.ID, Recipe: request.Recipe,
			Tool: request.Tool, Arguments: request.Arguments, Prior: request.Prior,
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	slices.SortStableFunc(pending, func(left, right PendingOperatorDecision) int {
		return strings.Compare(left.Operation.String(), right.Operation.String())
	})
	return pending, nil
}
