package runrecord

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/textcheck"
)

const (
	// CheckpointObligationMediaType identifies checkpoint obligations.
	CheckpointObligationMediaType = "application/vnd.overgo.checkpoint-obligation+json"
	// CheckpointObligationSchema identifies the checkpoint obligation schema.
	CheckpointObligationSchema = "overgo/checkpoint-obligation/v1"
	// CheckpointPromotionMediaType identifies checkpoint promotions.
	CheckpointPromotionMediaType = "application/vnd.overgo.checkpoint-promotion+json"
	// CheckpointPromotionSchema identifies the checkpoint promotion schema.
	CheckpointPromotionSchema = "overgo/checkpoint-promotion/v1"
)

// CheckpointObligation is one accepted checkpoint under a declared flush key:
// durable once committed, outstanding until a promotion names it, bound to
// the exact memo input the acceptance observed.
type CheckpointObligation struct {
	Version      uint16      `json:"version"`
	ID           artifact.ID `json:"-"`
	Key          string      `json:"key"`
	PlanRef      string      `json:"plan_ref"`
	Checkpoint   string      `json:"checkpoint"`
	Slot         artifact.ID `json:"slot"`
	Input        artifact.ID `json:"input"`
	Evidence     string      `json:"evidence"`
	Accepted     string      `json:"accepted"`
	PayloadBytes int64       `json:"payload_bytes"`
}

// CheckpointPromotion binds every obligation one flush promoted to the full
// gate result that carried them; absence means the obligations are open.
type CheckpointPromotion struct {
	Version     uint16        `json:"version"`
	ID          artifact.ID   `json:"-"`
	Key         string        `json:"key"`
	Reason      string        `json:"reason"`
	Obligations []artifact.ID `json:"obligations"`
	Result      artifact.ID   `json:"result"`
}

var checkpointObligationCodec = artifact.JSONDocumentCodec(
	"checkpoint obligation", artifact.KindEvidence, CheckpointObligationMediaType, CheckpointObligationSchema,
	canonicalizeCheckpointObligation, func(v CheckpointObligation) artifact.ID { return v.ID },
	func(v *CheckpointObligation, id artifact.ID) { v.ID = id }, nil,
)
var checkpointPromotionCodec = artifact.JSONDocumentCodec(
	"checkpoint promotion", artifact.KindEvidence, CheckpointPromotionMediaType, CheckpointPromotionSchema,
	canonicalizeCheckpointPromotion, func(v CheckpointPromotion) artifact.ID { return v.ID },
	func(v *CheckpointPromotion, id artifact.ID) { v.ID = id },
	func(v CheckpointPromotion) CheckpointPromotion { v.Obligations = slices.Clone(v.Obligations); return v },
)

// NewCheckpointObligation canonicalizes and identifies one immutable obligation.
func NewCheckpointObligation(value CheckpointObligation) (CheckpointObligation, error) {
	return checkpointObligationCodec.NewInitial(value)
}

// NewCheckpointPromotion canonicalizes and identifies one immutable promotion.
func NewCheckpointPromotion(value CheckpointPromotion) (CheckpointPromotion, error) {
	return checkpointPromotionCodec.NewInitial(value)
}

// Content returns the canonical committed bytes of the obligation.
func (v CheckpointObligation) Content() (artifact.Content, error) {
	return checkpointObligationCodec.Content(v)
}

// Content returns the canonical committed bytes of the promotion.
func (v CheckpointPromotion) Content() (artifact.Content, error) {
	return checkpointPromotionCodec.Content(v)
}

// Lineage links the promotion to its obligations and the gate result.
func (v CheckpointPromotion) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(v.ID, append(slices.Clone(v.Obligations), v.Result)...)
}

// RequireCheckpointPromotion loads one exact typed promotion.
func RequireCheckpointPromotion(ctx context.Context, reader artifact.Reader, id artifact.ID) (CheckpointPromotion, error) {
	return checkpointPromotionCodec.RequireExactLineage(ctx, reader, id, CheckpointPromotion.Lineage)
}

// LoadOutstandingCheckpointObligations reads every committed obligation of
// key that no committed promotion names, ordered by acceptance time then id.
func LoadOutstandingCheckpointObligations(ctx context.Context, store *overgodb.Store, key string) ([]CheckpointObligation, error) {
	if ctx == nil || store == nil {
		return nil, errors.New("run record: outstanding obligations require the store")
	}
	promoted := map[artifact.ID]bool{}
	promotions, err := store.Query(ctx, overgodb.Query{
		Kind: artifact.KindEvidence, MediaType: CheckpointPromotionMediaType, Schema: CheckpointPromotionSchema,
		MaxResults: MaximumAttemptPopulation, Projection: overgodb.ProjectContentPresence,
	})
	if err != nil {
		return nil, err
	}
	for _, content := range promotions.Contents {
		// Exact lineage: a promotion whose obligations or result are absent
		// from the store cannot close anything.
		promotion, err := RequireCheckpointPromotion(ctx, store, content.Artifact)
		if err != nil {
			return nil, err
		}
		if promotion.Key == key {
			for _, id := range promotion.Obligations {
				promoted[id] = true
			}
		}
	}
	obligations, err := store.Query(ctx, overgodb.Query{
		Kind: artifact.KindEvidence, MediaType: CheckpointObligationMediaType, Schema: CheckpointObligationSchema,
		MaxResults: MaximumAttemptPopulation, Projection: overgodb.ProjectContentPresence,
	})
	if err != nil {
		return nil, err
	}
	var outstanding []CheckpointObligation
	for _, content := range obligations.Contents {
		obligation, found, err := checkpointObligationCodec.Read(ctx, store, content.Artifact)
		if err != nil {
			return nil, err
		}
		if found && obligation.Key == key && !promoted[obligation.ID] {
			outstanding = append(outstanding, obligation)
		}
	}
	slices.SortFunc(outstanding, func(left, right CheckpointObligation) int {
		return cmp.Or(strings.Compare(left.Accepted, right.Accepted), artifact.CompareID(left.ID, right.ID))
	})
	return outstanding, nil
}

func canonicalizeCheckpointObligation(v *CheckpointObligation) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || v.Key == "" || !strings.Contains(v.PlanRef, "/") ||
		!textcheck.LowerIdentifier(v.Checkpoint, len(v.Checkpoint)) || v.Slot.Kind() != artifact.KindRecipe ||
		v.Input.Kind() != artifact.KindProfile || v.Evidence == "" || v.Accepted == "" || v.PayloadBytes < 0 {
		return errors.New("run record: invalid checkpoint obligation")
	}
	return nil
}

func canonicalizeCheckpointPromotion(v *CheckpointPromotion) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || v.Key == "" || v.Reason == "" ||
		len(v.Obligations) == 0 || v.Result.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid checkpoint promotion")
	}
	for _, id := range v.Obligations {
		if id.Kind() != artifact.KindEvidence {
			return errors.New("run record: invalid promoted obligation")
		}
	}
	slices.SortFunc(v.Obligations, artifact.CompareID)
	v.Obligations = slices.Compact(v.Obligations)
	return nil
}
