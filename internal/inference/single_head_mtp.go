package inference

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/model"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// MTPSession: target snapshot plus independent single-head draft state.
type MTPSession struct {
	TrunkCache    *KVCache
	Layer         LayerCache
	PendingHidden reference.Value
	MTPStart      uint32
	Position      uint32
	targetModel   [32]byte
}

func (r *Runner) singleHeadMTP() (model.DraftPlan, model.DraftWeightCatalog, error) {
	if r == nil {
		return model.DraftPlan{}, model.DraftWeightCatalog{}, errors.New("inference: MTP runner is nil")
	}
	plan, catalog, ok := r.lookupSingleHeadMTP()
	if !ok {
		return model.DraftPlan{}, model.DraftWeightCatalog{}, errors.New("inference: model has no complete single-head MTP program")
	}
	return plan, catalog, nil
}

func (r *Runner) lookupSingleHeadMTP() (model.DraftPlan, model.DraftWeightCatalog, bool) {
	if r == nil {
		return model.DraftPlan{}, model.DraftWeightCatalog{}, false
	}
	plan := r.program.Model.Draft()
	if !plan.SingleCatalog || plan.Session != model.DraftSessionSingle || !plan.SessionEligible() {
		return model.DraftPlan{}, model.DraftWeightCatalog{}, false
	}
	catalog, ok := r.weights.DraftCatalog(plan.Kind, 0)
	if !ok || catalog.Kind != plan.Kind {
		return model.DraftPlan{}, model.DraftWeightCatalog{}, false
	}
	return plan, catalog, true
}

// NewMTPSession compiles one bundled target-prefix draft session.
func (r *Runner) NewMTPSession(ctx context.Context, tokenIDs []tokenizer.TokenID) (*MTPSession, error) {
	if len(tokenIDs) == 0 {
		return nil, errors.New("inference: MTP inputs are empty")
	}
	plan, catalog, err := r.singleHeadMTP()
	if err != nil {
		return nil, err
	}
	if catalog.MTPOnly {
		return nil, fmt.Errorf("inference: %s-only model requires a paired target session", plan.Label)
	}
	return r.newSingleHeadMTPSession(ctx, tokenIDs)
}

// NewMTPPairedSession compiles one sidecar/target prefix session.
func (r *Runner) NewMTPPairedSession(
	ctx context.Context,
	target *Runner,
	tokenIDs []tokenizer.TokenID,
) (*MTPSession, error) {
	if r == nil || target == nil || r == target || r.path == target.path || len(tokenIDs) == 0 {
		return nil, errors.New("inference: MTP sidecar and target inputs are invalid")
	}
	if err := r.validateMTPTarget(target); err != nil {
		return nil, err
	}
	return target.newSingleHeadMTPSession(ctx, tokenIDs)
}

// AdvanceMTP executes one compiled single-head draft step.
func (r *Runner) AdvanceMTP(
	ctx context.Context,
	tokenID tokenizer.TokenID,
	session *MTPSession,
) (reference.Value, *MTPSession, error) {
	if r == nil || session == nil || session.TrunkCache == nil {
		return reference.Value{}, nil, errors.New("inference: MTP session is invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: MTP runner is unavailable")
	}
	_, catalog, err := r.singleHeadMTP()
	if err != nil {
		return reference.Value{}, nil, err
	}
	if err := r.validateMTPSession(session); err != nil {
		return reference.Value{}, nil, err
	}
	if tokenID < 0 || int(tokenID) >= r.vocab.Len() {
		return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", tokenID)
	}
	program, err := r.draftLayerProgram(0)
	if err != nil {
		return reference.Value{}, nil, err
	}
	return r.advanceSingleHeadMTP(ctx, tokenID, session, singleHeadMTPAdapter{
		nodePrefix: "mtp", layer: catalog.Layer, program: program,
		embeddingNorm: catalog.EmbeddingNorm, hiddenNorm: catalog.HiddenNorm, project: catalog.EHProjection,
		tokenEmbedding: catalog.TokenEmbedding, outputNorm: catalog.OutputNorm, output: catalog.Output,
	})
}

func (r *Runner) validateMTPSession(session *MTPSession) error {
	plan, _, err := r.singleHeadMTP()
	if err != nil {
		return err
	}
	return r.validateSingleHeadMTPSession(session, plan.Label, true)
}

func (r *Runner) validateMTPTarget(target *Runner) error {
	plan, catalog, err := r.singleHeadMTP()
	if err != nil {
		return err
	}
	targetMTPOnly := false
	if target != nil {
		if targetPlan, targetCatalog, ok := target.lookupSingleHeadMTP(); ok && targetPlan.Kind == plan.Kind {
			targetMTPOnly = targetCatalog.MTPOnly
		}
	}
	return r.validateSingleHeadMTPTarget(target, catalog.MTPOnly, targetMTPOnly, plan.Label)
}
