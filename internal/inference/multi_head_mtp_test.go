package inference

import (
	"context"
	"os"
	"slices"
	"testing"

	"overgo/internal/model"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func TestCompiledMultiHeadMTPRequiresCompleteCatalog(t *testing.T) {
	for _, fixture := range []struct {
		architecture string
		weights      model.Weights
		truncate     func(*model.Weights)
	}{
		{architecture: "step35", weights: model.Weights{AppendedMultiCarryDraft: make([]model.AppendedDraftWeights, 2)},
			truncate: func(weights *model.Weights) { weights.AppendedMultiCarryDraft = weights.AppendedMultiCarryDraft[:1] }},
		{architecture: "hy_v3", weights: model.Weights{AppendedMultiDraft: make([]model.AppendedDraftWeights, 2)},
			truncate: func(weights *model.Weights) { weights.AppendedMultiDraft = weights.AppendedMultiDraft[:1] }},
	} {
		runner := attachFixtureProgram(&Runner{preparedModel: preparedModel{
			spec: model.Spec{CommonSpec: model.CommonSpec{
				Architecture: fixture.architecture, NextNPredictLayers: 2,
			}},
			weights: fixture.weights,
		}})
		if _, err := runner.multiHeadMTP(); err != nil {
			t.Fatalf("%s complete catalog: %v", fixture.architecture, err)
		}
		fixture.truncate(&runner.weights)
		if _, err := runner.multiHeadMTP(); err == nil {
			t.Fatalf("%s incomplete catalog was accepted", fixture.architecture)
		}
	}
}

func TestLastValueColumn(t *testing.T) {
	value := reference.Value{
		Shape: tensor.MustShape(2, 3),
		Data:  []float32{1, 2, 3, 4, 5, 6},
	}
	last, err := lastValueColumn(value)
	if err != nil {
		t.Fatal(err)
	}
	if !last.Shape.Equal(tensor.MustShape(2, 1)) || len(last.Data) != 2 || last.Data[0] != 5 || last.Data[1] != 6 {
		t.Fatalf("unexpected final column: %+v", last)
	}
}

func TestStep35MTPChainsIndependentHeads(t *testing.T) {
	requireIntegration(t)
	modelPath := os.Getenv("OVERGO_STEP35_MTP_MODEL")
	if modelPath == "" {
		t.Skip("OVERGO_STEP35_MTP_MODEL is not set")
	}
	runner, err := openNativeFixtureRunner(modelPath, OpenOptions{
		DeviceOrdinal: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	session, err := runner.NewMultiHeadMTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	if len(session.Heads) != int(runner.spec.NextNPredictLayers) {
		t.Fatalf("Step3.5 MTP head count = %d", len(session.Heads))
	}
	logits, next, err := runner.AdvanceMultiHeadMTP(ctx, 0, session)
	if err != nil {
		t.Fatal(err)
	}
	if logits.Shape.Dims[0] != uint64(runner.spec.VocabularySize) ||
		next.Position != session.Position+1 || len(next.DraftTokens) != 1 ||
		next.Heads[0].Key.Shape.Dims[2] != uint64(session.MTPStart+1) {
		t.Fatalf("unexpected Step3.5 head-0 state: logits=%v before=%+v after=%+v", logits.Shape, session, next)
	}
	if len(next.Heads) < 2 {
		return
	}
	_, third, err := runner.AdvanceMultiHeadMTP(ctx, 0, next)
	if err != nil {
		t.Fatal(err)
	}
	if third.Position != next.Position+1 || len(third.DraftTokens) != 2 ||
		third.Heads[1].Key.Shape.Dims[2] != uint64(session.MTPStart+2) {
		t.Fatalf("Step3.5 head-1 cache did not rebuild the draft prefix: %+v", third)
	}
	coordinatorSession, err := runner.NewMultiHeadMTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	draft, err := runner.DraftMultiHeadMTPGreedy(ctx, 0, coordinatorSession, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(draft.Tokens) > int(runner.spec.NextNPredictLayers) {
		t.Fatalf("Step3.5 greedy draft exceeded trained heads: %+v", draft)
	}
	verification, err := runner.VerifyMultiHeadMTPGreedy(ctx, runner, draft)
	if err != nil {
		t.Fatal(err)
	}
	if verification.Accepted < 0 || verification.Accepted > len(draft.Tokens) ||
		verification.Session.Position != coordinatorSession.Position+uint32(verification.Accepted)+1 ||
		verification.Session.TrunkCache.Position != verification.Session.Position ||
		len(verification.Session.DraftTokens) != 0 {
		t.Fatalf("unexpected Step3.5 greedy verification: draft=%+v result=%+v", draft, verification)
	}
	draftSampler, err := sampling.New(sampling.Config{
		Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 17,
	})
	if err != nil {
		t.Fatal(err)
	}
	targetSampler, err := sampling.New(sampling.Config{
		Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 23,
	})
	if err != nil {
		t.Fatal(err)
	}
	draftSamplerBefore, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	sampledDraft, err := runner.DraftMultiHeadMTPSampled(
		ctx, coordinatorSession, draftSampler, []tokenizer.TokenID{0}, 10, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	draftSamplerAfter, err := draftSampler.SaveState()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(draftSamplerBefore, draftSamplerAfter) {
		t.Fatal("Step3.5 sampled drafting changed caller sampler state")
	}
	sampledVerification, err := runner.VerifyMultiHeadMTPSampled(
		ctx, runner, sampledDraft, draftSampler, targetSampler,
	)
	if err != nil {
		t.Fatal(err)
	}
	if sampledVerification.Accepted < 0 ||
		sampledVerification.Accepted > len(sampledDraft.Tokens) ||
		sampledVerification.Session.Position !=
			coordinatorSession.Position+uint32(sampledVerification.Accepted)+1 ||
		sampledVerification.Session.TrunkCache.Position != sampledVerification.Session.Position ||
		len(sampledVerification.Session.DraftTokens) != 0 {
		t.Fatalf("unexpected sampled Step3.5 MTP verification: draft=%+v result=%+v", sampledDraft, sampledVerification)
	}
}

func TestHYV3MTPChainsIndependentHeads(t *testing.T) {
	requireIntegration(t)
	modelPath := os.Getenv("OVERGO_HYV3_MTP_MODEL")
	if modelPath == "" {
		t.Skip("OVERGO_HYV3_MTP_MODEL is not set")
	}
	runner, err := openNativeFixtureRunner(modelPath, OpenOptions{
		DeviceOrdinal: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	session, err := runner.NewMultiHeadMTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	logits, next, err := runner.AdvanceMultiHeadMTP(ctx, 0, session)
	if err != nil {
		t.Fatal(err)
	}
	if logits.Shape.Dims[0] != uint64(runner.spec.VocabularySize) ||
		next.Position != session.Position+1 || len(next.DraftTokens) != 1 {
		t.Fatalf("unexpected HY-V3 MTP state: logits=%v before=%+v after=%+v", logits.Shape, session, next)
	}
	coordinatorSession, err := runner.NewMultiHeadMTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	draft, err := runner.DraftMultiHeadMTPGreedy(ctx, 0, coordinatorSession, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	verification, err := runner.VerifyMultiHeadMTPGreedy(ctx, runner, draft)
	if err != nil {
		t.Fatal(err)
	}
	if verification.Accepted < 0 || verification.Accepted > len(draft.Tokens) ||
		verification.Session.Position != coordinatorSession.Position+uint32(verification.Accepted)+1 ||
		len(verification.Session.DraftTokens) != 0 {
		t.Fatalf("unexpected HY-V3 MTP verification: draft=%+v result=%+v", draft, verification)
	}
}
