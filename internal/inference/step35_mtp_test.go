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

func TestValidateStep35MTP(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "step35", NextNPredictLayers: 2}},
		weights: model.Weights{Step35MTP: make([]model.Step35MTPWeights, 2)}},
	}
	runner = attachFixtureProgram(runner)
	if err := runner.validateStep35MTP(); err != nil {
		t.Fatal(err)
	}
	runner.weights.Step35MTP = runner.weights.Step35MTP[:1]
	if err := runner.validateStep35MTP(); err == nil {
		t.Fatal("incomplete Step3.5 MTP head catalog was accepted")
	}
}

func TestValidateHYV3MTP(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "hy_v3", NextNPredictLayers: 2}},
		weights: model.Weights{HYV3MTP: make([]model.Step35MTPWeights, 2)}},
	}
	runner = attachFixtureProgram(runner)
	if err := runner.validateHYV3MTP(); err != nil {
		t.Fatal(err)
	}
	runner.weights.HYV3MTP = runner.weights.HYV3MTP[:1]
	if err := runner.validateHYV3MTP(); err == nil {
		t.Fatal("incomplete HY-V3 MTP head catalog was accepted")
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
	modelPath := os.Getenv("OVERGO_STEP35_MTP_MODEL")
	if modelPath == "" {
		t.Skip("OVERGO_STEP35_MTP_MODEL is not set")
	}
	runner, err := openFixtureRunnerWithOptions(modelPath, OpenOptions{
		DeviceOrdinal: 0, PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	session, err := runner.NewStep35MTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	if len(session.Heads) != int(runner.spec.NextNPredictLayers) {
		t.Fatalf("Step3.5 MTP head count = %d", len(session.Heads))
	}
	logits, next, err := runner.AdvanceStep35MTP(ctx, 0, session)
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
	_, third, err := runner.AdvanceStep35MTP(ctx, 0, next)
	if err != nil {
		t.Fatal(err)
	}
	if third.Position != next.Position+1 || len(third.DraftTokens) != 2 ||
		third.Heads[1].Key.Shape.Dims[2] != uint64(session.MTPStart+2) {
		t.Fatalf("Step3.5 head-1 cache did not rebuild the draft prefix: %+v", third)
	}
	coordinatorSession, err := runner.NewStep35MTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	draft, err := runner.DraftStep35MTPGreedy(ctx, 0, coordinatorSession, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(draft.Tokens) > int(runner.spec.NextNPredictLayers) {
		t.Fatalf("Step3.5 greedy draft exceeded trained heads: %+v", draft)
	}
	verification, err := runner.VerifyStep35MTPGreedy(ctx, runner, draft)
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
	sampledDraft, err := runner.DraftStep35MTPSampled(
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
	sampledVerification, err := runner.VerifyStep35MTPSampled(
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
	modelPath := os.Getenv("OVERGO_HYV3_MTP_MODEL")
	if modelPath == "" {
		t.Skip("OVERGO_HYV3_MTP_MODEL is not set")
	}
	runner, err := openFixtureRunnerWithOptions(modelPath, OpenOptions{
		DeviceOrdinal: 0, PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	session, err := runner.NewHYV3MTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	logits, next, err := runner.AdvanceHYV3MTP(ctx, 0, session)
	if err != nil {
		t.Fatal(err)
	}
	if logits.Shape.Dims[0] != uint64(runner.spec.VocabularySize) ||
		next.Position != session.Position+1 || len(next.DraftTokens) != 1 {
		t.Fatalf("unexpected HY-V3 MTP state: logits=%v before=%+v after=%+v", logits.Shape, session, next)
	}
	coordinatorSession, err := runner.NewHYV3MTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	draft, err := runner.DraftHYV3MTPGreedy(ctx, 0, coordinatorSession, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	verification, err := runner.VerifyHYV3MTPGreedy(ctx, runner, draft)
	if err != nil {
		t.Fatal(err)
	}
	if verification.Accepted < 0 || verification.Accepted > len(draft.Tokens) ||
		verification.Session.Position != coordinatorSession.Position+uint32(verification.Accepted)+1 ||
		len(verification.Session.DraftTokens) != 0 {
		t.Fatalf("unexpected HY-V3 MTP verification: draft=%+v result=%+v", draft, verification)
	}
}
