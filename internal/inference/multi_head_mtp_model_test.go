//go:build modeltest

package inference

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modeltest"
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
)

func TestStep35MTPChainsIndependentHeads(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openNativeFixtureRunner(modeltest.GGUF(t, "step35"), OpenOptions{
		DeviceOrdinal: 0,
	})
	testcheck.NoError(t, err)
	defer runner.Close()
	ctx := t.Context()
	session, err := runner.NewMultiHeadMTPSession(ctx, []tokenizer.TokenID{0})
	testcheck.NoError(t, err)
	if len(session.Heads) != int(runner.spec.NextNPredictLayers) {
		t.Fatalf("Step3.5 MTP head count = %d", len(session.Heads))
	}
	logits, next, err := runner.AdvanceMultiHeadMTP(ctx, 0, session)
	testcheck.NoError(t, err)
	if logits.Shape.Dims[0] != uint64(runner.spec.VocabularySize) ||
		next.Position != session.Position+1 || len(next.DraftTokens) != 1 ||
		next.Heads[0].Key.Shape.Dims[2] != uint64(session.MTPStart+1) {
		t.Fatalf("unexpected Step3.5 head-0 state: logits=%v before=%+v after=%+v", logits.Shape, session, next)
	}
	if len(next.Heads) < 2 {
		return
	}
	_, third, err := runner.AdvanceMultiHeadMTP(ctx, 0, next)
	testcheck.NoError(t, err)
	if third.Position != next.Position+1 || len(third.DraftTokens) != 2 ||
		third.Heads[1].Key.Shape.Dims[2] != uint64(session.MTPStart+2) {
		t.Fatalf("Step3.5 head-1 cache did not rebuild the draft prefix: %+v", third)
	}
	coordinatorSession, err := runner.NewMultiHeadMTPSession(ctx, []tokenizer.TokenID{0})
	testcheck.NoError(t, err)
	draft, err := runner.DraftMultiHeadMTPGreedy(ctx, 0, coordinatorSession, 10, 0)
	testcheck.NoError(t, err)
	if len(draft.Tokens) > int(runner.spec.NextNPredictLayers) {
		t.Fatalf("Step3.5 greedy draft exceeded trained heads: %+v", draft)
	}
	verification, err := runner.VerifyMultiHeadMTPGreedy(ctx, runner, draft)
	testcheck.NoError(t, err)
	if verification.Accepted < 0 || verification.Accepted > len(draft.Tokens) ||
		verification.Session.Position != coordinatorSession.Position+uint32(verification.Accepted)+1 ||
		verification.Session.TrunkCache.Position != verification.Session.Position ||
		len(verification.Session.DraftTokens) != 0 {
		t.Fatalf("unexpected Step3.5 greedy verification: draft=%+v result=%+v", draft, verification)
	}
	draftSampler, err := sampling.New(sampling.Config{
		Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 17,
	})
	testcheck.NoError(t, err)
	targetSampler, err := sampling.New(sampling.Config{
		Temperature: 0.8, TopK: 32, TopP: 0.95, Seed: 23,
	})
	testcheck.NoError(t, err)
	draftSamplerBefore, err := draftSampler.SaveState()
	testcheck.NoError(t, err)
	sampledDraft, err := runner.DraftMultiHeadMTPSampled(
		ctx, coordinatorSession, draftSampler, []tokenizer.TokenID{0}, 10, 0,
	)
	testcheck.NoError(t, err)
	draftSamplerAfter, err := draftSampler.SaveState()
	testcheck.NoError(t, err)
	if !slices.Equal(draftSamplerBefore, draftSamplerAfter) {
		t.Fatal("Step3.5 sampled drafting changed caller sampler state")
	}
	sampledVerification, err := runner.VerifyMultiHeadMTPSampled(
		ctx, runner, sampledDraft, draftSampler, targetSampler,
	)
	testcheck.NoError(t, err)
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
	cudatest.Require(t)
	runner, err := openNativeFixtureRunner(modeltest.GGUF(t, "hy_v3"), OpenOptions{
		DeviceOrdinal: 0,
	})
	testcheck.NoError(t, err)
	defer runner.Close()
	ctx := t.Context()
	session, err := runner.NewMultiHeadMTPSession(ctx, []tokenizer.TokenID{0})
	testcheck.NoError(t, err)
	logits, next, err := runner.AdvanceMultiHeadMTP(ctx, 0, session)
	testcheck.NoError(t, err)
	if logits.Shape.Dims[0] != uint64(runner.spec.VocabularySize) ||
		next.Position != session.Position+1 || len(next.DraftTokens) != 1 {
		t.Fatalf("unexpected HY-V3 MTP state: logits=%v before=%+v after=%+v", logits.Shape, session, next)
	}
	coordinatorSession, err := runner.NewMultiHeadMTPSession(ctx, []tokenizer.TokenID{0})
	testcheck.NoError(t, err)
	draft, err := runner.DraftMultiHeadMTPGreedy(ctx, 0, coordinatorSession, 10, 0)
	testcheck.NoError(t, err)
	verification, err := runner.VerifyMultiHeadMTPGreedy(ctx, runner, draft)
	testcheck.NoError(t, err)
	if verification.Accepted < 0 || verification.Accepted > len(draft.Tokens) ||
		verification.Session.Position != coordinatorSession.Position+uint32(verification.Accepted)+1 ||
		len(verification.Session.DraftTokens) != 0 {
		t.Fatalf("unexpected HY-V3 MTP verification: draft=%+v result=%+v", draft, verification)
	}
}
