package inference

import (
	"context"
	"errors"

	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

// Gemma4AssistantDraft: bounded greedy proposals.
type Gemma4AssistantDraft = greedyDraft[*Gemma4AssistantSession]

// Gemma4AssistantVerification: accepted prefix plus correction.
type Gemma4AssistantVerification = greedyVerification[*Gemma4AssistantSession]

// DraftGemma4AssistantGreedy: bounded confident proposals.
func (r *Runner) DraftGemma4AssistantGreedy(
	ctx context.Context,
	target *Runner,
	initialToken tokenizer.TokenID,
	session *Gemma4AssistantSession,
	maximum int,
	minimumProbability float64,
) (*Gemma4AssistantDraft, error) {
	if r == nil || target == nil || session == nil ||
		!validSampledLimits(maximum, minimumProbability) {
		return nil, errors.New("inference: Gemma 4 assistant draft inputs are invalid")
	}
	return draftGreedy(
		initialToken, session, maximum, minimumProbability, target.vocab.IsEOG,
		func(token tokenizer.TokenID, state *Gemma4AssistantSession) (reference.Value, *Gemma4AssistantSession, error) {
			return r.AdvanceGemma4Assistant(ctx, target, token, state)
		},
	)
}

// VerifyGemma4AssistantGreedy: target verification and hidden resync.
func (r *Runner) VerifyGemma4AssistantGreedy(
	ctx context.Context,
	target *Runner,
	draft *Gemma4AssistantDraft,
) (*Gemma4AssistantVerification, error) {
	if r == nil || target == nil || !validGreedyDraft(draft) ||
		draft.Base == nil || draft.Base.TargetCache == nil {
		return nil, errors.New("inference: Gemma 4 assistant verification inputs are invalid")
	}
	if err := r.validateGemma4AssistantTarget(target); err != nil {
		return nil, err
	}
	return verifyGreedy(draft, func(
		token tokenizer.TokenID,
		state *Gemma4AssistantSession,
	) (reference.Value, *Gemma4AssistantSession, error) {
		return r.advanceGemma4AssistantVerification(
			ctx, target, token, state, state.TargetCache,
		)
	})
}
