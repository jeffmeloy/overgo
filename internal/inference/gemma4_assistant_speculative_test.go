package inference

import (
	"context"
	"testing"

	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func TestGemma4AssistantCoordinatorRejectsInvalidState(t *testing.T) {
	runner, target := &Runner{}, &Runner{}
	if _, err := runner.DraftGemma4AssistantGreedy(context.Background(), target, 0, nil, 1, 0); err == nil {
		t.Fatal("nil draft session was accepted")
	}
	if _, err := runner.DraftGemma4AssistantGreedy(context.Background(), target, 0, &Gemma4AssistantSession{}, 0, 0); err == nil {
		t.Fatal("zero draft bound was accepted")
	}
	draft := &Gemma4AssistantDraft{
		InitialToken: 0, Tokens: []tokenizer.TokenID{1}, Probabilities: nil,
		Base: &Gemma4AssistantSession{TargetCache: &KVCache{}, PendingHidden: reference.Value{}},
	}
	if _, err := runner.VerifyGemma4AssistantGreedy(context.Background(), target, draft); err == nil {
		t.Fatal("inconsistent draft was accepted")
	}
}
