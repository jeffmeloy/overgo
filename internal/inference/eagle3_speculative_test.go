package inference

import (
	"context"
	"testing"

	"overgo/internal/tokenizer"
)

func TestEagle3CoordinatorRejectsInvalidState(t *testing.T) {
	runner, target := &Runner{}, &Runner{}
	if _, err := runner.DraftEagle3Greedy(context.Background(), target, 0, nil, 1, 0); err == nil {
		t.Fatal("nil Eagle3 session was accepted")
	}
	draft := &Eagle3Draft{
		Tokens: []tokenizer.TokenID{1}, Base: &Eagle3Session{TargetCache: &KVCache{}, TargetTokens: []tokenizer.TokenID{0}},
	}
	if _, err := runner.VerifyEagle3Greedy(context.Background(), target, draft); err == nil {
		t.Fatal("invalid Eagle3 coordinator state was accepted")
	}
}
