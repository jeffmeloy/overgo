package inference

import (
	"context"
	"testing"

	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func TestPairedProjectionCoordinatorRejectsInvalidState(t *testing.T) {
	runner, target := &Runner{}, &Runner{}
	if _, err := runner.DraftPairedProjectionGreedy(context.Background(), target, 0, nil, 1, 0); err == nil {
		t.Fatal("nil draft session was accepted")
	}
	if _, err := runner.DraftPairedProjectionGreedy(context.Background(), target, 0, &PairedProjectionSession{}, 0, 0); err == nil {
		t.Fatal("zero draft bound was accepted")
	}
	draft := &PairedProjectionDraft{
		InitialToken: 0, Tokens: []tokenizer.TokenID{1}, Probabilities: nil,
		Base: &PairedProjectionSession{TargetCache: &KVCache{}, PendingHidden: reference.Value{}},
	}
	if _, err := runner.VerifyPairedProjectionGreedy(context.Background(), target, draft); err == nil {
		t.Fatal("inconsistent draft was accepted")
	}
}
