package inference

import (
	"context"
	"testing"

	"overgo/internal/tokenizer"
)

func TestFeatureDraftCoordinatorRejectsInvalidState(t *testing.T) {
	runner, target := &Runner{}, &Runner{}
	if _, err := runner.DraftFeaturesGreedy(context.Background(), target, 0, nil, 1, 0); err == nil {
		t.Fatal("nil feature-draft session was accepted")
	}
	draft := &FeatureDraft{
		Tokens: []tokenizer.TokenID{1}, Base: &FeatureDraftSession{TargetCache: &KVCache{}, TargetTokens: []tokenizer.TokenID{0}},
	}
	if _, err := runner.VerifyFeatureDraftGreedy(context.Background(), target, draft); err == nil {
		t.Fatal("invalid feature-draft coordinator state was accepted")
	}
}
