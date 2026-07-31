package inference

import (
	"context"
	"strings"
	"testing"

	"llamacpp2go/internal/model"
)

func TestNonCausalRunnerRejectsCacheEntryPoint(t *testing.T) {
	runner := &Runner{spec: model.Spec{NonCausalAttention: true}}
	_, _, err := runner.ForwardCached(context.Background(), nil, nil)
	if err == nil || !strings.Contains(err.Error(), "do not support KV caching") {
		t.Fatalf("ForwardCached error = %v", err)
	}
}

func TestCausalRunnerRejectsNonCausalEntryPoint(t *testing.T) {
	runner := &Runner{spec: model.Spec{}}
	_, err := runner.ForwardNonCausal(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "not configured for non-causal") {
		t.Fatalf("ForwardNonCausal error = %v", err)
	}
}
