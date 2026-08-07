package inference

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor/reference"
)

func TestNonCausalRunnerRejectsCacheEntryPoint(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{AttentionSpec: model.AttentionSpec{NonCausalAttention: true}}}}
	_, _, err := runner.ForwardCached(context.Background(), nil, nil)
	if err == nil || !strings.Contains(err.Error(), "do not support KV caching") {
		t.Fatalf("ForwardCached error = %v", err)
	}
}

func TestEncoderRunnersRejectVocabularyLogits(t *testing.T) {
	for _, architecture := range []string{"bert", "llama-embed"} {
		t.Run(architecture, func(t *testing.T) {
			runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture}}}}
			_, err := runner.projectAllLogits(context.Background(), reference.Value{})
			if err == nil || !strings.Contains(err.Error(), "hidden states") {
				t.Fatalf("encoder logits error = %v", err)
			}
		})
	}
}

func TestCausalRunnerRejectsNonCausalEntryPoint(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{}}}
	_, err := runner.ForwardNonCausal(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "not configured for non-causal") {
		t.Fatalf("ForwardNonCausal error = %v", err)
	}
}

func TestLFM2RunnerAcceptsNonCausalEntryPoint(t *testing.T) {
	for _, architecture := range []string{"lfm2", "lfm2moe"} {
		t.Run(architecture, func(t *testing.T) {
			runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture}}}}
			_, err := runner.ForwardNonCausal(context.Background(), nil)
			if err == nil || !strings.Contains(err.Error(), "token sequence is empty") {
				t.Fatalf("ForwardNonCausal error = %v", err)
			}
		})
	}
}
