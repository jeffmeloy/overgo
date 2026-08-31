package inference

import (
	"strings"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor/reference"
)

func TestNonCausalRunnerRejectsCacheEntryPoint(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "bert"}, AttentionSpec: model.AttentionSpec{NonCausalAttention: true}}}}
	runner = attachFixtureProgram(runner)
	_, _, err := runner.ForwardCached(t.Context(), nil, nil)
	if err == nil || !strings.Contains(err.Error(), "do not support KV caching") {
		t.Fatalf("ForwardCached error = %v", err)
	}
}

func TestEncoderRunnersRejectVocabularyLogits(t *testing.T) {
	for _, architecture := range []string{"bert", "llama-embed"} {
		t.Run(architecture, func(t *testing.T) {
			runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture}}}}
			runner = attachFixtureProgram(runner)
			_, err := runner.projectAllLogits(t.Context(), reference.Value{})
			if err == nil || !strings.Contains(err.Error(), "hidden states") {
				t.Fatalf("encoder logits error = %v", err)
			}
		})
	}
}

func TestLFM2RunnerUsesCompiledNonCausalForward(t *testing.T) {
	for _, architecture := range []string{"lfm2", "lfm2moe"} {
		t.Run(architecture, func(t *testing.T) {
			runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture}}}}
			runner = attachFixtureProgram(runner)
			_, err := runner.Forward(t.Context(), nil)
			if err == nil || !strings.Contains(err.Error(), "token sequence is empty") {
				t.Fatalf("Forward error = %v", err)
			}
		})
	}
}
