package model

import (
	"testing"

	"overgo/internal/gguf"
)

// TestMetadataProgramReadsEitherEpsilonAndDerivesExpertWidths covers the
// compiled metadata reads no per-architecture spec test reaches: Cohere2-MoE
// accepts either norm epsilon (and rebinds its normalization to RMS) but
// refuses neither, and Qwen2-MoE derives absent expert widths from the model.
func TestMetadataProgramReadsEitherEpsilonAndDerivesExpertWidths(t *testing.T) {
	cohere2moeProfile, _ := LookupArchitecture("cohere2moe")
	epsilonCases := []struct {
		values    []gguf.Metadata
		wantRMS   float32
		wantLayer float32
		wantError bool
	}{
		{[]gguf.Metadata{metadata("cohere2moe.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6))}, 1e-6, 0, false},
		{[]gguf.Metadata{metadata("cohere2moe.attention.layer_norm_epsilon", gguf.ValueTypeFloat32, float32(1e-5))}, 0, 1e-5, false},
		{nil, 0, 0, true},
	}
	for index, entry := range epsilonCases {
		values := make(map[string]gguf.Value, len(entry.values))
		for _, item := range entry.values {
			values[item.Key] = item.Value
		}
		reader := specMetadata{values: values, architecture: "cohere2moe", prefix: "cohere2moe.", profile: cohere2moeProfile}
		bound := cohere2moeProfile
		probe := Spec{CommonSpec: CommonSpec{Architecture: "cohere2moe"}}
		probe.profile = &bound
		result, programErr := reader.runMetadataProgram(probe, specReadState{}, []metadataOp{{kind: metadataOpEpsilonEither}})
		if entry.wantError {
			if programErr == nil {
				t.Fatalf("epsilon case %d: missing epsilon accepted", index)
			}
			continue
		}
		if programErr != nil {
			t.Fatalf("epsilon case %d: %v", index, programErr)
		}
		if result.RMSNormEpsilon != entry.wantRMS || result.LayerNormEpsilon != entry.wantLayer {
			t.Fatalf("epsilon case %d: got rms=%g layer=%g", index, result.RMSNormEpsilon, result.LayerNormEpsilon)
		}
		if entry.wantRMS > 0 && bound.Normalization != NormalizationRMS {
			t.Fatalf("epsilon case %d: profile normalization was not rebound", index)
		}
	}

	qwen2moeProfile, _ := LookupArchitecture("qwen2moe")
	qwen2moeReader := specMetadata{values: map[string]gguf.Value{}, architecture: "qwen2moe", prefix: "qwen2moe.", profile: qwen2moeProfile}
	qwen2moeSpec, qwen2moeErr := qwen2moeReader.runMetadataProgram(
		Spec{CommonSpec: CommonSpec{FeedForwardLength: 48}}, specReadState{}, compileExpertProgram(qwen2moeProfile),
	)
	if qwen2moeErr != nil {
		t.Fatal(qwen2moeErr)
	}
	if qwen2moeSpec.ExpertFeedForward != 48 || qwen2moeSpec.SharedExpertCount != 1 ||
		qwen2moeSpec.SharedExpertFF != 48 {
		t.Fatalf("unexpected Qwen2-MoE compiled expert spec: %+v", qwen2moeSpec)
	}
}
