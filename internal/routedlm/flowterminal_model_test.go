//go:build modeltest

package routedlm

import (
	"math"
	"os"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

// Real-checkpoint shapes: every derived dim comes from config + tensors.
func TestSenseNovaFlowTerminalRealArtifact(t *testing.T) {
	if _, err := os.Stat(senseNovaDir(t)); err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s: %v", senseNovaDir(t), err)
	}
	binding := SenseNovaBinding()
	cfg, err := LoadConfig(senseNovaDir(t), binding)
	if err != nil {
		t.Fatal(err)
	}
	flowCfg, err := LoadFlowConfig(senseNovaDir(t))
	if err != nil {
		t.Fatal(err)
	}
	src, err := safetensors.OpenSource(senseNovaDir(t))
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	flow, err := InspectFlowProfile(senseNovaDir(t))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := CompileFlowPlan(src, cfg, flowCfg, flow)
	if err != nil {
		t.Fatal(err)
	}
	want := testFlowPlan()
	if plan != want {
		t.Fatalf("plan = %+v, want %+v", plan, want)
	}
	// noise_scale_base_image_seq_len is the 256x256 reference token count.
	shape, err := plan.ImagePlan(256, 256)
	if err != nil {
		t.Fatal(err)
	}
	if shape.Tokens != plan.NoiseScaleBase {
		t.Fatalf("256x256 tokens = %d, base = %d", shape.Tokens, plan.NoiseScaleBase)
	}
	if shape.NoiseScale != 1 || plan.NormalizedNoiseScale(shape.NoiseScale) != 0.125 {
		t.Fatalf("256x256 noise scale = %g (normalized %g)", shape.NoiseScale, plan.NormalizedNoiseScale(shape.NoiseScale))
	}
	weights, err := LoadFlowTerminalWeights(src, plan, flow)
	if err != nil {
		t.Fatal(err)
	}
	condition, err := FlowConditionRow(weights, plan, 0, plan.NormalizedNoiseScale(shape.NoiseScale))
	if err != nil {
		t.Fatal(err)
	}
	if len(condition) != plan.Hidden {
		t.Fatalf("condition len = %d, want %d", len(condition), plan.Hidden)
	}
	nonzero := 0
	for _, v := range condition {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("condition row has non-finite values")
		}
		if v != 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("condition row is all zeros")
	}
	for _, prefix := range []string{flow.SourceVisionPrefix, flow.GenerationVisionPrefix} {
		embedder, err := LoadVisionEmbedderWeights(src, prefix, plan)
		if err != nil {
			t.Fatal(err)
		}
		image := make([]float32, plan.VisionChannels*64*64)
		for i := range image {
			image[i] = float32(i%17)/17 - 0.5
		}
		tokens, tokenShape, err := VisionEmbedTokens(embedder, plan, image, 64, 64)
		if err != nil {
			t.Fatal(err)
		}
		if tokenShape.Tokens != 4 || len(tokens) != 4*plan.Hidden {
			t.Fatalf("%s: tokens = %d len = %d", prefix, tokenShape.Tokens, len(tokens))
		}
		for _, v := range tokens {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s: non-finite token embedding", prefix)
			}
		}
	}
	// flow head on one real-width row.
	hidden := make([]float32, plan.Hidden)
	z := make([]float32, plan.FlowDim)
	for i := range hidden {
		hidden[i] = dtype.RoundBF16(float32(i%13)/13 - 0.5)
	}
	velocity, err := FlowHeadVelocity(weights.Head, plan, hidden, z, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(velocity) != plan.FlowDim {
		t.Fatalf("velocity len = %d, want %d", len(velocity), plan.FlowDim)
	}
	for _, v := range velocity {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("velocity has non-finite values")
		}
	}
}
