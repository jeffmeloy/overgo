//go:build modeltest

package routedlm

import (
	"math"
	"os"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// Real-checkpoint shapes: every derived dim comes from config + tensors.
func TestSenseNovaFlowTerminalRealArtifact(t *testing.T) {
	if _, err := os.Stat(senseNovaDir(t)); err != nil {
		t.Skipf("UNAVAILABLE: artifact absent at %s: %v", senseNovaDir(t), err)
	}
	binding := SenseNovaBinding()
	cfg := testcheck.Must(LoadConfig(senseNovaDir(t), binding))(t)
	flowCfg := testcheck.Must(LoadFlowConfig(senseNovaDir(t)))(t)
	src := testcheck.Must(safetensors.OpenSource(senseNovaDir(t)))(t)
	defer src.Close()
	flow := testcheck.Must(InspectFlowProfile(senseNovaDir(t)))(t)
	plan := testcheck.Must(CompileFlowPlan(src, cfg, flowCfg, flow))(t)
	want := testFlowPlan()
	if plan != want {
		t.Fatalf("plan = %+v, want %+v", plan, want)
	}
	// noise_scale_base_image_seq_len is the 256x256 reference token count.
	shape := testcheck.Must(plan.ImagePlan(256, 256))(t)
	if shape.Tokens != plan.NoiseScaleBase {
		t.Fatalf("256x256 tokens = %d, base = %d", shape.Tokens, plan.NoiseScaleBase)
	}
	if shape.NoiseScale != 1 || plan.NormalizedNoiseScale(shape.NoiseScale) != 0.125 {
		t.Fatalf("256x256 noise scale = %g (normalized %g)", shape.NoiseScale, plan.NormalizedNoiseScale(shape.NoiseScale))
	}
	weights := testcheck.Must(LoadFlowTerminalWeights(src, plan, flow))(t)
	condition := testcheck.Must(FlowConditionRow(weights, plan, 0, plan.NormalizedNoiseScale(shape.NoiseScale)))(t)
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
		embedder := testcheck.Must(LoadVisionEmbedderWeights(src, prefix, plan))(t)
		image := make([]float32, plan.VisionChannels*64*64)
		for i := range image {
			image[i] = float32(i%17)/17 - 0.5
		}
		tokens, tokenShape, err := VisionEmbedTokens(embedder, plan, image, 64, 64)
		testcheck.NoError(t, err)
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
	velocity := testcheck.Must(FlowHeadVelocity(weights.Head, plan, hidden, z, 0))(t)
	if len(velocity) != plan.FlowDim {
		t.Fatalf("velocity len = %d, want %d", len(velocity), plan.FlowDim)
	}
	for _, v := range velocity {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("velocity has non-finite values")
		}
	}
}
