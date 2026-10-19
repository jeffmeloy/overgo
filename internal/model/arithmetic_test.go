package model

import (
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"strings"
	"testing"
)

func TestNativeArithmeticAdmission(t *testing.T) {
	profile, ok := LookupArchitecture("qwen35")
	if !ok {
		t.Fatal("missing architecture")
	}
	for _, fault := range []string{"valid", "unknown", "experts", "prediction", "hyper", "topology"} {
		t.Run(fault, func(t *testing.T) {
			spec := Spec{CommonSpec: CommonSpec{Arithmetic: ArithmeticTransformersBF16}}
			candidate := profile
			switch fault {
			case "unknown":
				spec.Arithmetic = "unknown"
			case "experts":
				spec.ExpertCount = 1
			case "prediction":
				spec.NextNPredictLayers = 1
			case "hyper":
				spec.HyperConnectionCount = 1
			case "topology":
				candidate.AttentionGraph.GatedDelta = gatedDeltaInterleavedProjections
			}
			if err := spec.validateArithmetic(candidate); (err == nil) != (fault == "valid") {
				t.Fatal("source arithmetic admission", err)
			}
		})
	}
	for _, fault := range []string{"dynamic", "offset", "empty", "multiple", "axes"} {
		t.Run(fault, func(t *testing.T) {
			b := tensor.NewBuilder()
			context := CachedBlockContext{Builder: b, Positions: []uint32{0}, Sequences: 1}
			switch fault {
			case "dynamic":
				context.CurrentPositions = b.Input("positions", dtype.F32, tensor.MustShape(1))
			case "offset":
				context.Positions[0] = 1
			case "empty":
				context.Positions = nil
			case "multiple":
				context.Sequences = 2
			case "axes":
				axes := [tensor.MaxDimensions][]uint32{{0}, {0}, {0}}
				context.MultiPositions = &axes
			}
			_, err := executeCompiledLayer(blockDispatchOptions{Context: context, Spec: Spec{CommonSpec: CommonSpec{Arithmetic: ArithmeticTransformersBF16}}, Plan: &LayerPlan{MultiAxis: true}})
			if err == nil || !strings.Contains(err.Error(), "native BF16 conditioning") {
				t.Fatal("unverified execution admitted", err)
			}
		})
	}
}
