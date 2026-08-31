package latentvideo

import (
	"context"
	"math"
	"reflect"
	"testing"
)

func TestLiveEditReferenceParity(t *testing.T) {
	t.Run("flow-schedule", func(t *testing.T) {
		got, err := CompileEditFlowSigmas(EditFlowConfig{InferenceSteps: 1000, TrainTimesteps: 1000, Shift: 5, SigmaMax: 1, ExtraStep: true}, []int64{1000, 750, 500, 250})
		if err != nil {
			t.Fatal(err)
		}
		want := []float32{1, 0.7500000596046448, 0.5005995631217957, 0.25159743428230286}
		for index := range want {
			if math.Abs(float64(got[index]-want[index])) > 1e-7 {
				t.Fatalf("sigma[%d]=%.9g want %.9g", index, got[index], want[index])
			}
		}
	})

	plan, err := CompileEditPlan(
		EditModelConfig{PatchSize: [3]int{1, 2, 2}, InputChannels: 8, Dim: 12, TextLength: 7, Layers: 2, TrainSteps: 1000},
		LatentGeometry{Channels: 4, LatentFrames: 7, LatentHeight: 6, LatentWidth: 10},
		4, 3, 3, []int64{1000, 500, 0},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.ChunkFrames, []int{3, 3, 1}) || plan.TokensPerFrame != 15 || plan.SelfAttentionTokens != 45 || plan.CrossAttentionTokens != 7 || plan.DenoiserCalls != 12 {
		t.Fatalf("plan=%+v", plan)
	}

	t.Run("bounded-sampler", func(t *testing.T) {
		samplerPlan := EditPlan{
			Latent:         LatentGeometry{Channels: 1, LatentFrames: 3, LatentHeight: 1, LatentWidth: 1},
			SourceChannels: 2, InputChannels: 3, TokensPerFrame: 1,
			ChunkFrames: []int{2, 1}, Timesteps: []int64{1000, 500}, DenoiserCalls: 6, ContextRefreshCalls: 2,
		}
		noise := [][]float32{{10, 11}, {12}}
		noiseCall := 0
		var visits []EditStep
		got, stats, err := RunEditSampler(t.Context(), EditSamplerRequest{
			Plan: samplerPlan, InitialNoise: []float32{1, 2, 3}, Source: []float32{101, 102, 103, 201, 202, 203},
			Sigmas: []float32{1, 0.25}, Arithmetic: EditFP32,
			Denoise: func(_ context.Context, step EditStep, combined, flow []float32) error {
				visits = append(visits, step)
				clear(flow)
				return nil
			},
			Noise: func(_ context.Context, _ EditStep, destination []float32) error {
				copy(destination, noise[noiseCall])
				noiseCall++
				return nil
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, []float32{3.25, 4.25, 5.25}) || stats.DenoiseCalls != 4 || stats.ContextRefreshCalls != 2 || stats.NoiseCalls != 2 || stats.PeakScratchElements != 12 || len(visits) != 6 {
			t.Fatalf("output=%v stats=%+v visits=%d", got, stats, len(visits))
		}
	})
}
