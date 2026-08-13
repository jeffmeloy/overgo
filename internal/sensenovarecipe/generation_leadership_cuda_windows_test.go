//go:build windows

package sensenovarecipe

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/latentimage"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/torchrng"
)

type generationSample struct {
	Elements int       `json:"elements"`
	Indices  []int     `json:"indices"`
	Values   []float32 `json:"values"`
}

type generationStep struct {
	Timestep              float64          `json:"timestep"`
	NextTimestep          float64          `json:"next_timestep"`
	Z                     generationSample `json:"z"`
	ConditionalBoundary   generationSample `json:"conditional_boundary"`
	UnconditionalBoundary generationSample `json:"unconditional_boundary"`
	GuidedVelocity        generationSample `json:"guided_velocity"`
	NextZ                 generationSample `json:"next_z"`
}

type generationLeadershipOracle struct {
	Schema  string `json:"schema"`
	Request struct {
		Width, Height int
		Seed          int64
		CFGScale      float32 `json:"cfg_scale"`
	} `json:"request"`
	TextInputs struct {
		Conditional   []int `json:"conditional_ids"`
		Unconditional []int `json:"unconditional_ids"`
	} `json:"text_inputs"`
	Steps []generationStep `json:"steps"`
}

type generationAdaptiveLayer struct {
	Conditional, Unconditional []float32
}

type generationAdaptiveOracle struct {
	Schema string                             `json:"schema"`
	Layers map[string]generationAdaptiveLayer `json:"layers"`
	Step0  struct {
		ConditionalBoundary   []float32 `json:"conditional_boundary"`
		UnconditionalBoundary []float32 `json:"unconditional_boundary"`
		GuidedVelocity        []float32 `json:"guided_velocity"`
		NextZ                 []float32 `json:"next_z"`
	} `json:"step0"`
	AdaptiveUpstreamCosine map[string]float64 `json:"adaptive_upstream_cosine"`
}

func TestSenseNovaGenerationLeadership(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv("OVERGO_SENSENOVA_BASELINE") != "1" {
		t.Skip("set OVERGO_SENSENOVA_BASELINE=1 for SenseNova generation evidence")
	}
	var oracle generationLeadershipOracle
	if err := jsonfile.Decode(senseNovaGenerationGold, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != "adaptive_gpt.sensenova_generation_oracle/v1" || len(oracle.Steps) == 0 {
		t.Fatalf("generation oracle schema=%q steps=%d", oracle.Schema, len(oracle.Steps))
	}
	var adaptive generationAdaptiveOracle
	if err := jsonfile.Decode(filepath.Join("..", "..", "fixtures", "sensenova", "generation_body_adaptive.json"), &adaptive); err != nil {
		t.Fatal(err)
	}
	if adaptive.Schema != "overgo.sensenova-generation-adaptive/v1" {
		t.Fatalf("adaptive generation schema=%q", adaptive.Schema)
	}
	var prefixGold prefixOracle
	if err := jsonfile.Decode(filepath.Join("..", "..", "fixtures", "sensenova", "prefix_oracle.json"), &prefixGold); err != nil {
		t.Fatal(err)
	}
	binding, flowBinding := routedlm.SenseNovaBinding(), routedlm.SenseNovaFlowBinding()
	cfg, err := routedlm.LoadConfig(senseNovaModelDir, binding)
	if err != nil {
		t.Fatal(err)
	}
	flowCfg, err := routedlm.LoadFlowConfig(senseNovaModelDir)
	if err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenSource(senseNovaModelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	rope, err := routedlm.CompileRopePlan(source, cfg, binding)
	if err != nil {
		t.Fatal(err)
	}
	flow, err := routedlm.CompileFlowPlan(source, cfg, flowCfg, flowBinding)
	if err != nil {
		t.Fatal(err)
	}
	image, err := flow.ImagePlan(oracle.Request.Width, oracle.Request.Height)
	if err != nil {
		t.Fatal(err)
	}
	vision, err := routedlm.LoadVisionEmbedderWeights(source, flowBinding.GenerationVisionPrefix, flow)
	if err != nil {
		t.Fatal(err)
	}
	flowTerminal, err := routedlm.LoadFlowTerminalWeights(source, flow, flowBinding)
	if err != nil {
		t.Fatal(err)
	}
	terminal, err := routedlm.LoadTerminalWeights(source, cfg, binding)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	cuda, err := executor.NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	started := time.Now()
	prefixNames := []string{"conditional", "unconditional"}
	prefixIDs := [][]int{oracle.TextInputs.Conditional, oracle.TextInputs.Unconditional}
	for index, name := range prefixNames {
		gold := prefixGold.Branches[name]
		if len(prefixIDs[index]) != gold.IDElements || intSHA256(prefixIDs[index]) != gold.IDSHA || gold.ImageTime != len(prefixIDs[index]) {
			t.Fatalf("%s input identity differs", name)
		}
	}
	prefixes, prefixStats, err := routedlm.RunDevicePrefixStacks(
		context.Background(), worker, cuda, source, cfg, binding, rope,
		nil,
		routedlm.DevicePrefixInput{TokenIDs: prefixIDs[0], ImageTime: len(prefixIDs[0])},
		routedlm.DevicePrefixInput{TokenIDs: prefixIDs[1], ImageTime: len(prefixIDs[1])},
	)
	if err != nil {
		t.Fatal(err)
	}
	conditional, unconditional := prefixes[0], prefixes[1]
	t.Logf("SenseNova shared prefix stream: layers=%d branches=%d wall=%.3fs", prefixStats.Layers, prefixStats.Branches, prefixStats.Wall.Seconds())
	var z []float32
	err = worker.Do(context.Background(), func(state *device.State) error {
		stream := torchrng.NewStream(oracle.Request.Seed)
		defer stream.Close(state)
		z, err = routedlm.SeededFlowLatent(stream, state, flow, image)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	for stepIndex, want := range oracle.Steps[:1] {
		checkGenerationSample(t, "z", z, want.Z)
		planar, err := latentimage.UnpackPlanarF32(z, flow.VisionChannels, image.TokenHeight, image.TokenWidth, image.TokenPatch, latentimage.PatchChannelsLast)
		if err != nil {
			t.Fatal(err)
		}
		hidden, gotImage, err := routedlm.VisionEmbedTokens(vision, flow, planar, image.Width, image.Height)
		if err != nil || gotImage != image {
			t.Fatalf("vision embedding: shape=%+v err=%v", gotImage, err)
		}
		condition, err := routedlm.FlowConditionRow(flowTerminal, flow, want.Timestep, flow.NormalizedNoiseScale(image.NoiseScale))
		if err != nil {
			t.Fatal(err)
		}
		if err := routedlm.AddConditionRows(hidden, condition); err != nil {
			t.Fatal(err)
		}
		branches, stats, err := routedlm.RunDeviceGenerationStackObserved(
			context.Background(), worker, cuda, source, cfg, binding, rope, image, hidden,
			func(branch, layer int, hidden []float32) {
				if stepIndex != 0 {
					return
				}
				if layer != 0 && layer != 20 && layer != 41 {
					return
				}
				gold := adaptive.Layers[fmt.Sprint(layer)]
				values := gold.Conditional
				if branch == 1 {
					values = gold.Unconditional
				}
				checkAdaptiveValues(t, fmt.Sprintf("body branch=%d layer=%d", branch, layer), hidden, values)
			}, conditional, unconditional,
		)
		if err != nil {
			t.Fatal(err)
		}
		final := make([][]float32, len(branches))
		for index := range branches {
			final[index], err = routedlm.GenerationFinalHidden(branches[index], terminal.FinalNorm[1], image.Tokens, cfg.HiddenSize, cfg.RMSNormEps)
			if err != nil {
				t.Fatal(err)
			}
		}
		checkAdaptiveValues(t, "conditional boundary", final[0], adaptive.Step0.ConditionalBoundary)
		checkAdaptiveValues(t, "unconditional boundary", final[1], adaptive.Step0.UnconditionalBoundary)
		velocities := make([][]float32, len(final))
		for index := range final {
			velocities[index], err = routedlm.FlowHeadVelocity(flowTerminal.Head, flow, final[index], z, want.Timestep)
			if err != nil {
				t.Fatal(err)
			}
		}
		guided, err := routedlm.GuidedFlowVelocity(velocities, []float32{oracle.Request.CFGScale, 1 - oracle.Request.CFGScale})
		if err != nil {
			t.Fatal(err)
		}
		checkAdaptiveValues(t, "guided velocity", guided, adaptive.Step0.GuidedVelocity)
		if err := routedlm.FlowEulerStep(z, guided, float32(want.NextTimestep-want.Timestep)); err != nil {
			t.Fatal(err)
		}
		checkAdaptiveValues(t, "next z", z, adaptive.Step0.NextZ)
		for name, comparison := range map[string]struct {
			got  []float32
			want generationSample
		}{
			"conditional_boundary":   {final[0], want.ConditionalBoundary},
			"unconditional_boundary": {final[1], want.UnconditionalBoundary},
			"guided_velocity":        {guided, want.GuidedVelocity},
			"next_z":                 {z, want.NextZ},
		} {
			cosine := generationSampleCosine(comparison.got, comparison.want)
			floor := adaptive.AdaptiveUpstreamCosine[name] - 0.002
			if cosine < floor {
				t.Fatalf("%s upstream cosine=%.9f below adaptive floor=%.9f", name, cosine, floor)
			}
			t.Logf("%s upstream cosine=%.9f adaptive=%.9f status=red", name, cosine, adaptive.AdaptiveUpstreamCosine[name])
		}
		t.Logf("SenseNova neutral generation step=%d layers=%d branches=%d body=%.3fs", stepIndex, stats.Layers, stats.Branches, stats.Wall.Seconds())
	}
	memory, err := worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SenseNova neutral generation leadership: steps=1 wall=%.3fs peak=%.3fGiB", time.Since(started).Seconds(), float64(memory.PeakBytes)/(1<<30))
}

func checkGenerationSample(t *testing.T, name string, got []float32, want generationSample) {
	t.Helper()
	if len(got) != want.Elements || len(want.Indices) != len(want.Values) || len(want.Indices) == 0 {
		t.Fatalf("%s invalid elements/probes got=%d want=%d", name, len(got), want.Elements)
	}
	dot, gotNorm, wantNorm, worst := 0.0, 0.0, 0.0, 0.0
	for probe, index := range want.Indices {
		gotValue, wantValue := float64(got[index]), float64(want.Values[probe])
		dot += gotValue * wantValue
		gotNorm += gotValue * gotValue
		wantNorm += wantValue * wantValue
		worst = max(worst, math.Abs(gotValue-wantValue))
	}
	cosine := dot / math.Sqrt(gotNorm*wantNorm)
	const floor = 1 - 8.0/256
	if cosine < floor {
		gotValues := make([]float32, len(want.Indices))
		for probe, index := range want.Indices {
			gotValues[probe] = got[index]
		}
		t.Fatalf("%s cosine=%.9f floor=%.9f worst=%.3e got=%v want=%v", name, cosine, floor, worst, gotValues, want.Values)
	}
	for index, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("%s value[%d]=%g", name, index, value)
		}
	}
	t.Logf("%s probes=%d cosine=%.9f worst=%.3e", name, len(want.Indices), cosine, worst)
}

func checkAdaptiveValues(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) == 0 || len(want) == 0 {
		t.Fatalf("%s empty adaptive comparison", name)
	}
	samples := make([]float32, len(want))
	for index := range samples {
		samples[index] = got[index*(len(got)-1)/(len(samples)-1)]
	}
	dot, gotNorm, wantNorm, worst := 0.0, 0.0, 0.0, 0.0
	for index := range samples {
		g, w := float64(samples[index]), float64(want[index])
		dot += g * w
		gotNorm += g * g
		wantNorm += w * w
		worst = max(worst, math.Abs(g-w))
	}
	cosine := dot / math.Sqrt(gotNorm*wantNorm)
	const floor = 1 - 8.0/256
	if cosine < floor {
		t.Fatalf("%s adaptive cosine=%.9f floor=%.9f got=%v want=%v", name, cosine, floor, samples, want)
	}
	t.Logf("%s adaptive probes=%d cosine=%.9f worst=%.3e", name, len(samples), cosine, worst)
}

func generationSampleCosine(got []float32, want generationSample) float64 {
	dot, gotNorm, wantNorm := 0.0, 0.0, 0.0
	for probe, index := range want.Indices {
		g, w := float64(got[index]), float64(want.Values[probe])
		dot += g * w
		gotNorm += g * g
		wantNorm += w * w
	}
	return dot / math.Sqrt(gotNorm*wantNorm)
}
