//go:build windows

package sensenovarecipe

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
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
	Schema string `json:"schema"`
	Source struct {
		InferencePath  string `json:"inference_path"`
		InferenceSHA   string `json:"inference_sha256"`
		ModelingPath   string `json:"modeling_path"`
		ModelingSHA    string `json:"modeling_sha256"`
		ConfigSHA      string `json:"config_sha256"`
		TensorIndexSHA string `json:"tensor_index_sha256"`
	} `json:"source"`
	Request struct {
		Width, Height int
		Steps         int
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
	Performance struct {
		AdaptiveMatchedWallSeconds   float64 `json:"adaptive_matched_wall_seconds"`
		OvergoColdBodyMaxSeconds     float64 `json:"overgo_cold_body_max_seconds"`
		OvergoReusableBodyMaxSeconds float64 `json:"overgo_reusable_body_max_seconds"`
	} `json:"performance"`
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
	if oracle.Schema != "overgo.sensenova-generation-native/v1" || len(oracle.Steps) != oracle.Request.Steps {
		t.Fatalf("generation oracle schema=%q steps=%d", oracle.Schema, len(oracle.Steps))
	}
	checkGenerationSource(t, oracle)
	var adaptive generationAdaptiveOracle
	if err := jsonfile.Decode(filepath.Join("..", "..", "fixtures", "sensenova", "generation_body_adaptive.json"), &adaptive); err != nil {
		t.Fatal(err)
	}
	if adaptive.Schema != "overgo.sensenova-generation-adaptive/v1" {
		t.Fatalf("adaptive generation schema=%q", adaptive.Schema)
	}
	if adaptive.Performance.AdaptiveMatchedWallSeconds <= 0 || adaptive.Performance.OvergoColdBodyMaxSeconds <= 0 ||
		adaptive.Performance.OvergoReusableBodyMaxSeconds <= 0 {
		t.Fatal("SenseNova lifecycle performance envelope is incomplete")
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
	generation, err := routedlm.NewDeviceGenerationSession(
		context.Background(), worker, cuda, source, cfg, binding, rope, image, conditional, unconditional,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer generation.Close(context.Background())
	sessionStats := generation.Stats()
	t.Logf("SenseNova retained generation session: graphs=%d setup=%.3fs prefix=%.3fGiB",
		sessionStats.Graphs, sessionStats.SetupWall.Seconds(), float64(sessionStats.PrefixBytes)/(1<<30))
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
	for stepIndex, want := range oracle.Steps {
		zFloor := 0.999
		if stepIndex == 0 {
			zFloor = 0.999999
		}
		checkGenerationSample(t, "z", z, want.Z, zFloor)
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
		observedLayers := []int(nil)
		if stepIndex == 0 {
			observedLayers = []int{0, 20, 41}
		}
		branches, stats, err := generation.Run(
			context.Background(), hidden, observedLayers,
			func(branch, layer int, hidden []float32) {
				gold := adaptive.Layers[fmt.Sprint(layer)]
				values := gold.Conditional
				if branch == 1 {
					values = gold.Unconditional
				}
				checkAdaptiveValues(t, fmt.Sprintf("body branch=%d layer=%d", branch, layer), hidden, values)
			},
		)
		if err != nil {
			t.Fatal(err)
		}
		limit := adaptive.Performance.OvergoColdBodyMaxSeconds
		lifecycle := "cold"
		if stepIndex > 0 {
			limit = adaptive.Performance.OvergoReusableBodyMaxSeconds
			lifecycle = "warm"
		}
		if stats.Wall.Seconds() > limit {
			t.Fatalf("SenseNova %s body wall=%.3fs exceeds ratchet %.3fs (adaptive %.3fs)", lifecycle, stats.Wall.Seconds(), limit, adaptive.Performance.AdaptiveMatchedWallSeconds)
		}
		final := make([][]float32, len(branches))
		for index := range branches {
			final[index], err = routedlm.GenerationFinalHidden(branches[index], terminal.FinalNorm[1], image.Tokens, cfg.HiddenSize, cfg.RMSNormEps)
			if err != nil {
				t.Fatal(err)
			}
		}
		if stepIndex == 0 {
			checkAdaptiveValues(t, "conditional boundary", final[0], adaptive.Step0.ConditionalBoundary)
			checkAdaptiveValues(t, "unconditional boundary", final[1], adaptive.Step0.UnconditionalBoundary)
		}
		checkGenerationSample(t, "conditional boundary native", final[0], want.ConditionalBoundary, 0.999)
		checkGenerationSample(t, "unconditional boundary native", final[1], want.UnconditionalBoundary, 0.999)
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
		if stepIndex == 0 {
			checkAdaptiveValues(t, "guided velocity", guided, adaptive.Step0.GuidedVelocity)
		}
		checkGenerationSample(t, "guided velocity native", guided, want.GuidedVelocity, 0.995)
		if err := routedlm.FlowEulerStep(z, guided, float32(want.NextTimestep-want.Timestep)); err != nil {
			t.Fatal(err)
		}
		if stepIndex == 0 {
			checkAdaptiveValues(t, "next z", z, adaptive.Step0.NextZ)
		}
		checkGenerationSample(t, "next z native", z, want.NextZ, 0.998)
		lifecycle = "cold-body"
		if stepIndex > 0 {
			lifecycle = "warm-body"
		}
		t.Logf("SenseNova neutral generation step=%d lifecycle=%s layers=%d branches=%d body=%.3fs htod=%d dtoh=%d",
			stepIndex, lifecycle, stats.Layers, stats.Branches, stats.Wall.Seconds(), stats.HostToDevice, stats.DeviceToHost)
	}
	memory, err := worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SenseNova neutral generation leadership: steps=%d wall=%.3fs peak=%.3fGiB", len(oracle.Steps), time.Since(started).Seconds(), float64(memory.PeakBytes)/(1<<30))
}

func checkGenerationSample(t *testing.T, name string, got []float32, want generationSample, floor float64) {
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

func checkGenerationSource(t *testing.T, oracle generationLeadershipOracle) {
	t.Helper()
	checks := map[string]string{
		filepath.Join(senseNovaModelDir, "config.json"):                               oracle.Source.ConfigSHA,
		filepath.Join(senseNovaModelDir, "model.safetensors.index.json"):              oracle.Source.TensorIndexSHA,
		filepath.Join(senseNovaModelDir, "SenseNova-U1", oracle.Source.InferencePath): oracle.Source.InferenceSHA,
		filepath.Join(senseNovaModelDir, "SenseNova-U1", oracle.Source.ModelingPath):  oracle.Source.ModelingSHA,
	}
	for path, want := range checks {
		file, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.New()
		_, copyErr := io.Copy(hash, file)
		closeErr := file.Close()
		if copyErr != nil {
			t.Fatal(copyErr)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		if got := hex.EncodeToString(hash.Sum(nil)); got != want {
			t.Fatalf("generation source %s sha256=%s want %s", path, got, want)
		}
	}
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
