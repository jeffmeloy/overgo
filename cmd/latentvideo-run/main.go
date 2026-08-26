// latentvideo-run: resident Wan denoise, causal decode, and clip evidence.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/dataroot"
	"overgo/internal/latentvideo"
	"overgo/internal/sampling"
	"overgo/internal/tensor/dtype"
)

type tensorSummary struct {
	Elements int     `json:"elements"`
	SHA256   string  `json:"sha256"`
	Min      float64 `json:"min"`
	Max      float64 `json:"max"`
	Mean     float64 `json:"mean"`
	Std      float64 `json:"std"`
	Finite   bool    `json:"finite"`
}

type runReport struct {
	Request        map[string]any  `json:"request"`
	WallSeconds    float64         `json:"wall_seconds"`
	WeightBytes    uint64          `json:"weight_bytes"`
	PeakBytes      uint64          `json:"peak_bytes"`
	Execution      map[string]any  `json:"execution"`
	FinalLatent    tensorSummary   `json:"final_latent"`
	Decode         map[string]any  `json:"decode"`
	FrameSummaries []tensorSummary `json:"frame_summaries"`
	TemporalDeltas []float64       `json:"temporal_mean_abs_deltas"`
	Reference      map[string]any  `json:"reference,omitempty"`
}

func summarize(data []float32) tensorSummary {
	summary := tensorSummary{Elements: len(data), Finite: true, Min: math.Inf(1), Max: math.Inf(-1)}
	hash := sha256.New()
	var sum, squares float64
	finiteCount := 0
	for _, value := range data {
		v := float64(value)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			summary.Finite = false
			continue
		}
		summary.Min = min(summary.Min, v)
		summary.Max = max(summary.Max, v)
		sum += v
		squares += v * v
		finiteCount++
	}
	_, _ = hash.Write(driver.Bytes(data))
	summary.SHA256 = hex.EncodeToString(hash.Sum(nil))
	if finiteCount == 0 {
		summary.Min, summary.Max = 0, 0
		return summary
	}
	summary.Mean = sum / float64(finiteCount)
	summary.Std = math.Sqrt(max(squares/float64(finiteCount)-summary.Mean*summary.Mean, 0))
	return summary
}

func loadContext(path string, elements int) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) == 4*elements+16 && string(raw[:7]) == "WANCTX1" {
		raw = raw[16:]
	}
	if len(raw) != 4*elements {
		return nil, fmt.Errorf("%s: %d bytes, want %d", path, len(raw), 4*elements)
	}
	values := make([]float32, elements)
	for index := range values {
		values[index] = math.Float32frombits(
			uint32(raw[4*index]) | uint32(raw[4*index+1])<<8 |
				uint32(raw[4*index+2])<<16 | uint32(raw[4*index+3])<<24,
		)
	}
	return values, nil
}

func noisePlan(seed, offset uint64, elements, ordinal int) (sampling.CounterNoisePlan, error) {
	library, err := driver.Open()
	if err != nil {
		return sampling.CounterNoisePlan{}, err
	}
	defer library.Close()
	if err := library.Init(); err != nil {
		return sampling.CounterNoisePlan{}, err
	}
	smCount, threadsPerSM, err := library.DeviceProfile(driver.Device(ordinal))
	if err != nil {
		return sampling.CounterNoisePlan{}, err
	}
	plan, _, err := sampling.CompileCounterNoisePlan(seed, offset, elements, smCount, threadsPerSM)
	return plan, err
}

func agreement(got, want []float32) (cosine, normalizedRMS, maxDelta float64) {
	var dot, gotNorm, wantNorm, deltaNorm float64
	for index := range want {
		g, w := float64(got[index]), float64(want[index])
		dot += g * w
		gotNorm += g * g
		wantNorm += w * w
		delta := g - w
		deltaNorm += delta * delta
		maxDelta = max(maxDelta, math.Abs(delta))
	}
	return dot / math.Sqrt(gotNorm*wantNorm), math.Sqrt(deltaNorm / wantNorm), maxDelta
}

func run() error {
	frames := clioptions.IntOverride(flag.CommandLine, "frames", "output frames; unset uses model profile")
	width := clioptions.IntOverride(flag.CommandLine, "width", "output width; unset uses model profile")
	height := clioptions.IntOverride(flag.CommandLine, "height", "output height; unset uses model profile")
	steps := clioptions.IntOverride(flag.CommandLine, "steps", "denoise steps; unset uses model profile")
	shift := clioptions.Float64Override(flag.CommandLine, "shift", "flow shift; unset uses model profile")
	guide := clioptions.Float64Override(flag.CommandLine, "guide", "guidance scale; unset uses model profile")
	seed := clioptions.Uint64Override(flag.CommandLine, "seed", "noise seed; unset uses model profile")
	weightType := flag.String("weight-type", "", "matmul weights: f32 or bf16; unset uses model profile")
	bf16Attention := clioptions.BoolOverride(flag.CommandLine, "bf16-attention", "enable BF16 attention storage; unset uses model profile")
	deviceOrdinal := flag.Int("device", device.DefaultOrdinal(), "CUDA device ordinal")
	outDirectory := flag.String("out", filepath.Join("build", "latentvideo"), "output directory")
	condPath := flag.String("cond-context", "", "conditional context; required")
	uncondPath := flag.String("uncond-context", "", "unconditional context; required")
	noiseOffset := clioptions.Uint64Override(flag.CommandLine, "noise-offset", "Philox counter offset")
	referencePath := flag.String("reference-latent", "", "reference final latent")
	flag.Parse()
	explicit := clioptions.ExplicitOverrides(flag.CommandLine)
	if flag.NArg() != 1 || *condPath == "" || *uncondPath == "" {
		return fmt.Errorf("usage: latentvideo-run [options] -cond-context FILE -uncond-context FILE <model-directory>")
	}

	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	modelReference, _ := checked.First(flag.Args())
	modelDirectory := roots.ResolveModelPath(modelReference)
	profile, err := latentvideo.ResolveProfile(modelDirectory)
	if err != nil {
		return err
	}
	clioptions.ApplyDefault(explicit, "frames", frames, profile.Generation.Frames)
	clioptions.ApplyDefault(explicit, "width", width, profile.Generation.Width)
	clioptions.ApplyDefault(explicit, "height", height, profile.Generation.Height)
	clioptions.ApplyDefault(explicit, "steps", steps, profile.Generation.Steps)
	clioptions.ApplyDefault(explicit, "shift", shift, profile.Generation.Shift)
	clioptions.ApplyDefault(explicit, "guide", guide, profile.Generation.GuideScale)
	clioptions.ApplyDefault(explicit, "seed", seed, profile.Generation.Seed)
	clioptions.ApplyDefault(explicit, "weight-type", weightType, profile.Precision.MatmulWeights)
	clioptions.ApplyDefault(explicit, "bf16-attention", bf16Attention, profile.Precision.RoundAttentionStorage)
	storage := dtype.BF16
	if *weightType == "f32" {
		storage = dtype.F32
	} else if *weightType != "bf16" {
		return fmt.Errorf("weight-type %q is not f32 or bf16", *weightType)
	}
	started := time.Now()
	generator, err := latentvideo.NewGenerator(latentvideo.GeneratorConfig{
		ModelDirectory: modelDirectory, Policy: profile.Policy, LatentStats: profile.LatentStats,
		Frames: *frames, Width: *width, Height: *height,
		Precision:     latentvideo.DenoiserPrecision{MatmulWeights: storage, RoundAttentionStorage: *bf16Attention},
		DeviceOrdinal: *deviceOrdinal,
	})
	if err != nil {
		return err
	}
	defer generator.Close()
	geometry := generator.Geometry()
	contextElements := generator.ContextElements()
	cond, err := loadContext(*condPath, contextElements)
	if err != nil {
		return err
	}
	uncond, err := loadContext(*uncondPath, contextElements)
	if err != nil {
		return err
	}
	noise, err := noisePlan(*seed, *noiseOffset, geometry.Elements(), *deviceOrdinal)
	if err != nil {
		return err
	}
	if err := clioptions.EnsureOutputDirectory(*outDirectory); err != nil {
		return err
	}
	report := runReport{Request: map[string]any{
		"frames": *frames, "width": *width, "height": *height, "steps": *steps,
		"shift": *shift, "guide_scale": *guide, "seed": *seed,
	}}
	var prior, firstFrame []float32
	result, err := generator.Generate(context.Background(), latentvideo.GenerateRequest{
		Steps: *steps, Shift: *shift, GuideScale: *guide, CondContext: cond, UncondContext: uncond,
		Noise: noise,
		StepHook: func(step int, timestep int64) {
			fmt.Printf("step %02d/%02d t=%d wall=%.1fs\n", step+1, *steps, timestep, time.Since(started).Seconds())
		},
		Sink: func(index int, frame []float32, height, width int) error {
			report.FrameSummaries = append(report.FrameSummaries, summarize(frame))
			if prior != nil {
				var delta float64
				for element := range frame {
					delta += math.Abs(float64(frame[element] - prior[element]))
				}
				report.TemporalDeltas = append(report.TemporalDeltas, delta/float64(len(frame)))
			}
			prior = append(prior[:0], frame...)
			if index == 0 {
				firstFrame = append([]float32(nil), frame...)
			}
			return nil
		},
	})
	if err != nil {
		return err
	}
	report.WallSeconds = time.Since(started).Seconds()
	report.WeightBytes = result.WeightBytes
	decoderWeights := uint64(result.Decode.WeightBytesRead)
	report.PeakBytes = max(result.DenoiseMemory.PeakBytes+decoderWeights, result.DecodeMemory.PeakBytes+result.WeightBytes-decoderWeights)
	report.Execution = map[string]any{
		"graph_launches": result.Execution.GraphLaunches, "graph_instantiations": result.Execution.GraphInstantiations,
		"graph_updates": result.Execution.GraphUpdates, "kernel_launches": result.Execution.KernelLaunches,
		"htod_bytes": result.Execution.HostToDeviceBytes, "dtoh_bytes": result.Execution.DeviceToHostBytes,
	}
	report.FinalLatent = summarize(result.Denoise.Latent)
	report.Decode = map[string]any{"stats": result.Decode, "frames": len(report.FrameSummaries)}
	if *referencePath != "" {
		reference, err := loadContext(*referencePath, geometry.Elements())
		if err != nil {
			return err
		}
		cosine, normalizedRMS, maxDelta := agreement(result.Denoise.Latent, reference)
		report.Reference = map[string]any{"path": *referencePath, "cosine": cosine, "normalized_rms": normalizedRMS, "max_abs_delta": maxDelta}
	}
	if err := clioptions.WriteOutputFile(filepath.Join(*outDirectory, "final_latent.f32le"), driver.Bytes(result.Denoise.Latent)); err != nil {
		return err
	}
	if err := clioptions.WriteOutputFile(filepath.Join(*outDirectory, "frame_00.f32le"), driver.Bytes(firstFrame)); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	if err := clioptions.WriteOutputFile(filepath.Join(*outDirectory, "run_report.json"), raw); err != nil {
		return err
	}
	fmt.Printf("clip frames=%d wall=%.1fs peak=%.2fGiB latent=%s\n", len(report.FrameSummaries), report.WallSeconds, float64(report.PeakBytes)/(1<<30), report.FinalLatent.SHA256)
	return nil
}

func main() {
	clioptions.MainNamed("latentvideo-run", run)
}
