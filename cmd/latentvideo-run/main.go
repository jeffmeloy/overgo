// latentvideo-run: production-shape Wan denoise + decode harness. Runs the
// full guided UniPC trajectory through the CUDA denoiser session (retained
// graph replay), releases denoiser device resources, then decodes the final
// latent (CUDA session by default; host oracle via -decode-engine host).
// Reports wall, device memory, execution counters, and non-degeneracy
// quality evidence; writes the final latent and a JSON summary for
// cross-checking.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"overgo/internal/cuda/driver"
	"overgo/internal/dataroot"
	"overgo/internal/latentvideo"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor/dtype"
)

type g0Config struct {
	Config struct {
		NumTrainTimesteps   int     `json:"num_train_timesteps"`
		SinusoidalPeriod    int     `json:"sinusoidal_period"`
		RotaryFrequencyBase float64 `json:"rotary_frequency_base"`
		VAEStride           [3]int  `json:"vae_stride"`
		TextLen             int     `json:"text_len"`
		Dim                 int     `json:"dim"`
	} `json:"config"`
	VAELatentStats struct {
		Mean []float32 `json:"mean"`
		Std  []float32 `json:"std"`
		ZDim int       `json:"z_dim"`
	} `json:"vae_latent_stats"`
}

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
	Request         map[string]any    `json:"request"`
	NoisePlan       map[string]any    `json:"noise_plan"`
	WeightType      string            `json:"weight_type"`
	WeightBytes     uint64            `json:"weight_bytes"`
	WallSeconds     map[string]any    `json:"wall_seconds"`
	Memory          map[string]any    `json:"memory"`
	Execution       map[string]any    `json:"execution"`
	FinalLatent     tensorSummary     `json:"final_latent"`
	Decode          map[string]any    `json:"decode,omitempty"`
	FrameSummaries  []tensorSummary   `json:"frame_summaries,omitempty"`
	TemporalDeltas  []float64         `json:"temporal_mean_abs_deltas,omitempty"`
	QualityFindings map[string]string `json:"quality_findings"`
}

func summarize(data []float32) tensorSummary {
	summary := tensorSummary{Elements: len(data), Finite: true, Min: math.Inf(1), Max: math.Inf(-1)}
	hash := sha256.New()
	var sum, sumSquares float64
	for _, value := range data {
		v := float64(value)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			summary.Finite = false
		}
		if v < summary.Min {
			summary.Min = v
		}
		if v > summary.Max {
			summary.Max = v
		}
		sum += v
		sumSquares += v * v
	}
	_, _ = hash.Write(driver.Bytes(data))
	summary.SHA256 = hex.EncodeToString(hash.Sum(nil))
	if len(data) > 0 {
		summary.Mean = sum / float64(len(data))
		summary.Std = math.Sqrt(sumSquares/float64(len(data)) - summary.Mean*summary.Mean)
	}
	return summary
}

func loadRawF32(path string, elements int) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) != 4*elements {
		return nil, fmt.Errorf("%s: %d bytes, want %d", path, len(raw), 4*elements)
	}
	out := make([]float32, elements)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24)
	}
	return out, nil
}

// loadContext: raw f32le, or the capture engine's .wanctx (16-byte header).
func loadContext(path string, elements int) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	const wanctxHeader = 16
	if len(raw) == 4*elements+wanctxHeader && string(raw[:7]) == "WANCTX1" {
		raw = raw[wanctxHeader:]
	}
	if len(raw) != 4*elements {
		return nil, fmt.Errorf("%s: %d bytes, want %d", path, len(raw), 4*elements)
	}
	out := make([]float32, elements)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24)
	}
	return out, nil
}

func deriveNoiseGrid(elements int, ordinal int) (int, error) {
	lib, err := driver.Open()
	if err != nil {
		return 0, err
	}
	defer lib.Close()
	if err := lib.Init(); err != nil {
		return 0, err
	}
	info, err := lib.DeviceInfo(ordinal)
	if err != nil {
		return 0, err
	}
	// torch.cuda distribution launch: block 256, unroll 4, grid capped at
	// SMs x (maxThreadsPerSM/block); 1536 threads/SM on sm_86/89.
	const blocksPerSM = 1536 / 256
	grid := (elements + 1023) / 1024
	if cap := info.MultiprocessorCount * blocksPerSM; grid > cap {
		grid = cap
	}
	if grid < 1 {
		grid = 1
	}
	return grid, nil
}

func run() error {
	frames := flag.Int("frames", 81, "output frame count")
	width := flag.Int("width", 832, "output width")
	height := flag.Int("height", 480, "output height")
	steps := flag.Int("steps", 50, "denoise steps")
	shift := flag.Float64("shift", 5, "UniPC flow shift")
	guide := flag.Float64("guide", 6, "guidance scale")
	seed := flag.Int64("seed", 31, "philox noise seed")
	weightType := flag.String("weight-type", "f32", "matmul weight storage: f32 or bf16")
	bf16Attention := flag.Bool("bf16-attention", false, "round attention q/k/v through BF16 storage (tensor-core flash path)")
	deviceOrdinal := flag.Int("device", 0, "CUDA device ordinal")
	decode := flag.Bool("decode", true, "run the causal VAE decode")
	decodeEngine := flag.String("decode-engine", "cuda", "vae decode engine: cuda (session) or host (oracle)")
	decodeFrames := flag.Int("decode-frames", 0, "latent frames to decode (0 = all; host decode is slow at production scale)")
	outDir := flag.String("out", filepath.Join("build", "latentvideo"), "output directory")
	fixtureDir := flag.String("fixtures", filepath.Join("fixtures", "wan"), "wan fixture directory")
	condPath := flag.String("cond-context", "", "conditional context override (.f32le or 16-byte-header .wanctx)")
	uncondPath := flag.String("uncond-context", "", "unconditional context override")
	noiseOffset := flag.Uint64("noise-offset", 0, "philox counter offset (4-aligned)")
	referenceLatent := flag.String("reference-latent", "", "reference final latent (.f32le) for cosine cross-check")
	flag.Parse()

	working, err := os.Getwd()
	if err != nil {
		return err
	}
	roots, err := dataroot.Resolve(working)
	if err != nil {
		return err
	}
	modelDir := filepath.Join(roots.Models, "Wan2.1-T2V-1.3B")
	if _, err := os.Stat(modelDir); err != nil {
		return fmt.Errorf("model dir absent: %w", err)
	}
	g0Raw, err := os.ReadFile(filepath.Join(*fixtureDir, "g0_config.json"))
	if err != nil {
		return err
	}
	var g0 g0Config
	if err := json.Unmarshal(g0Raw, &g0); err != nil {
		return err
	}
	policy := latentvideo.DenoiserPolicy{
		NumTrainTimesteps:   g0.Config.NumTrainTimesteps,
		SinusoidalPeriod:    g0.Config.SinusoidalPeriod,
		RotaryFrequencyBase: g0.Config.RotaryFrequencyBase,
		VAEStride:           g0.Config.VAEStride,
	}
	storage := dtype.F32
	if *weightType == "bf16" {
		storage = dtype.BF16
	} else if *weightType != "f32" {
		return fmt.Errorf("weight-type %q is not f32 or bf16", *weightType)
	}

	fmt.Printf("[%s] loading denoiser config + weights from %s\n", time.Now().Format(time.TimeOnly), modelDir)
	loadStart := time.Now()
	config, err := latentvideo.LoadDenoiserConfig(modelDir, policy)
	if err != nil {
		return err
	}
	weights, err := latentvideo.LoadDenoiserWeights(modelDir, config)
	if err != nil {
		return err
	}
	geometry, err := config.CompileLatentGeometry(*frames, *width, *height)
	if err != nil {
		return err
	}
	program, err := latentvideo.CompileDenoiserProgramPrecision(config, weights, geometry, latentvideo.DenoiserPrecision{
		MatmulWeights:         storage,
		RoundAttentionStorage: *bf16Attention,
	})
	if err != nil {
		return err
	}
	loadWall := time.Since(loadStart)
	fmt.Printf("[%s] weights loaded in %.1fs; latent %dx%dx%dx%d seq=%d\n",
		time.Now().Format(time.TimeOnly), loadWall.Seconds(),
		geometry.Channels, geometry.LatentFrames, geometry.LatentHeight, geometry.LatentWidth, geometry.Seq)

	contextElements := g0.Config.TextLen * g0.Config.Dim
	condFile := filepath.Join(*fixtureDir, "raw", "g1_cond_context.f32le")
	uncondFile := filepath.Join(*fixtureDir, "raw", "g1_uncond_context.f32le")
	if *condPath != "" {
		condFile = *condPath
	}
	if *uncondPath != "" {
		uncondFile = *uncondPath
	}
	condContext, err := loadContext(condFile, contextElements)
	if err != nil {
		return err
	}
	uncondContext, err := loadContext(uncondFile, contextElements)
	if err != nil {
		return err
	}

	grid, err := deriveNoiseGrid(geometry.Elements(), *deviceOrdinal)
	if err != nil {
		return err
	}
	noise := latentvideo.NoisePlan{Seed: uint64(*seed), Offset: *noiseOffset, Grid: grid, Block: 256, Unroll: 4}
	fmt.Printf("[%s] noise plan seed=%d offset=%d grid=%d block=256 unroll=4\n", time.Now().Format(time.TimeOnly), *seed, *noiseOffset, grid)

	sessionStart := time.Now()
	session, err := latentvideo.NewDenoiserCUDASession(program, *deviceOrdinal)
	if err != nil {
		return err
	}
	defer session.Close()
	sessionWall := time.Since(sessionStart)
	fmt.Printf("[%s] session ready in %.1fs (weight upload %d bytes, %s storage)\n",
		time.Now().Format(time.TimeOnly), sessionWall.Seconds(), session.WeightBytes, *weightType)

	denoiseStart := time.Now()
	lastStep := denoiseStart
	result, err := session.Denoise(context.Background(), latentvideo.DenoiseRequest{
		Steps: *steps, Shift: *shift, GuideScale: *guide,
		CondContext: condContext, UncondContext: uncondContext,
		Noise: noise,
		StepHook: func(step int, timestep int64) {
			now := time.Now()
			fmt.Printf("[%s] step %02d/%02d t=%d wall=%.2fs total=%.1fs\n",
				now.Format(time.TimeOnly), step+1, *steps, timestep,
				now.Sub(lastStep).Seconds(), now.Sub(denoiseStart).Seconds())
			lastStep = now
		},
	})
	if err != nil {
		return err
	}
	denoiseWall := time.Since(denoiseStart)
	memoryPeak, err := session.MemoryStats()
	if err != nil {
		return err
	}
	execution, err := session.ExecutionStats()
	if err != nil {
		return err
	}
	fmt.Printf("[%s] denoise done in %.1fs; device peak_bytes=%d current_bytes=%d\n",
		time.Now().Format(time.TimeOnly), denoiseWall.Seconds(), memoryPeak.PeakBytes, memoryPeak.CurrentBytes)
	fmt.Printf("graph_launches=%d instantiations=%d updates=%d kernel_launches=%d htod_bytes=%d\n",
		execution.GraphLaunches, execution.GraphInstantiations, execution.GraphUpdates,
		execution.KernelLaunches, execution.HostToDeviceBytes)

	// Pre-decode lifetime release: denoiser device resources close before
	// VAE admission.
	if err := session.ReleaseDenoiseResources(); err != nil {
		return err
	}
	memoryReleased, err := session.MemoryStats()
	if err != nil {
		return err
	}
	fmt.Printf("[%s] pre-decode release: current_bytes=%d (peak stays %d)\n",
		time.Now().Format(time.TimeOnly), memoryReleased.CurrentBytes, memoryReleased.PeakBytes)

	latentSummary := summarize(result.Latent)
	fmt.Printf("final_latent sha256=%s min=%.6f max=%.6f mean=%.6f std=%.6f finite=%v\n",
		latentSummary.SHA256, latentSummary.Min, latentSummary.Max, latentSummary.Mean, latentSummary.Std, latentSummary.Finite)

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*outDir, "final_latent.f32le"), driver.Bytes(result.Latent), 0o644); err != nil {
		return err
	}

	report := runReport{
		Request: map[string]any{
			"prompt_context": "fixtures g1 (fox prompt, umt5 conditioning)",
			"frames":         *frames, "width": *width, "height": *height,
			"steps": *steps, "shift": *shift, "guide_scale": *guide, "seed": *seed,
		},
		NoisePlan:   map[string]any{"seed": *seed, "grid": grid, "block": 256, "unroll": 4},
		WeightType:  fmt.Sprintf("%s(attention_bf16=%v)", *weightType, *bf16Attention),
		WeightBytes: session.WeightBytes,
		WallSeconds: map[string]any{
			"weights_load": loadWall.Seconds(),
			"session_up":   sessionWall.Seconds(),
			"denoise":      denoiseWall.Seconds(),
		},
		Memory: map[string]any{
			"peak_owned_bytes":        memoryPeak.PeakBytes,
			"post_release_bytes":      memoryReleased.CurrentBytes,
			"denoise_end_owned_bytes": memoryPeak.CurrentBytes,
		},
		Execution: map[string]any{
			"graph_launches":       execution.GraphLaunches,
			"graph_instantiations": execution.GraphInstantiations,
			"graph_updates":        execution.GraphUpdates,
			"kernel_launches":      execution.KernelLaunches,
			"htod_bytes":           execution.HostToDeviceBytes,
			"dtoh_bytes":           execution.DeviceToHostBytes,
		},
		FinalLatent:     latentSummary,
		QualityFindings: map[string]string{},
	}
	if latentSummary.Finite {
		report.QualityFindings["latent_finite"] = "pass"
	} else {
		report.QualityFindings["latent_finite"] = "FAIL"
	}
	if latentSummary.Std > 0.01 {
		report.QualityFindings["latent_nonconstant"] = "pass"
	} else {
		report.QualityFindings["latent_nonconstant"] = "FAIL"
	}

	if *referenceLatent != "" {
		reference, err := loadRawF32(*referenceLatent, geometry.Elements())
		if err != nil {
			return err
		}
		var dot, gotNorm, refNorm, deltaSquares, maxDelta float64
		for i := range reference {
			g, r := float64(result.Latent[i]), float64(reference[i])
			dot += g * r
			gotNorm += g * g
			refNorm += r * r
			delta := g - r
			deltaSquares += delta * delta
			if math.Abs(delta) > maxDelta {
				maxDelta = math.Abs(delta)
			}
		}
		cosine := dot / math.Sqrt(gotNorm*refNorm)
		normalizedRMS := math.Sqrt(deltaSquares / refNorm)
		report.QualityFindings["reference_latent_cosine"] = fmt.Sprintf("%.9f", cosine)
		report.QualityFindings["reference_latent_normalized_rms"] = fmt.Sprintf("%.6g", normalizedRMS)
		report.QualityFindings["reference_latent_max_abs_delta"] = fmt.Sprintf("%.6g", maxDelta)
		report.QualityFindings["reference_latent_path"] = *referenceLatent
		fmt.Printf("reference latent cosine=%.9f normalized_rms=%.6g max_abs_delta=%.6g\n", cosine, normalizedRMS, maxDelta)
	}

	if *decode {
		decodeStart := time.Now()
		checkpoint := filepath.Join(modelDir, "Wan2.1_VAE.pth")
		metas, err := pytorchzip.ReadTensorMetadata(checkpoint)
		if err != nil {
			return err
		}
		plan, err := latentvideo.CompileVAEDecoderPlan(metas)
		if err != nil {
			return err
		}
		stats := latentvideo.VAELatentStats{Mean: g0.VAELatentStats.Mean, Std: g0.VAELatentStats.Std}
		decodeLatent := result.Latent
		decodeLatentFrames := geometry.LatentFrames
		if *decodeFrames > 0 && *decodeFrames < geometry.LatentFrames {
			// Truncate channel-major [z][frames][spatial] to the leading frames.
			spatial := geometry.LatentHeight * geometry.LatentWidth
			decodeLatentFrames = *decodeFrames
			decodeLatent = make([]float32, geometry.Channels*decodeLatentFrames*spatial)
			for channel := 0; channel < geometry.Channels; channel++ {
				source := result.Latent[channel*geometry.LatentFrames*spatial:]
				copy(decodeLatent[channel*decodeLatentFrames*spatial:(channel+1)*decodeLatentFrames*spatial],
					source[:decodeLatentFrames*spatial])
			}
			fmt.Printf("[%s] partial decode: leading %d of %d latent frames\n",
				time.Now().Format(time.TimeOnly), decodeLatentFrames, geometry.LatentFrames)
		}
		var previous []float32
		var firstFrame []float32
		frameIndex := 0
		sink := func(index int, frame []float32, frameHeight, frameWidth int) error {
			summary := summarize(frame)
			report.FrameSummaries = append(report.FrameSummaries, summary)
			if previous != nil {
				var delta float64
				for i := range frame {
					delta += math.Abs(float64(frame[i]) - float64(previous[i]))
				}
				report.TemporalDeltas = append(report.TemporalDeltas, delta/float64(len(frame)))
			}
			previous = append(previous[:0], frame...)
			if index == 0 {
				firstFrame = append([]float32(nil), frame...)
			}
			frameIndex++
			if frameIndex%10 == 0 {
				fmt.Printf("[%s] decoded frame %d (%dx%d) wall=%.1fs\n",
					time.Now().Format(time.TimeOnly), frameIndex, frameWidth, frameHeight, time.Since(decodeStart).Seconds())
			}
			return nil
		}
		var decodeStats latentvideo.VAEDecodeStats
		switch *decodeEngine {
		case "cuda":
			session, sessionErr := latentvideo.NewVAEDecoderCUDASession(checkpoint, plan, *deviceOrdinal)
			if sessionErr != nil {
				return sessionErr
			}
			decodeStats, err = session.Decode(stats, decodeLatent,
				decodeLatentFrames, geometry.LatentHeight, geometry.LatentWidth, sink)
			err = errors.Join(err, session.Close())
		case "host":
			decodeStats, err = latentvideo.DecodeLatentVideo(
				checkpoint, plan, stats, decodeLatent,
				decodeLatentFrames, geometry.LatentHeight, geometry.LatentWidth, sink,
			)
		default:
			err = fmt.Errorf("unknown decode engine %q", *decodeEngine)
		}
		if err != nil {
			return err
		}
		decodeWall := time.Since(decodeStart)
		report.WallSeconds["decode"] = decodeWall.Seconds()
		report.Decode = map[string]any{
			"frames": frameIndex, "stats": decodeStats,
		}
		fmt.Printf("[%s] decode done: %d frames in %.1fs\n", time.Now().Format(time.TimeOnly), frameIndex, decodeWall.Seconds())
		if firstFrame != nil {
			if err := os.WriteFile(filepath.Join(*outDir, "frame_00.f32le"), driver.Bytes(firstFrame), 0o644); err != nil {
				return err
			}
		}
		finiteFrames, constantFrames := true, 0
		for _, summary := range report.FrameSummaries {
			finiteFrames = finiteFrames && summary.Finite
			if summary.Std < 1e-4 {
				constantFrames++
			}
		}
		movingPairs := 0
		for _, delta := range report.TemporalDeltas {
			if delta > 1e-4 {
				movingPairs++
			}
		}
		report.QualityFindings["frames_finite"] = passFail(finiteFrames)
		report.QualityFindings["frames_nonconstant"] = passFail(constantFrames == 0)
		report.QualityFindings["temporal_variation"] = fmt.Sprintf("%d/%d changing pairs", movingPairs, len(report.TemporalDeltas))
	}

	raw, err := json.MarshalIndent(report, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*outDir, "run_report.json"), raw, 0o644); err != nil {
		return err
	}
	fmt.Printf("[%s] report written to %s\n", time.Now().Format(time.TimeOnly), filepath.Join(*outDir, "run_report.json"))
	return nil
}

func passFail(ok bool) string {
	if ok {
		return "pass"
	}
	return "FAIL"
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "latentvideo-run:", err)
		os.Exit(1)
	}
}
