// Command dit-train-probe executes full video-diffusion (DiT) training as a
// director-supervised model-session: the trainable surface is EVERY tensor
// of the conditioned diffusion transformer (all 30 blocks, patch embedding,
// text/time embeddings, time projection, modulated head) as f32 masters over
// the shared Muon stepper with fully derived hyperparameters, the frozen
// text encoder and VAE sit outside the surface, and the objective is real
// flow matching — MSE against v = noise - x0 on the committed shifted
// schedule. Conditioning is real: raw umt5-xxl rows for the committed g1
// prompt through the streamed host encoder, x0 seeded from the committed
// sha-pinned g3/g4 final latents, noise from the golden Philox plan. The run
// publishes one typed session observation. With -checkpoint it trains the
// reference-edit (.pt) variant whose 32-channel patch embedding concatenates
// a real source latent (gradient stops at the source channels). This is the
// full video-diffusion training run the organ cells named as their
// promotion gate.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/latentvideo"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
	"overgo/internal/trainingworkflow"
)

func main() {
	clioptions.MainNamed("dit-train-probe", run)
}

// fixtureTensor: one sha256-pinned committed f32le fixture.
type fixtureTensor struct {
	File     string `json:"file"`
	Shape    []int  `json:"shape"`
	Elements int    `json:"elements"`
	SHA256   string `json:"sha256"`
}

func loadFixtureTensor(dir string, spec fixtureTensor) ([]float32, error) {
	raw, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(spec.File)))
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != spec.SHA256 {
		return nil, fmt.Errorf("fixture %s sha256=%s want %s", spec.File, got, spec.SHA256)
	}
	if len(raw) != spec.Elements*4 {
		return nil, fmt.Errorf("fixture %s bytes=%d want %d", spec.File, len(raw), spec.Elements*4)
	}
	out := make([]float32, spec.Elements)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}

// finalLatentFixture: the committed real denoised latent of a golden run.
func finalLatentFixture(fixturesDir, manifest string) ([]float32, [4]int, error) {
	var parsed struct {
		FinalLatent fixtureTensor `json:"final_latent"`
	}
	if err := jsonfile.Decode(filepath.Join(fixturesDir, manifest), &parsed); err != nil {
		return nil, [4]int{}, err
	}
	if len(parsed.FinalLatent.Shape) != 4 {
		return nil, [4]int{}, fmt.Errorf("fixture %s final latent shape %v", manifest, parsed.FinalLatent.Shape)
	}
	values, err := loadFixtureTensor(fixturesDir, parsed.FinalLatent)
	if err != nil {
		return nil, [4]int{}, err
	}
	var shape [4]int
	copy(shape[:], parsed.FinalLatent.Shape)
	return values, shape, nil
}

// tileLatent: replicate a real channel-major seed grid across the training
// grid (dst[c][f][y][x] = src[c][f%sf][y%sh][x%sw]).
func tileLatent(src []float32, srcShape [4]int, frames, height, width int) []float32 {
	channels, sf, sh, sw := srcShape[0], srcShape[1], srcShape[2], srcShape[3]
	dst := make([]float32, channels*frames*height*width)
	for c := 0; c < channels; c++ {
		for f := 0; f < frames; f++ {
			for y := 0; y < height; y++ {
				for x := 0; x < width; x++ {
					dst[((c*frames+f)*height+y)*width+x] = src[((c*sf+f%sf)*sh+y%sh)*sw+x%sw]
				}
			}
		}
	}
	return dst
}

// latentFrameZero: slice frame 0 of a channel-major [c, f, h, w] latent.
func latentFrameZero(src []float32, shape [4]int) ([]float32, [4]int) {
	channels, frames, height, width := shape[0], shape[1], shape[2], shape[3]
	plane := height * width
	out := make([]float32, channels*plane)
	for c := 0; c < channels; c++ {
		copy(out[c*plane:(c+1)*plane], src[c*frames*plane:c*frames*plane+plane])
	}
	return out, [4]int{channels, 1, height, width}
}

type rawTextCache struct {
	Prompt  string `json:"prompt"`
	Tokens  int    `json:"tokens"`
	TextDim int    `json:"text_dim"`
	SHA256  string `json:"sha256"`
}

// rawTextRows: real streamed-encoder rows for the prompt, with an optional
// deterministic on-disk cache (content-addressed; recomputed on mismatch).
func rawTextRows(spec latentvideo.TextConditioningSpec, prompt, cacheBase string) ([]float32, int, int, error) {
	if cacheBase != "" {
		var meta rawTextCache
		if err := jsonfile.Decode(cacheBase+".json", &meta); err == nil && meta.Prompt == prompt {
			raw, err := os.ReadFile(cacheBase + ".f32le")
			if err == nil {
				sum := sha256.Sum256(raw)
				if hex.EncodeToString(sum[:]) == meta.SHA256 && len(raw) == meta.Tokens*meta.TextDim*4 {
					rows := make([]float32, meta.Tokens*meta.TextDim)
					for i := range rows {
						rows[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
					}
					fmt.Printf("raw text rows: cache hit (%d tokens x %d)\n", meta.Tokens, meta.TextDim)
					return rows, meta.Tokens, meta.TextDim, nil
				}
			}
		}
	}
	started := time.Now()
	rows, tokens, textDim, err := latentvideo.RawTextRows(spec, prompt)
	if err != nil {
		return nil, 0, 0, err
	}
	fmt.Printf("raw text rows: streamed encoder produced %d tokens x %d in %s\n", tokens, textDim, time.Since(started).Round(time.Second))
	if cacheBase != "" {
		raw := make([]byte, len(rows)*4)
		for i, v := range rows {
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
		}
		sum := sha256.Sum256(raw)
		if err := os.WriteFile(cacheBase+".f32le", raw, 0o644); err != nil {
			return nil, 0, 0, err
		}
		if err := jsonfile.Write(cacheBase+".json", rawTextCache{
			Prompt: prompt, Tokens: tokens, TextDim: textDim, SHA256: hex.EncodeToString(sum[:]),
		}, 0o644); err != nil {
			return nil, 0, 0, err
		}
	}
	return rows, tokens, textDim, nil
}

func contextDiff(got, want []float32) (maxDiff, meanDiff float64) {
	var sum float64
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		sum += d
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff, sum / float64(len(want))
}

func run() error {
	modelDir := flag.String("model-dir", `C:\Users\jeffm\adaptive_new\models\Wan2.1-T2V-1.3B`, "Wan checkpoint dir (config, safetensors, tokenizer, text encoder)")
	checkpoint := flag.String("checkpoint", "", "reference-edit .pt checkpoint (empty: train the safetensors DiT)")
	fixturesDir := flag.String("fixtures", "fixtures/wan", "committed golden fixture dir (real conditioning and latent seeds)")
	frames := flag.Int("frames", 1, "requested video frames")
	width := flag.Int("width", 256, "requested video width")
	height := flag.Int("height", 256, "requested video height")
	scheduleIndex := flag.Int("schedule-index", 1, "index into the 4-step shift-5 committed schedule")
	steps := flag.Int("steps", 3, "observed training steps")
	maxWall := flag.Duration("max-wall", 90*time.Minute, "abort when the first step projects past this bound")
	storePath := flag.String("store", "repodb-store", "RepoDB for the session observation and recipe authority")
	stimulus := flag.String("stimulus", "docs/verification/wan-dit-training-stimulus.txt", "committed stimulus declaration grounding the recipe dataset")
	t5Cache := flag.String("t5-cache", "", "optional cache base path for the streamed raw text rows")
	flag.Parse()
	ctx := context.Background()

	store, err := repodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()

	// Recipe authority + model identity, then session admission BEFORE any
	// weight touches memory.
	weightsPath := filepath.Join(*modelDir, "diffusion_pytorch_model.safetensors")
	if *checkpoint != "" {
		weightsPath = *checkpoint
	}
	modelFile, err := os.Open(weightsPath)
	if err != nil {
		return err
	}
	modelID, _, err := artifact.Identify(artifact.KindModel, modelFile)
	modelFile.Close()
	if err != nil {
		return err
	}
	recipeID, err := trainingworkflow.BootstrapTokenRecipe(ctx, store, weightsPath, *stimulus)
	if err != nil {
		return err
	}
	observer, err := trainingworkflow.NewObserver(store, false)
	if err != nil {
		return err
	}
	if err := observer.Admit(ctx, modelID, recipeID); err != nil {
		return err
	}
	fmt.Printf("session admitted: model=%s recipe=%s\n", modelID, recipeID)

	loadStarted := time.Now()
	profile, err := latentvideo.ResolveProfile(*modelDir)
	if err != nil {
		return err
	}
	config, err := latentvideo.LoadDenoiserConfig(*modelDir, profile.Policy)
	if err != nil {
		return err
	}

	// Real raw text rows for the committed g1 prompt (frozen encoder,
	// outside the trainable surface).
	var g1 struct {
		Prompt      string `json:"prompt"`
		Conditional struct {
			Tensor fixtureTensor `json:"tensor"`
		} `json:"conditional"`
	}
	if err := jsonfile.Decode(filepath.Join(*fixturesDir, "g1_text_conditioning.json"), &g1); err != nil {
		return err
	}
	// RelativeMaxDistance/NormEps: published encoder-config facts the
	// checkpoint cannot carry (the same policy the g1 capture used).
	textSpec := latentvideo.TextConditioningSpec{
		TokenizerDir:        filepath.Join(*modelDir, "google", "umt5-xxl"),
		EncoderCheckpoint:   filepath.Join(*modelDir, "models_t5_umt5-xxl-enc-bf16.pth"),
		ProjectionDir:       *modelDir,
		SequenceLength:      config.TextLen,
		RelativeMaxDistance: 128,
		NormEps:             1e-6,
	}
	rawText, textTokens, rawTextDim, err := rawTextRows(textSpec, g1.Prompt, *t5Cache)
	if err != nil {
		return err
	}

	// Trainable tensor set: F32 safetensors directly, or the compiled
	// reference-edit checkpoint (BF16/F32 torch storage decoded to f32).
	cfg := config
	var tensors map[string][]float32
	var textDim int
	if *checkpoint != "" {
		compiled, err := latentvideo.CompileReferenceEditCheckpoint(*checkpoint, config)
		if err != nil {
			return err
		}
		cfg = compiled.Config
		for _, name := range []string{"patch_embedding.weight", "blocks.0.self_attn.q.weight", "text_embedding.0.weight"} {
			if binding, ok := compiled.Binding(name); ok {
				fmt.Printf("checkpoint storage: %s dtype=%s\n", name, binding.Meta.DType)
			}
		}
		if tensors, textDim, err = compiled.LoadTrainerTensors(); err != nil {
			return err
		}
	} else if tensors, textDim, err = latentvideo.LoadDiTTrainerTensors(*modelDir, config); err != nil {
		return err
	}
	if textDim != rawTextDim {
		return fmt.Errorf("dit train probe: text projection width %d != encoder width %d", textDim, rawTextDim)
	}
	geometry, err := cfg.CompileLatentGeometry(*frames, *width, *height)
	if err != nil {
		return err
	}
	trainer, err := latentvideo.NewDiTTrainer(cfg, textDim, geometry, tensors)
	if err != nil {
		return err
	}
	defer trainer.Close()
	observer.Phase(runrecord.PhaseLoad, time.Since(loadStarted))
	fmt.Printf("full DiT: layers=%d trainable_parameters=%d seq=%d latent=%dx%dx%dx%d text_tokens=%d\n",
		cfg.NumLayers, trainer.ParameterCount(), geometry.Seq,
		geometry.Channels, geometry.LatentFrames, geometry.LatentHeight, geometry.LatentWidth, textTokens)
	fmt.Printf("derived_lr=%.6g derived_momentum=%.10g load_wall=%s\n",
		trainer.Config().BaseLearningRate, trainer.Config().Momentum, time.Since(loadStarted).Round(time.Millisecond))

	// Evidence: the trainer's own unrounded projection of the raw rows vs
	// the committed (bf16-disciplined) g1 projected context.
	if g1Context, err := loadFixtureTensor(*fixturesDir, g1.Conditional.Tensor); err == nil {
		if projected, err := trainer.ProjectTextContext(rawText, textTokens); err == nil {
			maxDiff, meanDiff := contextDiff(projected, g1Context)
			fmt.Printf("text projection vs committed g1 context: max_abs_diff=%.6g mean_abs_diff=%.6g (bf16 serving rounding expected)\n", maxDiff, meanDiff)
		}
	}

	// Real flow-matching batch: x0 from committed sha-pinned real latents,
	// golden Philox noise, the committed shifted schedule.
	timesteps, sigmas, err := sampling.UniPCSchedule(cfg.Policy.NumTrainTimesteps, 4, 5)
	if err != nil {
		return err
	}
	if *scheduleIndex < 0 || *scheduleIndex >= len(timesteps) {
		return fmt.Errorf("dit train probe: schedule index %d outside %d timesteps", *scheduleIndex, len(timesteps))
	}
	timestep := timesteps[*scheduleIndex]
	sigma := float64(sigmas[*scheduleIndex])

	g3Latent, g3Shape, err := finalLatentFixture(*fixturesDir, "g3_denoise.json")
	if err != nil {
		return err
	}
	x0Seed, x0Shape := g3Latent, g3Shape
	var source []float32
	if *checkpoint != "" {
		g4Latent, g4Shape, err := finalLatentFixture(*fixturesDir, "g4_denoise.json")
		if err != nil {
			return err
		}
		x0Seed, x0Shape = latentFrameZero(g4Latent, g4Shape)
		source = tileLatent(g3Latent, g3Shape, geometry.LatentFrames, geometry.LatentHeight, geometry.LatentWidth)
	}
	x0 := tileLatent(x0Seed, x0Shape, geometry.LatentFrames, geometry.LatentHeight, geometry.LatentWidth)
	noiseChannels := cfg.OutDim
	noiseElements := noiseChannels * geometry.LatentFrames * geometry.LatentHeight * geometry.LatentWidth
	if len(x0) != noiseElements {
		return fmt.Errorf("dit train probe: x0 has %d elements, need %d", len(x0), noiseElements)
	}
	noise := make([]float32, noiseElements)
	// The golden capture engine's noise plan (g3 seed 31 geometry).
	if err := sampling.FillCounterNormalNoise(noise, sampling.CounterNoisePlan{Seed: 31, Grid: 1, Block: 256, Unroll: 4}); err != nil {
		return err
	}
	latent := make([]float32, geometry.Elements())
	target := make([]float32, noiseElements)
	for i := range x0 {
		latent[i] = float32((1-sigma)*float64(x0[i]) + sigma*float64(noise[i]))
		target[i] = noise[i] - x0[i]
	}
	if source != nil {
		copy(latent[noiseElements:], source)
	}
	batch := latentvideo.DiTTrainBatch{
		Latent: latent, RawText: rawText, TextTokens: textTokens,
		Timestep: float64(timestep), Target: target,
	}
	fmt.Printf("flow matching: timestep=%d sigma=%.8f (shift-5 schedule index %d)\n", timestep, sigma, *scheduleIndex)

	initial, err := trainer.Loss(batch)
	if err != nil {
		return err
	}
	fmt.Printf("initial loss %.6f (flow-matching MSE over %d latent elements)\n", initial, noiseElements)

	trainStarted := time.Now()
	runErr := func() error {
		for step := 0; step < *steps; step++ {
			stepStarted := time.Now()
			result, err := trainer.Step(batch)
			if err != nil {
				return err
			}
			observer.SampleStep()
			wall := time.Since(stepStarted)
			fmt.Printf("step %d/%d loss %.6f grad_l2 %.6f lr %.6g wall %s\n",
				result.Step, *steps, result.Loss, result.GradientL2, result.LearningRate, wall.Round(time.Millisecond))
			if result.GradientL2 <= 0 {
				return fmt.Errorf("dit train probe: step %d gradient norm %g", result.Step, result.GradientL2)
			}
			if step == 0 && *maxWall > 0 {
				projected := wall * time.Duration(*steps+1)
				if projected > *maxWall {
					return fmt.Errorf("dit train probe: first step %s projects %d steps to %s, over %s", wall.Round(time.Second), *steps, projected.Round(time.Minute), *maxWall)
				}
			}
		}
		final, err := trainer.Loss(batch)
		if err != nil {
			return err
		}
		fmt.Printf("after-evaluation loss %.6f (initial %.6f)\n", final, initial)
		if !(final < initial) {
			return fmt.Errorf("dit train probe: no descent: %.6f -> %.6f", initial, final)
		}
		return nil
	}()
	observer.Phase(runrecord.PhaseForwardBackward, time.Since(trainStarted))
	streamed := uint64(*steps) * uint64(geometry.Seq)
	observation, observeErr := observer.Finish(ctx, modelID, recipeID, runErr, streamed)
	if runErr != nil {
		return runErr
	}
	if observeErr != nil {
		return observeErr
	}
	fmt.Printf("session observation: %s\n", observation)
	return nil
}
