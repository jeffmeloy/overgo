//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/hfbpe"
	"overgo/internal/modelrecipe"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
	"overgo/internal/torchrng"
	"overgo/internal/workflowruntime"
)

func readEmbedRowsF32(modelDir string, spec TextEncoderSpec, ids []int) ([]float32, error) {
	source, err := safetensors.OpenSource(filepath.Join(modelDir, "text_encoder"))
	if err != nil {
		return nil, fmt.Errorf("resident encoder embed: %w", err)
	}
	defer source.Close()
	rows, err := readEmbedRows(source, spec, ids)
	if err != nil {
		return nil, err
	}
	result := make([]float32, len(rows))
	for index, value := range rows {
		result[index] = float32(value)
	}
	return result, nil
}

const (
	DefaultSteps             = 8
	DefaultNumTrainTimesteps = 1000
	DefaultDynamicShiftMu    = 1.15
)

var generatedImageContract = artifact.JSONContract(artifact.KindOutput, "overgo.generated-image.v1")

type Request struct {
	Prompt            string  `json:"prompt"`
	Width             int     `json:"width"`
	Height            int     `json:"height"`
	Steps             int     `json:"steps,omitempty"`
	Seed              int64   `json:"seed"`
	DynamicShiftMu    float64 `json:"mu,omitempty"`
	NumTrainTimesteps int     `json:"num_train_timesteps,omitempty"`
}

type Image struct {
	Pixels   []float32 `json:"pixels"`
	Channels int       `json:"channels"`
	Height   int       `json:"height"`
	Width    int       `json:"width"`
}

type Generator struct {
	request    Request
	spec       *Spec
	pipeline   *ResidentImagePipeline
	embed      []float32
	schedule   FlowSchedule
	shape      LatentShape
	prepared   bool
	integrated bool
}

func ValidateRequest(request Request) error {
	if request.Prompt == "" {
		return errors.New("latent image: prompt is empty")
	}
	if request.Width <= 0 || request.Height <= 0 {
		return fmt.Errorf("latent image: invalid extent %dx%d", request.Width, request.Height)
	}
	if request.Steps < 0 || request.NumTrainTimesteps < 0 || math.IsNaN(request.DynamicShiftMu) || math.IsInf(request.DynamicShiftMu, 0) {
		return errors.New("latent image: invalid sampling policy")
	}
	return nil
}

func LoadGenerator(ctx context.Context, modelDir string, request Request) (*Generator, error) {
	if err := ValidateRequest(request); err != nil {
		return nil, err
	}
	request = request.withDefaults()
	spec, err := Derive(modelDir)
	if err != nil {
		return nil, err
	}
	if _, err := spec.VerifyCheckpoint(modelDir); err != nil {
		return nil, err
	}
	shape, err := spec.LatentShape(request.Height, request.Width)
	if err != nil {
		return nil, err
	}
	if spec.PatchSize <= 0 || shape.Height%spec.PatchSize != 0 || shape.Width%spec.PatchSize != 0 {
		return nil, fmt.Errorf("latent image: latent extent %dx%d is incompatible with patch %d", shape.Width, shape.Height, spec.PatchSize)
	}
	tokenizer, err := hfbpe.Load(filepath.Join(modelDir, "tokenizer"))
	if err != nil {
		return nil, err
	}
	text, err := RenderKreaTextInput(tokenizer, request.Prompt, KreaChatPromptTemplate())
	if err != nil {
		return nil, err
	}
	embed, err := readEmbedRowsF32(modelDir, spec.TextEncoder, text.IDs)
	if err != nil {
		return nil, err
	}
	encoder, err := CompileEncoderProgramMasked(
		spec.TextEncoder, float32(spec.TextEncoder.RMSNormEps), len(text.IDs), dtype.BF16, text.Mask,
	)
	if err != nil {
		return nil, err
	}
	textMask := text.Mask[len(text.Mask)-text.PromptRows:]
	fusion, err := CompileFusionProgram(
		spec.Transformer, float32(spec.Transformer.NormEps), textMask, dtype.BF16,
	)
	if err != nil {
		return nil, err
	}
	gridH, gridW := shape.Height/spec.PatchSize, shape.Width/spec.PatchSize
	denoiser, err := CompileDenoiserProgram(
		spec.Transformer, float32(spec.Transformer.NormEps), textMask, gridH, gridW, dtype.BF16,
	)
	if err != nil {
		return nil, err
	}
	decoder, err := LoadVAEDecoder(modelDir)
	if err != nil {
		return nil, err
	}
	vae, err := CompileVAEProgram(decoder, shape.Height, shape.Width, dtype.F32)
	if err != nil {
		return nil, err
	}
	pipeline, err := NewResidentImagePipeline(ctx, encoder, fusion, denoiser, vae, modelDir, 0)
	if err != nil {
		return nil, err
	}
	schedule, err := CompileFlowSchedule(request.Steps, request.NumTrainTimesteps, request.DynamicShiftMu)
	if err != nil {
		_ = pipeline.Close(ctx)
		return nil, err
	}
	return &Generator{
		request: request, spec: spec, pipeline: pipeline, embed: embed,
		schedule: schedule, shape: shape,
	}, nil
}

func (r Request) withDefaults() Request {
	if r.Steps == 0 {
		r.Steps = DefaultSteps
	}
	if r.NumTrainTimesteps == 0 {
		r.NumTrainTimesteps = DefaultNumTrainTimesteps
	}
	if r.DynamicShiftMu == 0 {
		r.DynamicShiftMu = DefaultDynamicShiftMu
	}
	return r
}

func (g *Generator) prepare(ctx context.Context, request Request) (*Generator, error) {
	if g == nil || g.pipeline == nil || request.withDefaults() != g.request || g.prepared {
		return nil, errors.New("latent image: generation session is unavailable")
	}
	if _, err := g.pipeline.Condition(ctx, g.embed); err != nil {
		return nil, err
	}
	var latent []float32
	err := g.pipeline.runtime.worker.Do(ctx, func(state *device.State) error {
		var seedErr error
		latent, seedErr = SeededInitLatent(
			torchrng.NewStream(g.request.Seed), state, g.shape, InitNoiseMix,
		)
		return seedErr
	})
	if err != nil {
		return nil, err
	}
	packed, gridH, gridW, err := packLatent(
		latent, g.shape.ZDim, g.shape.Height, g.shape.Width, g.spec.PatchSize,
	)
	if err != nil {
		return nil, err
	}
	if gridH != g.pipeline.Denoiser.GH || gridW != g.pipeline.Denoiser.GW {
		return nil, errors.New("latent image: compiled grid differs from seeded latent")
	}
	if err := g.pipeline.Begin(ctx, packed); err != nil {
		return nil, err
	}
	g.prepared = true
	return g, nil
}

func (g *Generator) integrate(ctx context.Context, session *Generator) (*Generator, error) {
	if g == nil || session != g || !g.prepared || g.integrated {
		return nil, errors.New("latent image: integration session is unavailable")
	}
	for step := range g.schedule.Steps {
		if err := g.pipeline.Advance(ctx, g.schedule.Sigmas[step], g.schedule.Deltas[step]); err != nil {
			return nil, err
		}
	}
	g.integrated = true
	return g, nil
}

func (g *Generator) decode(ctx context.Context, session *Generator) (Image, error) {
	if g == nil || session != g || !g.integrated {
		return Image{}, errors.New("latent image: decode session is unavailable")
	}
	pixels, height, width, err := g.pipeline.Decode(ctx)
	if err != nil {
		return Image{}, err
	}
	return Image{Pixels: pixels, Channels: g.pipeline.VAE.OutChannels, Height: height, Width: width}, nil
}

func (g *Generator) Close(ctx context.Context) error {
	if g == nil || g.pipeline == nil {
		return nil
	}
	err := g.pipeline.Close(ctx)
	g.pipeline = nil
	g.embed = nil
	return err
}

func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generator *Generator) error {
	if generator == nil || generator.pipeline == nil {
		return errors.New("latent image: incomplete runtime binding")
	}
	if err := workflowruntime.RegisterContextStage(
		runtime, modelrecipe.ModuleLatentImagePrepare, modelID, generator.prepare, nil,
	); err != nil {
		return err
	}
	if err := workflowruntime.RegisterContextStage(
		runtime, modelrecipe.ModuleLatentImageIntegrate, modelID, generator.integrate, nil,
	); err != nil {
		return err
	}
	return workflowruntime.RegisterContextStage(
		runtime, modelrecipe.ModuleLatentImageDecode, modelID, generator.decode,
		func(image Image) (artifact.Content, error) {
			return artifact.JSONContent(generatedImageContract, image)
		},
	)
}
