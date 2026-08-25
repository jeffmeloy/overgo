//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/hfbpe"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/torchrng"
	"overgo/internal/workflowruntime"
)

type Request struct {
	Prompt            string  `json:"prompt"`
	Width             int     `json:"width"`
	Height            int     `json:"height"`
	Steps             int     `json:"steps,omitempty"`
	Seed              int64   `json:"seed"`
	DynamicShiftMu    float64 `json:"mu,omitempty"`
	NumTrainTimesteps int     `json:"num_train_timesteps,omitempty"`
}

type Generator struct {
	request    Request
	profile    Profile
	spec       *Spec
	pipeline   *ResidentImagePipeline
	embed      []float32
	schedule   FlowSchedule
	shape      LatentShape
	prepared   bool
	integrated bool
	completed  bool
}

func ValidateRequest(request Request) error {
	if !checked.Nonzero(request.Prompt) {
		return errors.New("latent image: prompt is empty")
	}
	if err := media.ValidateSpatialGeometry(request.Height, request.Width); err != nil {
		return fmt.Errorf("latent image: invalid extent: %w", err)
	}
	if !checked.NonNegativeInts(request.Steps, request.NumTrainTimesteps) || !checked.Finite64(request.DynamicShiftMu) {
		return errors.New("latent image: invalid sampling policy")
	}
	return nil
}

func LoadGenerator(ctx context.Context, modelDir string, profile Profile, request Request) (*Generator, error) {
	if err := ValidateRequest(request); err != nil {
		return nil, err
	}
	if err := profile.validateIdentity(); err != nil {
		return nil, err
	}
	spec, err := Derive(modelDir)
	if err != nil {
		return nil, err
	}
	if spec.Profile != profile.ID {
		return nil, errors.New("latent image: recipe profile differs from artifact")
	}
	if _, err := spec.VerifyCheckpoint(modelDir); err != nil {
		return nil, err
	}
	channels, latentHeight, latentWidth, err := media.DownsampledPlanarGeometry(
		spec.VAE.ZDim, request.Height, request.Width, spec.VAE.SpatialScale,
	)
	if err != nil {
		return nil, err
	}
	shape := LatentShape{ZDim: channels, Height: latentHeight, Width: latentWidth}
	gridH, gridW, err := media.PatchGrid(shape.Height, shape.Width, spec.PatchSize)
	if err != nil {
		return nil, fmt.Errorf("latent image: patch grid: %w", err)
	}
	request = request.withPolicy(profile.Sampling, gridH*gridW)
	tokenizer, err := hfbpe.Load(filepath.Join(modelDir, "tokenizer"))
	if err != nil {
		return nil, err
	}
	text, err := renderTextInput(tokenizer, request.Prompt, profile.Prompt)
	if err != nil {
		return nil, err
	}
	embed, err := safetensors.ReadTensorRowsF32(
		filepath.Join(modelDir, "text_encoder"), textEncoderPrefix+"embed_tokens.weight", spec.TextEncoder.Hidden, text.IDs,
	)
	if err != nil {
		return nil, err
	}
	encoder, err := CompileEncoderProgramMasked(
		spec.TextEncoder, float32(spec.TextEncoder.RMSNormEps), len(text.IDs), dtype.BF16, text.Mask,
	)
	if err != nil {
		return nil, err
	}
	textMask, err := checked.SuffixExact(text.Mask, text.PromptRows)
	if err != nil {
		return nil, fmt.Errorf("latent image: prompt mask: %w", err)
	}
	fusion, err := CompileFusionProgram(
		spec.Transformer, float32(spec.Transformer.NormEps), textMask, dtype.BF16,
	)
	if err != nil {
		return nil, err
	}
	denoiser, err := CompileDenoiserProgram(
		spec.Transformer, float32(spec.Transformer.NormEps), textMask, gridH, gridW, dtype.BF16,
	)
	if err != nil {
		return nil, err
	}
	decoder, err := loadVAEDecoder(modelDir, profile.Classes.VAE)
	if err != nil {
		return nil, err
	}
	vae, err := CompileVAEProgram(decoder, shape.Height, shape.Width, dtype.F32)
	if err != nil {
		return nil, err
	}
	timestepEncoding := media.SinusoidalProgram{
		Dimensions: spec.Transformer.TimestepEmbed, FrequencyBase: profile.Sampling.TimestepFrequencyBase,
		InputScale: float64(request.NumTrainTimesteps),
	}
	pipeline, err := NewResidentImagePipeline(ctx, encoder, fusion, denoiser, vae, timestepEncoding, modelDir, tensor.FirstOffset)
	if err != nil {
		return nil, err
	}
	schedule, err := CompileFlowSchedule(request.Steps, request.NumTrainTimesteps, request.DynamicShiftMu)
	if err != nil {
		_ = pipeline.Close(ctx)
		return nil, err
	}
	return &Generator{
		request: request, profile: profile, spec: spec, pipeline: pipeline, embed: embed,
		schedule: schedule, shape: shape,
	}, nil
}

func (r Request) withPolicy(policy samplingPolicy, imageSequence int) Request {
	if !checked.Nonzero(r.Steps) {
		r.Steps = policy.DefaultSteps
	}
	if !checked.Nonzero(r.NumTrainTimesteps) {
		r.NumTrainTimesteps = policy.TrainTimesteps
	}
	if !checked.Nonzero(r.DynamicShiftMu) {
		r.DynamicShiftMu = policy.dynamicShiftMu(imageSequence)
	}
	return r
}

// SessionKey identifies reusable resident state for the request.
func (r Request) SessionKey() (string, error) {
	prompt := sha256.Sum256([]byte(r.Prompt))
	return fmt.Sprintf("%dx%d/prompt:%x", r.Width, r.Height, prompt), nil
}

// Reset reuses resident graphs for compatible request-local sampling state.
func (g *Generator) Reset(ctx context.Context, request Request) error {
	if g == nil || g.pipeline == nil || !g.completed {
		return errors.New("latent image: resident session is unavailable")
	}
	if err := ValidateRequest(request); err != nil {
		return err
	}
	request = request.withPolicy(g.profile.Sampling, g.pipeline.Denoiser.GH*g.pipeline.Denoiser.GW)
	if request.Prompt != g.request.Prompt || request.Width != g.request.Width || request.Height != g.request.Height {
		return errors.New("latent image: request requires another resident session")
	}
	schedule, err := CompileFlowSchedule(request.Steps, request.NumTrainTimesteps, request.DynamicShiftMu)
	if err != nil {
		return err
	}
	g.request, g.schedule = request, schedule
	g.prepared, g.integrated, g.completed = false, false, false
	return nil
}

func (g *Generator) prepare(ctx context.Context, request Request) (*Generator, error) {
	if g == nil || g.pipeline == nil || request.withPolicy(g.profile.Sampling, g.pipeline.Denoiser.GH*g.pipeline.Denoiser.GW) != g.request || g.prepared {
		return nil, errors.New("latent image: generation session is unavailable")
	}
	if g.pipeline.conditioning == nil {
		if _, err := g.pipeline.Condition(ctx, g.embed); err != nil {
			return nil, err
		}
	}
	var latent []float32
	err := g.pipeline.runtime.Do(ctx, func(state *device.State) error {
		var seedErr error
		latent, seedErr = SeededInitLatent(
			torchrng.NewStream(g.request.Seed), state, g.shape, InitNoiseMix,
		)
		return seedErr
	})
	if err != nil {
		return nil, err
	}
	packed, gridH, gridW, err := media.PackPlanar(
		latent, g.shape.ZDim, g.shape.Height, g.shape.Width, g.spec.PatchSize, media.PatchChannelsFirst,
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

func (g *Generator) decode(ctx context.Context, session *Generator) (EncodedImage, error) {
	if g == nil || session != g || !g.integrated {
		return EncodedImage{}, errors.New("latent image: decode session is unavailable")
	}
	pixels, height, width, err := g.pipeline.DecodeHWC(ctx)
	if err != nil {
		return EncodedImage{}, err
	}
	if err := g.pipeline.Finish(ctx); err != nil {
		return EncodedImage{}, err
	}
	g.completed = true
	return encodePNG(pixels, height, width)
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
		PNGContent,
	)
}
