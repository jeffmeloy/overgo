//go:build windows

package latentimage

import (
	"context"
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
	Steps             int     `json:"steps,omitzero"`
	Seed              int64   `json:"seed"`
	DynamicShiftMu    float64 `json:"mu,omitzero"`
	NumTrainTimesteps int     `json:"num_train_timesteps,omitzero"`
}

type Generator struct {
	request    Request
	profile    Profile
	spec       *Spec
	modelDir   string
	tokenizer  *hfbpe.Tokenizer
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
	if err := ctx.Err(); err != nil {
		return nil, err
	}
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
	request = request.withPolicy(profile.Sampling, gridH*gridW, spec.Distilled)
	tokenizer, err := hfbpe.Load(filepath.Join(modelDir, "tokenizer"))
	if err != nil {
		return nil, err
	}
	generator := &Generator{request: request, profile: profile, spec: spec, modelDir: modelDir, tokenizer: tokenizer, shape: shape}
	prompt, err := generator.compilePrompt(request.Prompt, gridH, gridW)
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
	pipeline, err := NewResidentImagePipeline(ctx, prompt.encoder, prompt.fusion, prompt.denoiser, vae, timestepEncoding, modelDir, tensor.FirstOffset)
	if err != nil {
		return nil, err
	}
	schedule, err := CompileFlowSchedule(request.Steps, request.NumTrainTimesteps, request.DynamicShiftMu)
	if err != nil {
		_ = pipeline.Close(ctx)
		return nil, err
	}
	generator.pipeline, generator.embed, generator.schedule = pipeline, prompt.embed, schedule
	return generator, nil
}

// promptPrograms is what a prompt gives a session: the rows of its tokens'
// embeddings, and the text encoder, fusion and denoiser compiled under its
// padding mask. Every prompt is padded to one length, so the three are of one
// shape whatever the prompt, and differ in what the mask leaves unattended.
type promptPrograms struct {
	embed    []float32
	encoder  *EncoderProgram
	fusion   *FusionProgram
	denoiser *DenoiserProgram
}

// compilePrompt renders a prompt as the model's text template has it and
// compiles the programs that read its mask, for a picture of gridH by gridW
// patches.
func (g *Generator) compilePrompt(prompt string, gridH, gridW int) (promptPrograms, error) {
	text, err := renderTextInput(g.tokenizer, prompt, g.profile.Prompt)
	if err != nil {
		return promptPrograms{}, err
	}
	embed, err := safetensors.ReadTensorRowsF32(
		filepath.Join(g.modelDir, "text_encoder"), textEncoderPrefix+"embed_tokens.weight", g.spec.TextEncoder.Hidden, text.IDs,
	)
	if err != nil {
		return promptPrograms{}, err
	}
	encoder, err := CompileEncoderProgramMasked(
		g.spec.TextEncoder, float32(g.spec.TextEncoder.RMSNormEps), len(text.IDs), dtype.BF16, text.Mask,
	)
	if err != nil {
		return promptPrograms{}, err
	}
	textMask, err := checked.SuffixExact(text.Mask, text.PromptRows)
	if err != nil {
		return promptPrograms{}, fmt.Errorf("latent image: prompt mask: %w", err)
	}
	fusion, err := CompileFusionProgram(
		g.spec.Transformer, float32(g.spec.Transformer.NormEps), textMask, dtype.BF16,
	)
	if err != nil {
		return promptPrograms{}, err
	}
	denoiser, err := CompileDenoiserProgram(
		g.spec.Transformer, float32(g.spec.Transformer.NormEps), textMask, gridH, gridW, dtype.BF16,
	)
	return promptPrograms{embed: embed, encoder: encoder, fusion: fusion, denoiser: denoiser}, err
}

// withPolicy fills what a request leaves unnamed from the sampling policy: the
// steps, the training timesteps, and the timestep shift for a picture of
// imageSequence patches drawn by a checkpoint that is or is not distilled.
func (r Request) withPolicy(policy samplingPolicy, imageSequence int, distilled bool) Request {
	if !checked.Nonzero(r.Steps) {
		r.Steps = policy.DefaultSteps
	}
	if !checked.Nonzero(r.NumTrainTimesteps) {
		r.NumTrainTimesteps = policy.TrainTimesteps
	}
	if !checked.Nonzero(r.DynamicShiftMu) {
		r.DynamicShiftMu = policy.dynamicShiftMu(imageSequence, distilled)
	}
	return r
}

// SessionKey identifies reusable resident state for the request: the
// picture's extent, which the denoiser and the decoder are compiled for. A
// prompt and a seed are what a request gives a session, not what one is made
// of: the weights a session holds are the same under every prompt.
func (r Request) SessionKey() (string, error) {
	return fmt.Sprintf("%dx%d", r.Width, r.Height), nil
}

// Reset gives a resident session its next request: the sampling state of any
// request of its extent, and, for another prompt, that prompt's conditioning,
// made by the text encoder placed on the device again beside the denoiser the
// session keeps.
func (g *Generator) Reset(ctx context.Context, request Request) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if g == nil || g.pipeline == nil || !g.completed {
		return errors.New("latent image: resident session is unavailable")
	}
	if err := ValidateRequest(request); err != nil {
		return err
	}
	gridH, gridW := g.pipeline.Denoiser.GH, g.pipeline.Denoiser.GW
	request = request.withPolicy(g.profile.Sampling, gridH*gridW, g.spec.Distilled)
	if request.Width != g.request.Width || request.Height != g.request.Height {
		return errors.New("latent image: request requires another resident session")
	}
	if request.Prompt != g.request.Prompt {
		prompt, err := g.compilePrompt(request.Prompt, gridH, gridW)
		if err != nil {
			return err
		}
		if err := g.pipeline.Recondition(ctx, prompt.encoder, prompt.fusion, prompt.denoiser, prompt.embed); err != nil {
			return err
		}
		g.embed = prompt.embed
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
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if g == nil || g.pipeline == nil || request.withPolicy(g.profile.Sampling, g.pipeline.Denoiser.GH*g.pipeline.Denoiser.GW, g.spec.Distilled) != g.request || g.prepared {
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
	err := g.pipeline.Close(context.WithoutCancel(ctx))
	g.pipeline = nil
	g.embed = nil
	return err
}

func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generator *Generator) error {
	if generator == nil || generator.pipeline == nil {
		return errors.New("latent image: incomplete runtime binding")
	}
	return workflowruntime.RegisterContextPipeline(
		runtime, modelID,
		modelrecipe.ModuleLatentImagePrepare, generator.prepare,
		modelrecipe.ModuleLatentImageIntegrate, generator.integrate,
		modelrecipe.ModuleLatentImageDecode, generator.decode,
		PNGContent,
	)
}
