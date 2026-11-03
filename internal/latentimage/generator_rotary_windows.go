//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/torchrng"
)

// rotaryGeneration holds composition state under the shared Generator lifecycle.
// The resident session owns graph resources and sampling retains device latents.
type rotaryGeneration struct {
	runtime            *graphruntime.ResidentSession
	conditioner        *modelConditioner
	conditioned        conditioningCache
	transformer        TransformerSpec
	decoder            *VAEProgram
	latent             *residentLatent
	positive, negative []float32
	guidance           float32
}

func loadRotaryGenerator(ctx context.Context, root string, profile Profile, request Request) (_ *Generator, err error) {
	actual, err := ResolveProfile(root)
	if err != nil {
		return nil, err
	}
	if actual.ID != profile.ID {
		return nil, errors.New("latent image: recipe profile differs from artifact")
	}
	spec, err := ReadRotaryTransformer(root)
	if err != nil {
		return nil, err
	}
	conditioner, err := loadModelConditioner(root, profile.Prompt)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, conditioner.source.Close())
		}
	}()
	if int(conditioner.catalog.Spec.EmbeddingLength) != spec.TextHidden {
		return nil, errors.New("rotary image: conditioner and transformer widths differ")
	}
	decoder, err := loadSpatialVAEDecoder(root)
	if err != nil {
		return nil, err
	}
	channels, err := media.PatchVectorWidth(decoder.ZDim, decoder.patch)
	if err != nil {
		return nil, err
	}
	scale, ok := checked.MulInt(decoder.SpatialScale, decoder.patch)
	if !ok || scale != profile.Rotary.SpatialScale || decoder.OutChannels != media.RGBChannels || channels != spec.InChannels || request.Height%scale != 0 || request.Width%scale != 0 {
		return nil, errors.New("rotary image: request and component geometry differ")
	}
	height, width := request.Height/scale, request.Width/scale
	vae, err := CompileVAEProgram(decoder, height, width, dtype.F32)
	if err != nil {
		return nil, err
	}
	runtime, err := graphruntime.NewResidentSession(0)
	if err != nil {
		return nil, err
	}
	// The cache holds what two branches of the source's longest prompt hold.
	limit, ok := checked.ProductInt(tensor.PairedExtent, profile.Prompt.MaxTokens, spec.TextHidden)
	if !ok {
		return nil, errors.Join(errors.New("rotary image: conditioning cache limit overflows"), runtime.Close(context.WithoutCancel(ctx)))
	}
	g := &Generator{profile: profile, modelDir: root, shape: LatentShape{ZDim: channels, Height: height, Width: width}, rotary: &rotaryGeneration{runtime: runtime, conditioner: conditioner, conditioned: conditioningCache{limit: limit}, transformer: spec, decoder: vae}}
	if err = g.configureRotary(ctx, request); err == nil {
		err = admitPicture(ctx, runtime, height*width, spec.Hidden)
	}
	if err != nil {
		err = errors.Join(err, runtime.Close(context.WithoutCancel(ctx)))
		return nil, err
	}
	return g, nil
}

func (g *Generator) configureRotary(ctx context.Context, request Request) error {
	if err := ValidateRequest(request); err != nil {
		return err
	}
	if request.DynamicShiftMu != 0 || (request.NumTrainTimesteps != 0 && request.NumTrainTimesteps != g.profile.Sampling.TrainTimesteps) {
		return errors.New("rotary image: request cannot override the source training grid")
	}
	if request.Steps == 0 {
		request.Steps = g.profile.Sampling.DefaultSteps
	}
	request.NumTrainTimesteps = g.profile.Sampling.TrainTimesteps
	schedule, err := compileNearestGridFlowSchedule(ctx, request.Steps, request.NumTrainTimesteps, g.profile.Rotary.Shift, g.profile.Rotary.TrainingShift)
	if err != nil {
		return err
	}
	guidance := g.profile.Rotary.Guidance
	if request.GuidanceScale != nil {
		guidance = *request.GuidanceScale
	}
	settings := rotarySamplingSpec{schedule: schedule, guidance: guidance, skip: g.profile.Rotary.NegativeSkip, stop: 1}
	if err = settings.validate(g.rotary.transformer.Layers); err != nil {
		return err
	}
	g.request, g.schedule = request, schedule
	g.rotary.guidance = guidance
	return nil
}

func (g *Generator) resetRotary(ctx context.Context, request Request) error {
	if !g.completed || g.rotary.runtime == nil || g.rotary.latent != nil {
		return errors.New("rotary image: resident session is unavailable")
	}
	if request.Width != g.request.Width || request.Height != g.request.Height {
		return errors.New("rotary image: request requires another resident geometry")
	}
	if err := g.configureRotary(ctx, request); err != nil {
		return err
	}
	g.prepared, g.integrated, g.completed = false, false, false
	return nil
}

// negativeBranch is the place of the negative prompt among a request's prompts.
const negativeBranch = 1

func (g *Generator) prepareRotary(ctx context.Context, request Request) (_ *Generator, err error) {
	if g.prepared || g.rotary.runtime == nil {
		return nil, errors.New("rotary image: preparation session is unavailable")
	}
	normalized := request
	if normalized.Steps == 0 {
		normalized.Steps = g.profile.Sampling.DefaultSteps
	}
	if normalized.NumTrainTimesteps == 0 {
		normalized.NumTrainTimesteps = g.profile.Sampling.TrainTimesteps
	}
	expected := g.request
	actualGuidance := g.profile.Rotary.Guidance
	if normalized.GuidanceScale != nil {
		actualGuidance = *normalized.GuidanceScale
	}
	normalized.GuidanceScale, expected.GuidanceScale = nil, nil
	if normalized != expected || actualGuidance != g.rotary.guidance {
		return nil, errors.New("rotary image: request differs from loaded session")
	}
	if g.observe != nil {
		for name, values := range map[string][]float64{"schedule.sigmas": g.schedule.Sigmas, "schedule.timesteps": g.schedule.Timesteps} {
			actual := make([]float32, len(values))
			for i, v := range values {
				actual[i] = float32(v)
			}
			if err := g.observe(name, actual); err != nil {
				return nil, err
			}
		}
	}
	// The request's prompts not encoded before pass through the text encoder
	// together, a block at a time, so the device holds one block of it. An
	// observed request reports every boundary, so it encodes what a cached
	// prompt skips.
	prompts := []string{request.Prompt}
	if g.rotary.guidance != tensor.UnitScale {
		prompts = append(prompts, request.NegativePrompt)
	}
	hidden := make([][]float32, len(prompts))
	var texts []textInput
	var embeddings [][]float32
	var missing []int
	for index, prompt := range prompts {
		if cached, ok := g.rotary.conditioned.lookup(prompt); ok && g.observe == nil {
			hidden[index] = cached
			continue
		}
		text, embedding, err := g.rotary.conditioner.input(prompt)
		if err != nil {
			return nil, err
		}
		texts, embeddings, missing = append(texts, text), append(embeddings, embedding), append(missing, index)
	}
	if len(missing) != 0 {
		encoded, err := g.rotary.conditioner.encodeEach(ctx, g.rotary.runtime, texts, embeddings)
		if err = errors.Join(err, g.rotary.runtime.ReleaseStageMemory(context.WithoutCancel(ctx))); err != nil {
			return nil, err
		}
		for position, index := range missing {
			hidden[index] = encoded[position]
			g.rotary.conditioned.store(prompts[index], encoded[position])
			if g.observe == nil {
				continue
			}
			ids := make([]float32, len(texts[position].IDs))
			for i, v := range texts[position].IDs {
				ids[i] = float32(v)
			}
			if err = g.observe(fmt.Sprintf("conditioning.%d.ids", index), ids); err != nil {
				return nil, err
			}
			if err = g.observe(fmt.Sprintf("conditioning.%d.hidden", index), encoded[position]); err != nil {
				return nil, err
			}
		}
	}
	g.rotary.positive, g.rotary.negative = hidden[0], nil
	if len(hidden) > negativeBranch {
		g.rotary.negative = hidden[negativeBranch]
	}
	g.prepared = true
	return g, nil
}

func (g *Generator) integrateRotary(ctx context.Context) (*Generator, error) {
	count, ok := checked.ProductInt(g.shape.ZDim, g.shape.Height, g.shape.Width)
	if !ok {
		return nil, errors.New("rotary image: latent extent overflows")
	}
	initial, err := torchrng.CPUNormal(g.request.Seed, count)
	if err != nil {
		return nil, err
	}
	settings := rotarySamplingSpec{schedule: g.schedule, guidance: g.rotary.guidance, skip: g.profile.Rotary.NegativeSkip, stop: 1}
	result, err := runRotarySampling(ctx, g.rotary.runtime, g.modelDir, g.rotary.transformer, g.shape.Height, g.shape.Width, initial, g.rotary.positive, g.rotary.negative, settings, g.observe)
	if err != nil {
		return nil, err
	}
	g.rotary.latent = result
	g.rotary.positive, g.rotary.negative = nil, nil
	g.integrated = true
	return g, g.rotary.runtime.ReleaseStageMemory(ctx)
}

func (g *Generator) decodeRotary(ctx context.Context) (result EncodedImage, err error) {
	state := g.rotary
	if state.latent == nil || g.completed {
		return result, errors.New("rotary image: decoded latent is unavailable")
	}
	defer func() {
		err = errors.Join(err, state.latent.outputs.Release(context.WithoutCancel(ctx)))
		state.latent = nil
	}()
	p := state.decoder
	static := map[*tensor.Tensor][]float32{}
	for _, feed := range p.feeds {
		static[feed.node] = feed.data
	}
	program, err := compileResidentStatic(ctx, state.runtime, "rotary image decoder", static, []*tensor.Tensor{p.Latent}, p.pixels)
	if err != nil {
		return result, err
	}
	defer func() {
		err = errors.Join(err, state.runtime.Release(context.WithoutCancel(ctx), program), state.runtime.ReleaseStageMemory(context.WithoutCancel(ctx)))
	}()
	output, err := state.runtime.ExecuteDevice(ctx, program, nil, []driver.DevicePtr{state.latent.value.Pointer})
	if err != nil {
		return result, err
	}
	pixels := output[p.pixels].Data
	if g.observe != nil {
		planar := make([]float32, len(pixels))
		for i := range p.OutH * p.OutW {
			for c := range p.OutChannels {
				planar[c*p.OutH*p.OutW+i] = pixels[i*p.OutChannels+c]
			}
		}
		if err = g.observe("pixels", planar); err != nil {
			return result, err
		}
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	result, err = encodeUnitPNG(pixels, p.OutH, p.OutW)
	if err != nil {
		return result, err
	}
	if err = ctx.Err(); err != nil {
		return EncodedImage{}, err
	}
	g.completed = true
	return result, nil
}

func (g *Generator) closeRotary(ctx context.Context) error {
	state := g.rotary
	g.rotary = nil
	var err error
	if state.latent != nil {
		err = state.latent.outputs.Release(context.WithoutCancel(ctx))
	}
	return errors.Join(err, state.runtime.Close(context.WithoutCancel(ctx)), state.conditioner.source.Close())
}
