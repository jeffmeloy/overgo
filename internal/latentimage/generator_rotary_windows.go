//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/media"
	"overgo/internal/processmeasure"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/torchrng"
)

// rotaryGeneration holds composition state under the shared Generator lifecycle.
// The resident session owns graph resources and sampling retains device latents.
type rotaryGeneration struct {
	runtime            *graphruntime.ResidentSession
	conditioner        *modelConditioner
	conditioned        media.ConditioningCache
	transformer        TransformerSpec
	decoder            *VAEProgram
	latent             *residentLatent
	positive, negative []float32
	// kept are the programs the last sampling left, whose transformer weights the next sampling binds again on the card
	// (runRotarySampling); the session's close frees them.
	kept []*graphruntime.ResidentProgram
	// weights are the attention-value weights of the prompt's rows, and negativeWeights the negative prompt's, none
	// for a prompt that weights nothing.
	weights, negativeWeights []float32
	guidance                 float32
	// adapters are the adapters the request applies (Request.applied), each read from its file when the request is
	// configured, none where it names none, and contents the content each adapter file a request named was last read
	// with, by its path, under which the session keeps its pairs (bindMerged).
	adapters []*mediaAdapter
	contents map[string]string
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
			err = errors.Join(err, conditioner.close())
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
	g := &Generator{profile: profile, modelDir: root, shape: LatentShape{ZDim: channels, Height: height, Width: width}, rotary: &rotaryGeneration{runtime: runtime, conditioner: conditioner, conditioned: media.ConditioningCache{Limit: limit, Kept: []string{""}}, transformer: spec, decoder: vae}}
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
	request.NumTrainTimesteps = g.profile.Sampling.TrainTimesteps
	schedule, err := g.profile.rotarySchedule(ctx, request)
	if err != nil {
		return err
	}
	request.Steps = schedule.Steps
	guidance := g.profile.Rotary.Guidance
	if request.GuidanceScale != nil {
		guidance = *request.GuidanceScale
	}
	g.rotary.adapters = nil
	if g.rotary.contents == nil {
		g.rotary.contents = map[string]string{}
	}
	for _, applied := range request.applied() {
		adapter, err := modelAdapter(g.modelDir, applied.Name)
		if err != nil {
			return err
		}
		// A file replaced under its name leaves the pairs the session keeps under its earlier content unasked for.
		if before, read := g.rotary.contents[adapter.path]; read && before != adapter.content {
			if err := g.rotary.runtime.Unkeep(ctx, before); err != nil {
				return err
			}
		}
		g.rotary.contents[adapter.path] = adapter.content
		g.rotary.adapters = append(g.rotary.adapters, adapter)
	}
	if twice := replacedTwice(g.rotary.adapters); twice != "" {
		return fmt.Errorf("rotary image: two of the request's adapters replace %s, which a graph reads from one", twice)
	}
	if err = g.rotarySettings(request, schedule, guidance).validate(g.rotary.transformer.Layers); err != nil {
		return err
	}
	g.request, g.schedule = request, schedule
	g.rotary.guidance = guidance
	return nil
}

// rotarySchedule is a rotary request's schedule at its steps, the profile's default where it names none: the profile's
// own, its shifted knots snapped to its training grid, or the request's (Request.Schedule), drawn over the profile's
// flow levels as ComfyUI's KSampler draws it and not snapped, since ComfyUI samples the sigmas as drawn
// (moss-ultra-sections/schedules). A list of the request's own sigmas is its steps. A request that denoises part of
// its schedule has it drawn for int(steps / denoise) steps, its last steps + 1 sigmas kept, as ComfyUI's KSampler does
// (moss-ultra-sections/img2img).
func (p Profile) rotarySchedule(ctx context.Context, request Request) (FlowSchedule, error) {
	total := p.Sampling.TrainTimesteps
	// A schedule that names a sampler alone samples by the profile's own sigmas.
	own := request.Schedule != nil && request.Schedule.OwnSigmas()
	if request.Steps == 0 && (!own || request.Schedule.Name != "") {
		request.Steps = p.Sampling.DefaultSteps
	}
	drawn := request.Steps
	if request.cutsSchedule() {
		drawn = int(float64(request.Steps) / request.Denoise)
	}
	var schedule FlowSchedule
	if !own {
		var err error
		if schedule, err = compileNearestGridFlowSchedule(ctx, drawn, total, p.Rotary.Shift, p.Rotary.TrainingShift); err != nil {
			return FlowSchedule{}, err
		}
	} else {
		table, err := sampling.ShiftedFlowTable(total, p.Rotary.Shift)
		if err != nil {
			return FlowSchedule{}, err
		}
		sigmas, err := request.Schedule.Draw(drawn, table)
		if err != nil {
			return FlowSchedule{}, fmt.Errorf("rotary image: %w", err)
		}
		schedule = sigmaFlowSchedule(sigmas, total)
	}
	if !request.cutsSchedule() {
		return schedule, nil
	}
	return cutSchedule(schedule, request.Steps)
}

// rotarySettings is the sampling a rotary request runs: its schedule, its guidance scale and how the guidance varies over
// the steps, the negative branch changing the profile's blocks, and the sampler it names and its seed.
func (g *Generator) rotarySettings(request Request, schedule FlowSchedule, guidance float32) rotarySamplingSpec {
	settings := rotarySamplingSpec{schedule: schedule, guidance: guidance, varies: request.Guidance, override: g.profile.Rotary.negativeOverride(), seed: request.Seed,
		eightBit: request.EightBitAttention, adapters: g.rotary.adapters}
	for _, applied := range request.applied() {
		settings.strengths = append(settings.strengths, applied.strength())
	}
	if request.SkipBlocks != nil {
		settings.override.Blocks = slices.Clone(*request.SkipBlocks)
	}
	if request.Schedule != nil {
		settings.sampler = request.Schedule.Sampler
	}
	return settings
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
	// A depth control is compared by what it draws (sameControl), a preset's fill of it being a copy each time.
	controlsAlike := sameControl(normalized.Control, expected.Control)
	normalized.Control, expected.Control = nil, nil
	if !controlsAlike || normalized != expected || actualGuidance != g.rotary.guidance {
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
	// prompt skips. Each prompt is rendered once, its rows' weights read from that
	// rendering whether its conditioning is kept or encoded.
	prompts := []string{request.prompt()}
	if g.rotary.guidance != tensor.UnitScale {
		prompts = append(prompts, request.NegativePrompt)
	}
	hidden, weights := make([][]float32, len(prompts)), make([][]float32, len(prompts))
	var texts []textInput
	var embeddings [][]float32
	var missing []int
	for index, prompt := range prompts {
		text, err := renderTextInput(g.rotary.conditioner.tokenizer, prompt, g.rotary.conditioner.template)
		if err != nil {
			return nil, err
		}
		weights[index] = text.Weights
		if cached, ok := g.rotary.conditioned.Lookup("", prompt); ok && g.observe == nil {
			hidden[index] = cached
			continue
		}
		embedding, err := g.rotary.conditioner.embedding(text)
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
			g.rotary.conditioned.Store("", prompts[index], encoded[position])
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
	g.rotary.positive, g.rotary.weights, g.rotary.negative, g.rotary.negativeWeights = hidden[0], weights[0], nil, nil
	if len(hidden) > negativeBranch {
		g.rotary.negative, g.rotary.negativeWeights = hidden[negativeBranch], weights[negativeBranch]
	}
	g.prepared = true
	return g, nil
}

func (g *Generator) integrateRotary(ctx context.Context) (*Generator, error) {
	count, ok := checked.ProductInt(g.shape.ZDim, g.shape.Height, g.shape.Width)
	if !ok {
		return nil, errors.New("rotary image: latent extent overflows")
	}
	noise, err := torchrng.CPUNormal(g.request.Seed, count)
	if err != nil {
		return nil, err
	}
	initial, err := g.startLatent(ctx, noise)
	if err != nil {
		return nil, err
	}
	settings := g.rotarySettings(g.request, g.schedule, g.rotary.guidance)
	settings.weights, settings.negativeWeights = g.rotary.weights, g.rotary.negativeWeights
	if control := g.request.Control; control != nil {
		if settings.control, err = g.sections().controlLatent(ctx, *control); err != nil {
			return nil, err
		}
		settings.controlLow, settings.controlHigh = control.window(g.profile.Rotary.Shift)
	}
	result, err := runRotarySampling(ctx, g.rotary.runtime, &g.rotary.kept, g.modelDir, g.rotary.transformer, g.shape.Height, g.shape.Width, initial, g.rotary.positive, g.rotary.negative, settings, g.observe)
	// The adapters' files close as the sampling ends, the pairs their merges read kept on the card (SQA-439).
	for _, adapter := range g.rotary.adapters {
		err = errors.Join(err, adapter.release())
	}
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
	// The decode's parts are named for a ledger that reads them (processmeasure.Stage): the decoder compiled, the picture
	// decoded on the card and copied out, and the picture finished (png-level).
	compiling := processmeasure.Stage(ctx, "decoder compiled")
	program, err := compileResidentStatic(ctx, state.runtime, "rotary image decoder", static, []*tensor.Tensor{p.Latent}, p.pixels)
	compiling()
	if err != nil {
		return result, err
	}
	defer func() {
		err = errors.Join(err, state.runtime.Release(context.WithoutCancel(ctx), program), state.runtime.ReleaseStageMemory(context.WithoutCancel(ctx)))
	}()
	decoding := processmeasure.Stage(ctx, "decoded on the card")
	output, err := state.runtime.ExecuteDevice(ctx, program, nil, []driver.DevicePtr{state.latent.value.Pointer})
	decoding()
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
	// The picture is sharpened as it is finished, after the decoded pixels a parity observes; the picture drawn is kept
	// beside it, as the MOSS Ultra workflow saves each section's picture and its sharpened picture and compares them
	// (moss-sharpened-pair), each encoded beside the other (pair-encode).
	var drawn EncodedImage
	finishing := processmeasure.Stage(ctx, "finished")
	result, drawn, err = finishedPNGs(pixels, p.OutH, p.OutW, g.request.Sharpen)
	finishing()
	if err != nil {
		return result, err
	}
	result.Drawn = drawn.Data
	// What the picture was drawn from travels with it (EncodedImage.Prompt, draw-result-carries): the prompt its text
	// encoder read, its layout's where it gives one, and its negative, its preset's filled.
	result.Prompt, result.NegativePrompt, result.Sharpened = g.request.prompt(), g.request.NegativePrompt, g.request.Sharpen != nil
	// Its bytes are those its store keeps (PNGContent), what it was drawn from written into them.
	if result.Data, err = withPictureTexts(result.Data, result.pictureTexts()); err != nil {
		return result, err
	}
	sampler := "flow Euler"
	if g.request.Schedule != nil && g.request.Schedule.Sampler == sampling.SamplerERSDE {
		sampler = "ER-SDE"
	}
	result.Preset = media.Preset{Width: g.request.Width, Height: g.request.Height, Steps: g.request.Steps, Guidance: float64(state.guidance),
		Sampler: sampler, Shift: g.profile.Rotary.Shift}
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
	return errors.Join(err, state.runtime.Close(context.WithoutCancel(ctx)), state.conditioner.close())
}
