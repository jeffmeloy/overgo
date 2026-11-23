//go:build windows

package latentimage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"math"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/graphruntime"
	"overgo/internal/heldfile"
	"overgo/internal/hfbpe"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/processmeasure"
	"overgo/internal/safetensors"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/torchrng"
	"overgo/internal/workflowruntime"
)

// activationBytes is the storage of one F32 activation.
const activationBytes = 4

type Request struct {
	Prompt            string     `json:"prompt" role:"prompt"`
	PromptMode        PromptMode `json:"prompt_mode,omitempty"`
	NegativePrompt    string     `json:"negative_prompt,omitempty" role:"negative prompt"`
	GuidanceScale     *float32   `json:"guidance_scale,omitempty" role:"guidance"`
	Width             int        `json:"width" role:"width"`
	Height            int        `json:"height" role:"height"`
	Steps             int        `json:"steps,omitzero" role:"steps"`
	Seed              int64      `json:"seed" role:"seed"`
	DynamicShiftMu    float64    `json:"mu,omitzero"`
	NumTrainTimesteps int        `json:"num_train_timesteps,omitzero"`
	// Guidance is how the guidance varies over the steps (sampling.Guider.At), which a rotary profile alone runs.
	Guidance sampling.Guidance `json:"guidance,omitzero"`
	// Preset names a recipe of the model's catalog presets, its steps and guidance filling what the request leaves unset.
	Preset string `json:"preset,omitzero" role:"preset"`
	// EightBitAttention is the speed a preset chose (imagePreset.EightBitAttention). A request asks it only by that
	// preset, so a page offers it as the speed choice it is and no control beside it.
	EightBitAttention bool `json:"-"`
	// Schedule is the sigmas a rotary profile (MOSS) samples by in place of its own: a scheduler by name (bong_tangent,
	// beta57) or a list of the request's own, as ComfyUI's KSampler takes them.
	Schedule *sampling.Schedule `json:"schedule,omitempty" controls:"fields"`
	// Canvas sizes the picture from an aspect ratio and megapixels, as ComfyUI's ResolutionSelector does, in place of a
	// width and height the request then leaves unset.
	Canvas *media.Canvas `json:"canvas,omitempty" controls:"fields"`
	// Sharpen sharpens a rotary profile's picture after it is decoded, as ComfyUI's ImageSharpen does.
	Sharpen *media.Sharpening `json:"sharpen,omitempty" controls:"fields" label:"sharpening"`
	// Adapter names one of the model's adapters (<model>/adapters/<name>.safetensors), which a rotary profile applies
	// unmerged at AdapterStrength, 1 where the request states none (mediaAdapter): a new strength rewrites no weight.
	Adapter         string   `json:"adapter,omitzero"`
	AdapterStrength *float32 `json:"adapter_strength,omitempty"`
	// Adapters names several of the model's adapters, each at its strength, applied together in their order, as the
	// MOSS Ultra workflow's Turbo sections stack theirs (rgthree's Power Lora Loader: DMD2, moss-control, the SFT
	// adapter); a request names its adapters so or one Adapter, not both.
	Adapters *[]AppliedAdapter `json:"adapters,omitempty" controls:"fields"`
	// Image is a picture, a PNG in base64 as JSON gives bytes, that a rotary profile samples from (image to image):
	// scaled to the canvas as ComfyUI's ImageScale scales it at lanczos with a centre crop (media.ScaleLikeComfyUI),
	// encoded by the model's VAE, and started from at the schedule's first sigma, σ₀ times the noise plus 1 − σ₀
	// times the picture's latent (sourceLatent). It is text so the request stays comparable. A page names it by its
	// stored identity instead, the file it attached, as the workflow's LoadImage reads one, which the image capability
	// reads into its bytes before the session takes the request (ReadPictures).
	Image string `json:"image,omitzero" label:"picture" media:"image" role:"source"`
	// Denoise is the share of its schedule a request samples, as ComfyUI's KSampler takes it: below 1 the schedule is
	// drawn for int(steps / denoise) steps and its last steps + 1 sigmas kept (cutSchedule), and the sampling starts
	// from σ₀ times the noise plus 1 − σ₀ times the picture's latent, or none. 0 where the request states none, 1 in
	// effect, and above 0.9999 the whole schedule, as ComfyUI rounds it.
	Denoise float64 `json:"denoise,omitzero"`
	// Control is a depth map a rotary request samples by, through the control adapter it names (DepthControl).
	Control *DepthControl `json:"control,omitempty" controls:"fields" label:"depth control"`
	// SkipBlocks are the blocks a rotary profile's negative pass skips, as ComfyUI's SkipLayerGuidanceDiTSimple names
	// them: the profile's where the request states none (MOSS's 22, which the MOSS Ultra workflow's text and image
	// sections skip), none where it states an empty list (its depth sections, which skip none).
	SkipBlocks *[]int `json:"skip_blocks,omitempty"`
	// Layout is a rotary request's prompt as the MOSS Ultra workflow's prompt mode 3 takes it, its Ideogram 4 builder's
	// fields and regions, which make MOSS's layout prompt on the canvas the request draws (LayoutDraft.prompt) in place
	// of a prompt of its own.
	Layout *LayoutDraft `json:"layout,omitempty" controls:"fields"`
}

// Enhanced says the request asks the MOSS Ultra workflow's prompt mode 2 of a rotary request (PromptMode enhanced): its
// prompt a source idea and its negative the exclusions, of which the model's enhancer chat writes the prompt and
// negative it is drawn by (EnhanceRequest), before the picture session takes it.
func (r Request) Enhanced() bool { return r.PromptMode == promptEnhanced }

// errEnhanceBesideLayout refuses prompt modes 2 and 3 asked together, as the workflow's one prompt mode takes either.
var errEnhanceBesideLayout = errors.New("latent image: prompt mode 2 enhances a prompt, and a layout is mode 3's; ask the one or the other")

// errBuilderWithoutLayout refuses the builder's prompt mode (prompt mode 3) with no layout to make its prompt.
var errBuilderWithoutLayout = errors.New("latent image: prompt mode builder makes its prompt from a layout; give its layout")

// errNotEnhanced refuses a request still asking prompt mode 2 where a picture session takes it: its enhancer chat
// writes it first (EnhanceRequest).
var errNotEnhanced = errors.New("latent image: a request asking prompt mode 2 is enhanced (EnhanceRequest) before a picture session draws it")

// denoiseWhole is the denoise above which ComfyUI samples the whole schedule (KSampler.set_steps).
const denoiseWhole = 0.9999

// prompt is what the request's text encoder reads: its prompt, or the layout prompt its layout makes on its canvas.
func (r Request) prompt() string {
	if r.Layout == nil {
		return r.Prompt
	}
	return r.Layout.prompt(r.Width, r.Height)
}

// cutsSchedule is whether the request samples part of its schedule.
func (r Request) cutsSchedule() bool { return r.Denoise != 0 && r.Denoise <= denoiseWhole }

// Beside are the files a request's picture reads from its model's folder at root beyond the model's components, with
// what its preset fills from catalog (withPreset), in the order it reads them: the adapters it applies (adapterPath)
// and, for a depth control drawn from a picture, the depth estimator (sectionDepth), and for prompt mode 2 the files of
// its enhancer's folder, whose chat writes its prompt (enhanced-repeat, SQA-465), none where it holds no folder. The
// model's inventory holds none of them, so a run is keyed by their bytes beside the request (SQA-406). An adapter's
// name that is no file name is refused.
func (r Request) Beside(root string, catalog Catalog) ([]string, error) {
	filled, err := r.withPreset(catalog)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, applied := range filled.applied() {
		path, err := adapterPath(root, applied.Name)
		if err != nil {
			return nil, err
		}
		files = append(files, path)
	}
	if filled.Control != nil && filled.Control.Picture != "" {
		files = append(files, filepath.Join(root, depthDirectory, depthCheckpoint))
	}
	if r.Enhanced() {
		folder := filepath.Join(root, enhancerDirectory)
		entries, err := os.ReadDir(folder)
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return nil, err
		}
		for _, entry := range entries {
			if entry.Type().IsRegular() {
				files = append(files, filepath.Join(folder, entry.Name()))
			}
		}
	}
	return files, nil
}

// applied are the adapters the request applies, in its order: its Adapters, or its one Adapter at AdapterStrength.
func (r Request) applied() []AppliedAdapter {
	if r.Adapters != nil {
		return *r.Adapters
	}
	if r.Adapter != "" {
		return []AppliedAdapter{{Name: r.Adapter, Strength: r.AdapterStrength}}
	}
	return nil
}

// errDrawsItsOwnSchedule refuses a request's schedule on a profile that samples its model's own (a flow profile's
// dynamic shift): only a rotary profile draws a request's schedule.
var errDrawsItsOwnSchedule = errors.New("latent image: this model samples its own schedule; a request's schedule runs on a rotary profile (MOSS)")

// flowSchedule is a flow profile's schedule for a request, its model's own.
func flowSchedule(request Request) (FlowSchedule, error) {
	if request.Schedule != nil {
		return FlowSchedule{}, errDrawsItsOwnSchedule
	}
	return CompileFlowSchedule(request.Steps, request.NumTrainTimesteps, request.DynamicShiftMu)
}

type Generator struct {
	request   Request
	profile   Profile
	spec      *Spec
	modelDir  string
	tokenizer *hfbpe.Tokenizer
	pipeline  *ResidentImagePipeline
	rotary    *rotaryGeneration
	observe   func(string, []float32) error
	embed     []float32
	// pictures are what the session's requests read their pictures with, held across them (pictureSections).
	pictures *pictureSections
	// held lets go the session's ownership of its model's directory, whose files it reads beside the model (its
	// adapters, its depth checkpoint) it holds from its load to its close (heldfile.Session.Own, held-release).
	held func() error
	// negative is the negative prompt's programs, whose conditioning prepare makes for a guided denoiser (Guide).
	negative   *promptPrograms
	schedule   FlowSchedule
	shape      LatentShape
	prepared   bool
	integrated bool
	completed  bool
}

// admitPicture refuses a picture whose token activations alone exceed the card.
// Every token holds a row of hidden values on the device through each block, so
// this is the least such a picture needs, read from the card rather than a
// ceiling on the canvas. It runs before the host allocates the picture's noise
// and rotary tables, each as long as the picture: an allocation that fails ends
// the process, where this ends the request.
func admitPicture(ctx context.Context, session *graphruntime.ResidentSession, tokens, hidden int) error {
	return session.Do(ctx, func(state *device.State) error {
		_, total, err := state.Driver.MemInfo()
		if err != nil {
			return err
		}
		bytes, ok := checked.Bytes(uint64(tokens), uint64(hidden)*activationBytes)
		if !ok || bytes > total {
			return fmt.Errorf("latent image: a picture of %d tokens needs more than the %d bytes of this card", tokens, total)
		}
		return nil
	})
}

// Admit refuses a request its picture session would refuse, as it will reach the session: one asking prompt mode 2 as
// its enhancement leaves it (EnhanceRequest), the rest as they are (ValidateRequest). The image capability admits a
// request so before it reads a file the request names (SQA-425): a request with steps of -1 read and encoded its stored
// picture, and hashed its adapter, before its session refused it, and with its picture missing was refused for that.
func Admit(request Request) error {
	if request.Enhanced() {
		if request.Layout != nil {
			return errEnhanceBesideLayout
		}
		request.PromptMode = promptText
	}
	return ValidateRequest(request)
}

func ValidateRequest(request Request) error {
	if request.Enhanced() {
		if request.Layout != nil {
			return errEnhanceBesideLayout
		}
		return errNotEnhanced
	}
	// The builder's mode gives its prompt by its layout alone, and a layout gives the prompt of no other mode.
	if request.PromptMode == promptBuilder && request.Layout == nil {
		return errBuilderWithoutLayout
	}
	if request.Layout != nil && request.PromptMode != "" && request.PromptMode != promptBuilder {
		return fmt.Errorf("latent image: a layout gives the builder's prompt (prompt mode builder), not prompt mode %s", request.PromptMode)
	}
	if request.Layout != nil {
		if request.Prompt != "" {
			return errLayoutBesidePrompt
		}
		if err := request.Layout.validate(); err != nil {
			return err
		}
	}
	if err := media.ValidateSpatialGeometry(request.Height, request.Width); err != nil {
		return fmt.Errorf("latent image: invalid extent: %w", err)
	}
	if !checked.NonNegativeInts(request.Steps, request.NumTrainTimesteps) || !checked.Finite64(request.DynamicShiftMu) {
		return errors.New("latent image: invalid sampling policy")
	}
	if request.GuidanceScale != nil && !checked.NonNegativeFinite32(*request.GuidanceScale) {
		return errors.New("latent image: guidance must be finite and nonnegative")
	}
	if err := request.Guidance.Validate(); err != nil {
		return fmt.Errorf("latent image: %w", err)
	}
	if request.Schedule != nil {
		if err := request.Schedule.Validate(); err != nil {
			return fmt.Errorf("latent image: %w", err)
		}
	}
	if request.Sharpen != nil {
		if err := request.Sharpen.Admit(request.Height, request.Width); err != nil {
			return fmt.Errorf("latent image: %w", err)
		}
	}
	if request.AdapterStrength != nil && (request.Adapter == "" || !checked.Finite32(*request.AdapterStrength)) {
		return errors.New("latent image: an adapter's strength is a finite number beside the adapter it scales")
	}
	if request.Adapters != nil {
		if request.Adapter != "" || request.AdapterStrength != nil || len(*request.Adapters) == 0 {
			return errors.New("latent image: a request names its adapters or one adapter, not both, and a list names at least one")
		}
		for index, applied := range *request.Adapters {
			if applied.Name == "" || applied.Strength != nil && !checked.Finite32(*applied.Strength) {
				return fmt.Errorf("latent image: adapter %d needs a name and a finite strength", index+1)
			}
		}
	}
	if !(request.Denoise >= 0 && request.Denoise <= 1) {
		return fmt.Errorf("latent image: denoise %g is not a share of the schedule, above 0 and at most 1", request.Denoise)
	}
	if request.cutsSchedule() && request.Schedule != nil && request.Schedule.Name == "" && len(request.Schedule.Sigmas) != 0 {
		return errors.New("latent image: a request that lists its own sigmas samples them all; denoise cuts a drawn schedule")
	}
	if request.Control != nil {
		if err := request.Control.Validate(); err != nil {
			return err
		}
		// A request naming a preset takes the preset's adapters where it names none, and is checked again with them
		// (LoadGenerator, configureRotary): the picture director checks it before they are filled, and refused a
		// preset-only depth request (SQA-409).
		if len(request.applied()) == 0 && request.Preset == "" {
			return errors.New("latent image: a depth control is read through its control adapter; name it (moss-control)")
		}
	}
	return validatePrompt(request.PromptMode, request.Prompt, request.Width, request.Height)
}

// LoadGenerator loads a picture session of the model at modelDir for request, which owns the model's directory from its
// load to its close (heldfile.Session.Own): the files it reads beside the model are held while it lives (held-release).
func LoadGenerator(ctx context.Context, modelDir string, profile Profile, request Request) (*Generator, error) {
	release, err := heldfile.Session.Own(modelDir)
	if err != nil {
		return nil, err
	}
	generator, err := loadGenerator(ctx, modelDir, profile, request)
	if err != nil {
		return nil, errors.Join(err, release())
	}
	generator.held = release
	return generator, nil
}

// loadGenerator is LoadGenerator's load, its model's directory owned.
func loadGenerator(ctx context.Context, modelDir string, profile Profile, request Request) (*Generator, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	request, err := request.withPreset(CatalogIn(modelDir))
	if err != nil {
		return nil, err
	}
	if err := ValidateRequest(request); err != nil {
		return nil, err
	}
	if err := profile.validateIdentity(); err != nil {
		return nil, err
	}
	if profile.Rotary != nil {
		return loadRotaryGenerator(ctx, modelDir, profile, request)
	}
	if request.rotaryOnly() {
		return nil, errRotaryOnly
	}
	if !checked.Nonzero(request.Prompt) {
		return nil, errEmptyPrompt
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
	prompt, err := generator.compilePrompt(request, gridH, gridW)
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
	if err := admitPicture(ctx, pipeline.runtime, gridH*gridW, spec.Transformer.Hidden); err != nil {
		return nil, errors.Join(err, pipeline.Close(context.WithoutCancel(ctx)))
	}
	schedule, err := flowSchedule(request)
	if err != nil {
		_ = pipeline.Close(ctx)
		return nil, err
	}
	generator.pipeline, generator.embed, generator.negative, generator.schedule = pipeline, prompt.embed, prompt.negative, schedule
	return generator, nil
}

// promptPrograms is what a prompt gives a session: the rows of its tokens'
// embeddings, and the text encoder, fusion and denoiser compiled under its
// padding mask. Every prompt is padded to one length, so the three are of one
// shape whatever the prompt, and differ in what the mask leaves unattended.
// mask and weights are the prompt's rows as the denoiser reads them, and
// negative the negative prompt's text programs where the denoiser guides by one.
type promptPrograms struct {
	embed    []float32
	encoder  *EncoderProgram
	fusion   *FusionProgram
	denoiser *DenoiserProgram
	mask     []bool
	weights  []float32
	negative *promptPrograms
}

// compilePrompt renders a request's prompt, and the negative prompt it guides
// by (Request.guide), as the model's text template has them, and compiles the
// programs that read their masks, for a picture of gridH by gridW patches.
func (g *Generator) compilePrompt(request Request, gridH, gridW int) (promptPrograms, error) {
	prompt, err := g.compileText(request.Prompt)
	if err != nil {
		return promptPrograms{}, err
	}
	var guides []NegativeGuide
	if guide := request.guide(); guide.negative != "" {
		negative, err := g.compileText(guide.negative)
		if err != nil {
			return promptPrograms{}, err
		}
		prompt.negative, guides = &negative, []NegativeGuide{{Mask: negative.mask, Weights: negative.weights, Attention: guide.attention}}
	}
	prompt.denoiser, err = CompileDenoiserProgram(
		g.spec.Transformer, float32(g.spec.Transformer.NormEps), prompt.mask, prompt.weights, gridH, gridW, dtype.BF16,
		request.EightBitAttention, guides...,
	)
	return prompt, err
}

// compileText renders a prompt as the model's text template has it and
// compiles the text encoder and the fusion under its padding mask.
func (g *Generator) compileText(prompt string) (promptPrograms, error) {
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
	// The denoiser reads the prompt's rows, as the mask, and their weights where the prompt weights any.
	var textWeights []float32
	if text.Weights != nil {
		textWeights = text.Weights[len(text.Weights)-text.PromptRows:]
	}
	return promptPrograms{embed: embed, encoder: encoder, fusion: fusion, mask: textMask, weights: textWeights}, nil
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
	if g == nil {
		return errors.New("latent image: resident session is unavailable")
	}
	request, err := request.withPreset(CatalogIn(g.modelDir))
	if err != nil {
		return err
	}
	if g.rotary != nil {
		return g.resetRotary(ctx, request)
	}
	if request.rotaryOnly() {
		return errRotaryOnly
	}
	if !checked.Nonzero(request.Prompt) {
		return errEmptyPrompt
	}
	if g == nil || g.pipeline == nil || !g.completed {
		return errors.New("latent image: resident session is unavailable")
	}
	if err := ValidateRequest(request); err != nil {
		return err
	}
	gridH, gridW := g.pipeline.Denoiser.GH, g.pipeline.Denoiser.GW
	request = request.withPolicy(g.profile.Sampling, gridH*gridW, g.spec.Distilled)
	// A session's programs are compiled for a picture's extent, and encode each
	// step's timestep at the scale of the training timesteps it was loaded
	// for: a request of another extent or another count would be drawn with a
	// new schedule and the old encoding. It is refused here, before anything
	// of the session changes, and whoever holds the session loads one for it.
	if request.Width != g.request.Width || request.Height != g.request.Height || request.NumTrainTimesteps != g.request.NumTrainTimesteps {
		return errors.New("latent image: request requires another resident session")
	}
	// The denoiser compiled with its attention, 8-bit or not, is recompiled with the prompt for a request asking the other.
	if request.Prompt != g.request.Prompt || request.guide() != g.request.guide() || request.EightBitAttention != g.request.EightBitAttention {
		prompt, err := g.compilePrompt(request, gridH, gridW)
		if err != nil {
			return err
		}
		if err := g.pipeline.Recondition(ctx, request.Prompt, prompt.encoder, prompt.fusion, prompt.denoiser, prompt.embed); err != nil {
			return err
		}
		g.embed, g.negative = prompt.embed, prompt.negative
	}
	schedule, err := flowSchedule(request)
	if err != nil {
		return err
	}
	g.request, g.schedule = request, schedule
	g.prepared, g.integrated, g.completed = false, false, false
	return nil
}

// prepare, integrate and decode are a picture request's three stages, each named in its ledger (processmeasure.Stage):
// its prompt encoded and its noise drawn, its latent denoised, and its picture decoded.
func (g *Generator) prepare(ctx context.Context, request Request) (*Generator, error) {
	defer processmeasure.Stage(ctx, "picture prompt encoded")()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if g == nil {
		return nil, errors.New("latent image: generation session is unavailable")
	}
	request, err := request.withPreset(CatalogIn(g.modelDir))
	if err != nil {
		return nil, err
	}
	if g.rotary != nil {
		return g.prepareRotary(ctx, request)
	}
	if g == nil || g.pipeline == nil || request.withPolicy(g.profile.Sampling, g.pipeline.Denoiser.GH*g.pipeline.Denoiser.GW, g.spec.Distilled) != g.request || g.prepared {
		return nil, errors.New("latent image: generation session is unavailable")
	}
	if g.pipeline.conditioning == nil {
		if _, err := g.pipeline.Condition(ctx, g.request.Prompt, g.embed); err != nil {
			return nil, err
		}
	}
	if g.negative != nil {
		if err := g.pipeline.Guide(ctx, g.request.NegativePrompt, g.negative.encoder, g.negative.fusion, g.negative.embed); err != nil {
			return nil, err
		}
		g.negative = nil
	}
	var latent []float32
	err = g.pipeline.runtime.Do(ctx, func(state *device.State) error {
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
	defer processmeasure.Stage(ctx, "picture denoised")()
	if g == nil || session != g || !g.prepared || g.integrated {
		return nil, errors.New("latent image: integration session is unavailable")
	}
	if g.rotary != nil {
		return g.integrateRotary(ctx)
	}
	// The device runs each Euler update itself, so it is given the walk's knot and delta.
	sampler, err := g.schedule.sampler()
	if err != nil {
		return nil, err
	}
	sigmas := sampler.Sigmas()
	for step := range len(sigmas) - 1 {
		if err := g.pipeline.Advance(ctx, sigmas[step], float64(sampler.Delta(step))); err != nil {
			return nil, err
		}
	}
	g.integrated = true
	return g, nil
}

func (g *Generator) decode(ctx context.Context, session *Generator) (EncodedImage, error) {
	defer processmeasure.Stage(ctx, "picture decoded")()
	if g == nil || session != g || !g.integrated {
		return EncodedImage{}, errors.New("latent image: decode session is unavailable")
	}
	if g.rotary != nil {
		return g.decodeRotary(ctx)
	}
	pixels, height, width, err := g.pipeline.DecodeHWC(ctx)
	if err != nil {
		return EncodedImage{}, err
	}
	if err := g.pipeline.Finish(ctx); err != nil {
		return EncodedImage{}, err
	}
	g.completed = true
	image, err := encodePNG(pixels, height, width)
	image.Preset = media.Preset{Width: g.request.Width, Height: g.request.Height, Steps: g.request.Steps, Sampler: "flow Euler",
		Shift: math.Exp(g.request.DynamicShiftMu)}
	return image, err
}

func (g *Generator) Close(ctx context.Context) error {
	if g == nil {
		return nil
	}
	// The files the session held beside its model (its adapters, its depth checkpoint) go with it, so they may be
	// replaced and the next session reads them again (adapter-hold).
	pictures := g.pictures.close(context.WithoutCancel(ctx))
	if g.held != nil {
		pictures = errors.Join(pictures, g.held())
	}
	g.pictures, g.held = nil, nil
	if g.rotary != nil {
		return errors.Join(pictures, g.closeRotary(ctx))
	}
	if g.pipeline == nil {
		return pictures
	}
	err := g.pipeline.Close(context.WithoutCancel(ctx))
	g.pipeline = nil
	g.embed = nil
	return errors.Join(pictures, err)
}

// PeakDeviceBytes is the most the session's device allocations have held.
func (g *Generator) PeakDeviceBytes() (uint64, error) {
	var session *graphruntime.ResidentSession
	switch {
	case g == nil:
		return 0, errors.New("latent image: resident session is unavailable")
	case g.rotary != nil:
		session = g.rotary.runtime
	case g.pipeline != nil:
		session = g.pipeline.runtime
	}
	stats, err := session.Stats(context.Background())
	return stats.Device.PeakBytes, err
}

func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generator *Generator) error {
	if generator == nil || (generator.pipeline == nil && generator.rotary == nil) {
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

// PresetSettings are the settings each preset of the model in directory fills in a request that sets nothing, as a
// request states them (withPreset), each a JSON object of the fields it sets, so a page choosing a preset shows what it
// fills: its steps, guidance and schedule, and a MOSS Ultra section's denoise, adapters, size, sharpening, skipped
// blocks and backup negative.
func PresetSettings(directory string) (map[string]json.RawMessage, error) {
	names, err := PresetNames(directory)
	if err != nil {
		return nil, err
	}
	fields := func(request Request) (map[string]json.RawMessage, error) {
		var held map[string]json.RawMessage
		written, err := json.Marshal(request)
		if err == nil {
			err = json.Unmarshal(written, &held)
		}
		return held, err
	}
	unset, err := fields(Request{})
	if err != nil {
		return nil, err
	}
	settings := make(map[string]json.RawMessage, len(names))
	for _, name := range names {
		filled, err := Request{Preset: name}.withPreset(CatalogIn(directory))
		if err != nil {
			return nil, err
		}
		set, err := fields(filled)
		if err != nil {
			return nil, err
		}
		maps.DeleteFunc(set, func(field string, value json.RawMessage) bool { return bytes.Equal(value, unset[field]) })
		if settings[name], err = json.Marshal(set); err != nil {
			return nil, err
		}
	}
	return settings, nil
}

// Sized is the request with the size its canvas, or its preset's, gives it where it states no width and height (a MOSS
// Ultra section's 16:9 at 3 megapixels), the rest as it came, its preset still named for what it fills besides; a
// request that states its size, or names neither canvas, is as it came. A session is chosen and checked by its size
// before its model loads (ValidateRequest, SessionKey), so a size the preset gives is given before (SQA-409). The
// preset is read from catalog.
func (r Request) Sized(catalog Catalog) (Request, error) {
	if r.Width != 0 || r.Height != 0 || r.Canvas == nil && r.Preset == "" {
		return r, nil
	}
	filled, err := r.withPreset(catalog)
	if err != nil {
		return Request{}, err
	}
	r.Width, r.Height, r.Canvas = filled.Width, filled.Height, nil
	return r, nil
}

// withPreset fills what a request leaves unset from the preset it names in its model's catalog, and clears the name,
// so a request naming a preset and one stating its values are one request. A request naming none is the authors'
// recipe, its profile's.
func (r Request) withPreset(catalog Catalog) (Request, error) {
	if r.Preset != "" {
		preset, err := presetOf(catalog, r.Preset)
		if err != nil {
			return Request{}, err
		}
		if r.Steps == 0 {
			r.Steps = preset.Steps
		}
		if r.GuidanceScale == nil {
			r.GuidanceScale = preset.GuidanceScale
		}
		if r.Guidance == (sampling.Guidance{}) {
			r.Guidance = preset.Guidance
		}
		r.EightBitAttention = r.EightBitAttention || preset.EightBitAttention
		if r, err = r.withSection(preset); err != nil {
			return Request{}, err
		}
		r.Preset = ""
	}
	// A canvas, the request's or its preset's, sizes the picture.
	if r.Canvas != nil {
		if r.Width != 0 || r.Height != 0 {
			return Request{}, errors.New("latent image: a request gives its width and height or its canvas, not both")
		}
		var err error
		if r.Width, r.Height, err = r.Canvas.Size(); err != nil {
			return Request{}, fmt.Errorf("latent image: %w", err)
		}
		r.Canvas = nil
	}
	// A distilled model honours a negative prompt given it by its normalized attention guidance, where the request
	// states no guidance of its own (catalogProfile.NegativeGuidance).
	if r.NegativePrompt != "" && r.Guidance == (sampling.Guidance{}) {
		entry, err := catalog.entry()
		if err != nil {
			return Request{}, err
		}
		if entry.NegativeGuidance != nil {
			r.Guidance = sampling.Guidance{Attention: *entry.NegativeGuidance}
		}
	}
	return r, nil
}

// withSection fills what the request leaves unset of a MOSS Ultra section's preset (imagePreset): its schedule, its
// denoise, its adapters where the request names none, its canvas where it gives no size, its sharpening, its skipped
// blocks and its depth control's end, each the preset's own value, the catalog's, so a request filled twice is one; and
// its negative the workflow's backup where the request's is empty or a none (isNoNegative).
func (r Request) withSection(preset imagePreset) (Request, error) {
	if r.Schedule == nil {
		r.Schedule = preset.Schedule
	}
	if r.Denoise == 0 {
		r.Denoise = preset.Denoise
	}
	if r.Adapter == "" && r.Adapters == nil {
		r.Adapters = preset.Adapters
	}
	if r.Canvas == nil && r.Width == 0 && r.Height == 0 {
		r.Canvas = preset.Canvas
	}
	if r.Sharpen == nil {
		r.Sharpen = preset.Sharpen
	}
	if r.SkipBlocks == nil {
		r.SkipBlocks = preset.SkipBlocks
	}
	if r.Control != nil && r.Control.End == nil && preset.ControlEnd != nil {
		control := *r.Control
		control.End = preset.ControlEnd
		r.Control = &control
	}
	if preset.BackupNegative && isNoNegative(r.NegativePrompt) {
		data, err := mossEnhancer()
		if err != nil {
			return Request{}, err
		}
		r.NegativePrompt = data.NegativeBackup
	}
	return r, nil
}

// errEmptyPrompt refuses a flow profile's request of no prompt. A rotary profile (MOSS) conditions on an empty prompt
// as ComfyUI's encoder does, as the MOSS Ultra workflow saves its sections and its recordings draw them
// (moss-ultra-compare/golden-floor).
var errEmptyPrompt = errors.New("latent image: prompt is empty")

// errRotaryOnly refuses a request that asks what only a rotary profile runs (rotaryOnly) of a profile that is not one.
var errRotaryOnly = errors.New("latent image: selected profile does not support negative prompts, guidance overrides, layouts, adapters or image to image")

// rotaryOnly is whether the request asks what only a rotary profile runs: a negative prompt or guidance over its
// steps, except normalized attention guidance's (attentionGuided), a guidance scale of its own, layout validation or
// a layout, a sharpening of its picture, an adapter, a picture or denoise to sample from, a depth control, or the blocks
// its negative pass skips.
func (r Request) rotaryOnly() bool {
	return !r.attentionGuided() && (r.NegativePrompt != "" || r.Guidance != (sampling.Guidance{})) ||
		r.GuidanceScale != nil || r.PromptMode == promptLayout || r.Layout != nil || r.Sharpen != nil || r.Adapter != "" || r.Adapters != nil ||
		r.AdapterStrength != nil || r.Image != "" || r.Denoise != 0 || r.Control != nil || r.SkipBlocks != nil
}

// attentionGuided says the request's guidance is normalized attention guidance's alone, which a Krea session runs by
// the negative prompt where the request gives one (NegativeGuide), and as it would unguided where not.
func (r Request) attentionGuided() bool {
	return r.Guidance.Attention.On() && r.Guidance == (sampling.Guidance{Attention: r.Guidance.Attention})
}

// attentionGuide is what a session's denoiser is compiled to guide by: a negative prompt and the attention guidance
// that weighs it. The zero guide guides by nothing.
type attentionGuide struct {
	negative  string
	attention sampling.NormalizedAttention
}

// guide is the request's attention guide: its negative prompt where it guides by normalized attention and gives one.
func (r Request) guide() attentionGuide {
	if r.NegativePrompt == "" || !r.attentionGuided() {
		return attentionGuide{}
	}
	return attentionGuide{negative: r.NegativePrompt, attention: r.Guidance.Attention}
}
