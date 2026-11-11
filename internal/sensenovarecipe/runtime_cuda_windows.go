//go:build windows

package sensenovarecipe

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	"overgo/internal/hfbpe"
	"overgo/internal/latentimage"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/torchrng"
	"overgo/internal/workflowruntime"
)

type GenerationRequest struct {
	Prompt        string  `json:"prompt"`
	Width         int     `json:"width"`
	Height        int     `json:"height"`
	Steps         int     `json:"steps"`
	Seed          int64   `json:"seed"`
	CFGScale      float32 `json:"cfg_scale"`
	TimestepShift float64 `json:"timestep_shift"`
	// Preset names one of SenseNova's presets (PresetNames), its guidance varying over the steps; none is the authors'
	// recipe.
	Preset string `json:"preset,omitzero"`
}

// GenerationStats partitions one production request without synchronizing
// inside device stages.
type GenerationStats struct {
	Prepare, Vision, Condition, Body, Flow, Guidance, Decode time.Duration
	Steps                                                    int
	// Passes are the branches the body ran over the steps: two a guided step, one a step its guidance reads no
	// unconditioned output of, none a step it zeroes.
	Passes int
}

func ValidateGenerationRequest(request GenerationRequest) error {
	if request.Prompt == "" || request.Width <= 0 || request.Height <= 0 || request.Steps <= 0 ||
		request.CFGScale < 0 || math.IsNaN(float64(request.CFGScale)) || math.IsInf(float64(request.CFGScale), 0) ||
		request.TimestepShift <= 0 || math.IsNaN(request.TimestepShift) || math.IsInf(request.TimestepShift, 0) {
		return errors.New("sensenova recipe: invalid generation request")
	}
	_, err := guidanceOf(request.Preset)
	return err
}

// SessionKey identifies reusable resident state for the request.
func (request GenerationRequest) SessionKey() (string, error) {
	prompt := sha256.Sum256([]byte(request.Prompt))
	return fmt.Sprintf("%dx%d/%d/%x", request.Width, request.Height, request.Steps, prompt), nil
}

type generationPlan struct {
	request GenerationRequest
	// guidance is how the request's preset varies the guidance over the steps.
	guidance sampling.Guidance
	image    routedlm.FlowImagePlan
	schedule []float64
	z        []float32
	session  *routedlm.DeviceGenerationSession
	flow     *routedlm.DeviceFlowPeripheralSession
}

type generationFeatures struct {
	patches []float32
	image   routedlm.FlowImagePlan
	flow    routedlm.FlowPlan
}

type Generator struct {
	source       *safetensors.Source
	worker       *device.Worker
	cuda         *executor.Executor
	tokenizer    *hfbpe.Tokenizer
	config       routedlm.Config
	flow         routedlm.FlowPlan
	rope         routedlm.RopePlan
	binding      routedlm.BranchBinding
	vision       routedlm.VisionEmbedderWeights
	flowTerminal routedlm.FlowTerminalWeights
	terminal     routedlm.TerminalWeights
	active       *routedlm.DeviceGenerationSession
	activeFlow   *routedlm.DeviceFlowPeripheralSession
	stats        GenerationStats
}

func LoadGenerator(
	ctx context.Context,
	store artifact.Reader,
	modelDir string,
	definition recipe.Definition,
) (_ *Generator, result error) {
	flowProfile, err := routedlm.ResolveFlowProfile(ctx, store, definition)
	if err != nil {
		return nil, err
	}
	binding := routedlm.SenseNovaBinding()
	config, err := routedlm.LoadConfig(modelDir, binding)
	if err != nil {
		return nil, err
	}
	flowConfig, err := routedlm.LoadFlowConfig(modelDir)
	if err != nil {
		return nil, err
	}
	source, err := safetensors.OpenSource(modelDir)
	if err != nil {
		return nil, err
	}
	generator := &Generator{source: source, config: config, binding: binding}
	defer func() {
		if result != nil {
			_ = generator.Close(context.Background())
		}
	}()
	if generator.rope, err = routedlm.CompileRopePlan(source, config, binding); err != nil {
		return nil, err
	}
	if generator.flow, err = routedlm.CompileFlowPlan(source, config, flowConfig, flowProfile); err != nil {
		return nil, err
	}
	if generator.vision, err = routedlm.LoadVisionEmbedderWeights(source, flowProfile.GenerationVisionPrefix, generator.flow); err != nil {
		return nil, err
	}
	if generator.flowTerminal, err = routedlm.LoadFlowTerminalWeights(source, generator.flow, flowProfile); err != nil {
		return nil, err
	}
	if generator.terminal, err = routedlm.LoadTerminalWeights(source, config, binding); err != nil {
		return nil, err
	}
	if generator.tokenizer, err = hfbpe.LoadSplit(modelDir); err != nil {
		return nil, err
	}
	if generator.worker, err = device.New(0); err != nil {
		return nil, err
	}
	if generator.cuda, err = executor.NewWithWorker(generator.worker); err != nil {
		return nil, err
	}
	return generator, nil
}

func (g *Generator) Reset(ctx context.Context, request GenerationRequest) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if g == nil || g.source == nil || g.worker == nil || g.cuda == nil {
		return errors.New("sensenova recipe: incomplete generator")
	}
	if err := ValidateGenerationRequest(request); err != nil {
		return err
	}
	if g.active != nil {
		if err := g.active.Close(ctx); err != nil {
			return err
		}
		g.active = nil
	}
	if g.activeFlow != nil {
		if err := g.activeFlow.Close(ctx); err != nil {
			return err
		}
		g.activeFlow = nil
	}
	return nil
}

func (g *Generator) prepare(ctx context.Context, request GenerationRequest) (*generationPlan, error) {
	started := time.Now()
	g.stats = GenerationStats{Steps: request.Steps}
	defer func() { g.stats.Prepare = time.Since(started) }()
	if err := g.Reset(ctx, request); err != nil {
		return nil, err
	}
	guidance, err := guidanceOf(request.Preset)
	if err != nil {
		return nil, err
	}
	image, err := g.flow.ImagePlan(request.Width, request.Height)
	if err != nil {
		return nil, err
	}
	schedule, err := routedlm.ShiftedFlowTimeSchedule(request.Steps, request.TimestepShift)
	if err != nil {
		return nil, err
	}
	template := routedlm.SenseNovaPromptTemplate()
	conditional, err := routedlm.RenderEditPrompt(g.tokenizer, request.Prompt, routedlm.PromptRolePrompt, template, routedlm.PromptSource{})
	if err != nil {
		return nil, err
	}
	unconditional, err := routedlm.RenderEditPrompt(g.tokenizer, "", routedlm.PromptRoleUnconditional, template, routedlm.PromptSource{})
	if err != nil {
		return nil, err
	}
	prefixes, _, err := routedlm.RunDevicePrefixStacks(
		ctx, g.worker, g.cuda, g.source, g.config, g.binding, g.rope, nil,
		routedlm.DevicePrefixInput{TokenIDs: conditional.IDs, ImageTime: len(conditional.IDs)},
		routedlm.DevicePrefixInput{TokenIDs: unconditional.IDs, ImageTime: len(unconditional.IDs)},
	)
	if err != nil {
		return nil, err
	}
	session, err := routedlm.NewDeviceGenerationSession(
		ctx, g.worker, g.cuda, g.source, g.config, g.binding, g.rope,
		image, prefixes[0], prefixes[1],
	)
	if err != nil {
		return nil, err
	}
	g.active = session
	flow, err := routedlm.NewDeviceFlowPeripheralSession(
		ctx, g.worker, g.cuda, g.flow, image, g.vision, g.flowTerminal,
		g.terminal.FinalNorm[1], g.config.RMSNormEps,
	)
	if err != nil {
		_ = session.Close(context.WithoutCancel(ctx))
		g.active = nil
		return nil, err
	}
	g.activeFlow = flow
	var z []float32
	if err := g.worker.Do(ctx, func(state *device.State) error {
		stream := torchrng.NewStream(request.Seed)
		defer stream.Close(state)
		var seedErr error
		z, seedErr = routedlm.SeededFlowLatent(stream, state, g.flow, image)
		return seedErr
	}); err != nil {
		_ = flow.Close(context.WithoutCancel(ctx))
		_ = session.Close(context.WithoutCancel(ctx))
		g.active, g.activeFlow = nil, nil
		return nil, err
	}
	return &generationPlan{request: request, guidance: guidance, image: image, schedule: schedule, z: z, session: session, flow: flow}, nil
}

func (g *Generator) integrate(ctx context.Context, plan *generationPlan) (generationFeatures, error) {
	if plan == nil || plan.session == nil || plan.session != g.active || plan.flow == nil || plan.flow != g.activeFlow || len(plan.schedule) != plan.request.Steps+1 {
		return generationFeatures{}, errors.New("sensenova recipe: invalid generation plan")
	}
	defer func() {
		_ = plan.flow.Close(context.WithoutCancel(ctx))
		_ = plan.session.Close(context.WithoutCancel(ctx))
		plan.session, plan.flow, g.active, g.activeFlow = nil, nil, nil, nil
	}()
	z := plan.z
	guider := sampling.Guider{Scale: float64(plan.request.CFGScale), Guidance: plan.guidance}
	// APG carries its momentum over the generation's steps.
	adaptive := &sampling.AdaptiveGuider{Adaptive: plan.guidance.Adaptive, Scale: float64(plan.request.CFGScale)}
	for step := range plan.request.Steps {
		if err := ctx.Err(); err != nil {
			return generationFeatures{}, err
		}
		timestep := plan.schedule[step]
		// A step the guidance zeroes gives no velocity: the latent stays, and neither branch runs.
		at := guider.At(step, imageTime-timestep)
		if at.Zeroed() {
			continue
		}
		started := time.Now()
		planar, err := media.UnpackPlanar(
			z, g.flow.VisionChannels, plan.image.TokenHeight, plan.image.TokenWidth,
			plan.image.TokenPatch, media.PatchChannelsLast,
		)
		if err != nil {
			return generationFeatures{}, err
		}
		hidden, err := plan.flow.VisionEmbedTokens(ctx, planar)
		if err != nil {
			return generationFeatures{}, fmt.Errorf("sensenova recipe: step %d vision embedding: %w", step, err)
		}
		g.stats.Vision += time.Since(started)
		started = time.Now()
		condition, err := plan.flow.FlowConditionRow(
			ctx, timestep, g.flow.NormalizedNoiseScale(plan.image.NoiseScale),
		)
		if err != nil {
			return generationFeatures{}, fmt.Errorf("sensenova recipe: step %d condition: %w", step, err)
		}
		if err := routedlm.AddConditionRows(hidden, condition); err != nil {
			return generationFeatures{}, err
		}
		g.stats.Condition += time.Since(started)
		started = time.Now()
		var branches [][]float32
		if at.Unconditional() {
			branches, _, err = plan.session.Run(ctx, hidden, nil, nil)
		} else {
			// A step outside the guidance interval reads no unconditioned output, so its branch does not run.
			branches, _, err = plan.session.RunFirst(ctx, hidden)
		}
		if err != nil {
			return generationFeatures{}, fmt.Errorf("sensenova recipe: step %d body: %w", step, err)
		}
		g.stats.Body += time.Since(started)
		g.stats.Passes += len(branches)
		started = time.Now()
		velocities := make([][]float32, len(branches))
		for branch := range branches {
			velocities[branch], err = plan.flow.FlowHeadVelocity(ctx, branches[branch], z, timestep)
			if err != nil {
				return generationFeatures{}, fmt.Errorf("sensenova recipe: step %d branch %d flow head: %w", step, branch, err)
			}
		}
		g.stats.Flow += time.Since(started)
		started = time.Now()
		guided, err := guidedVelocity(at, adaptive, plan.request.CFGScale, velocities, z, imageTime-timestep)
		if err != nil {
			return generationFeatures{}, err
		}
		if err := routedlm.FlowEulerStep(z, guided, float32(plan.schedule[step+1]-timestep)); err != nil {
			return generationFeatures{}, err
		}
		g.stats.Guidance += time.Since(started)
	}
	return generationFeatures{patches: z, image: plan.image, flow: g.flow}, nil
}

// SenseNova's time runs from noise at zero to the image at imageTime, so a step's noise level is imageTime less its time.
const imageTime = 1

// A guided step runs the conditioned and the unconditioned branch, in that order (prepare's prefixes); a step that reads
// no unconditioned output runs the first alone, conditionedBranch of them.
const conditionedBranch = 1

// guidedVelocity is the velocity a step steps by: the authors' combine of the two branches, the conditioned branch's alone
// where the guider reads no unconditioned output, CFG-Zero*'s projected combine (sampling.Guider), or APG's, which denoises the
// sample at the step's noise level (sampling.AdaptiveGuider), each stored as BF16 as the authors' is.
func guidedVelocity(at sampling.Guider, adaptive *sampling.AdaptiveGuider, scale float32, velocities [][]float32, sample []float32, noise float64) ([]float32, error) {
	switch {
	case !at.Unconditional():
		return routedlm.GuidedFlowVelocity(velocities[:conditionedBranch], []float32{tensor.UnitScale})
	case at.Guidance.Projected, at.Guidance.Adaptive.On():
		guided := make([]float32, len(velocities[0]))
		combined := at.Combine
		if at.Guidance.Adaptive.On() {
			// SenseNova's velocity points to the image: it denoises the sample to the sample plus the noise level times it.
			combined = func(out, conditional, unconditional []float32) error {
				return adaptive.Combine(out, conditional, unconditional, sample, -noise)
			}
		}
		if err := combined(guided, velocities[0], velocities[1]); err != nil {
			return nil, err
		}
		dtype.RoundBF16Slice(guided)
		return guided, nil
	default:
		return routedlm.GuidedFlowVelocity(velocities, []float32{scale, 1 - scale})
	}
}

func (g *Generator) decode(ctx context.Context, features generationFeatures) (latentimage.EncodedImage, error) {
	if err := ctx.Err(); err != nil {
		return latentimage.EncodedImage{}, err
	}
	started := time.Now()
	image, err := DecodeGeneratedImage(features.patches, features.flow, features.image)
	g.stats.Decode += time.Since(started)
	return image, err
}

// Stats returns the last completed or failed request's measured phases.
func (g *Generator) Stats() GenerationStats { return g.stats }

func (g *Generator) Close(ctx context.Context) error {
	if g == nil {
		return nil
	}
	ctx = context.WithoutCancel(ctx)
	var result error
	if g.active != nil {
		result = errors.Join(result, g.active.Close(ctx))
		g.active = nil
	}
	if g.activeFlow != nil {
		result = errors.Join(result, g.activeFlow.Close(ctx))
		g.activeFlow = nil
	}
	if g.cuda != nil {
		result = errors.Join(result, g.cuda.Close())
		g.cuda = nil
	}
	if g.worker != nil {
		result = errors.Join(result, g.worker.Close())
		g.worker = nil
	}
	if g.source != nil {
		result = errors.Join(result, g.source.Close())
		g.source = nil
	}
	return result
}

func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generator *Generator) error {
	if generator == nil {
		return errors.New("sensenova recipe: incomplete runtime binding")
	}
	return workflowruntime.RegisterContextPipeline(
		runtime, modelID,
		modelrecipe.ModuleRoutedImagePrepare, generator.prepare,
		modelrecipe.ModuleRoutedImageIntegrate, generator.integrate,
		modelrecipe.ModuleRoutedImageDecode, generator.decode,
		latentimage.PNGContent,
	)
}
