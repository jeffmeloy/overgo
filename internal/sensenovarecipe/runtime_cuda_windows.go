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
	"overgo/internal/modelrecipe"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
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
}

// GenerationStats partitions one production request without synchronizing
// inside device stages.
type GenerationStats struct {
	Prepare, Vision, Condition, Body, Flow, Guidance, Decode time.Duration
	Steps                                                    int
}

func ValidateGenerationRequest(request GenerationRequest) error {
	if request.Prompt == "" || request.Width <= 0 || request.Height <= 0 || request.Steps <= 0 ||
		request.CFGScale < 0 || math.IsNaN(float64(request.CFGScale)) || math.IsInf(float64(request.CFGScale), 0) ||
		request.TimestepShift <= 0 || math.IsNaN(request.TimestepShift) || math.IsInf(request.TimestepShift, 0) {
		return errors.New("sensenova recipe: invalid generation request")
	}
	return nil
}

func GenerationSessionPolicy(request GenerationRequest) (string, error) {
	if err := ValidateGenerationRequest(request); err != nil {
		return "", err
	}
	prompt := sha256.Sum256([]byte(request.Prompt))
	return fmt.Sprintf("%dx%d/%d/%x", request.Width, request.Height, request.Steps, prompt), nil
}

type generationPlan struct {
	request  GenerationRequest
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

func LoadGenerator(modelDir string) (_ *Generator, result error) {
	binding, flowBinding := routedlm.SenseNovaBinding(), routedlm.SenseNovaFlowBinding()
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
	if generator.flow, err = routedlm.CompileFlowPlan(source, config, flowConfig, flowBinding); err != nil {
		return nil, err
	}
	if generator.vision, err = routedlm.LoadVisionEmbedderWeights(source, flowBinding.GenerationVisionPrefix, generator.flow); err != nil {
		return nil, err
	}
	if generator.flowTerminal, err = routedlm.LoadFlowTerminalWeights(source, generator.flow, flowBinding); err != nil {
		return nil, err
	}
	if generator.terminal, err = routedlm.LoadTerminalWeights(source, config, binding); err != nil {
		return nil, err
	}
	if generator.tokenizer, err = hfbpe.LoadLegacy(modelDir); err != nil {
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

func (g *Generator) prepare(request GenerationRequest) (*generationPlan, error) {
	started := time.Now()
	g.stats = GenerationStats{Steps: request.Steps}
	defer func() { g.stats.Prepare = time.Since(started) }()
	if err := g.Reset(context.Background(), request); err != nil {
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
		context.Background(), g.worker, g.cuda, g.source, g.config, g.binding, g.rope, nil,
		routedlm.DevicePrefixInput{TokenIDs: conditional.IDs, ImageTime: len(conditional.IDs)},
		routedlm.DevicePrefixInput{TokenIDs: unconditional.IDs, ImageTime: len(unconditional.IDs)},
	)
	if err != nil {
		return nil, err
	}
	session, err := routedlm.NewDeviceGenerationSession(
		context.Background(), g.worker, g.cuda, g.source, g.config, g.binding, g.rope,
		image, prefixes[0], prefixes[1],
	)
	if err != nil {
		return nil, err
	}
	g.active = session
	flow, err := routedlm.NewDeviceFlowPeripheralSession(
		context.Background(), g.worker, g.cuda, g.flow, image, g.vision, g.flowTerminal,
		g.terminal.FinalNorm[1], g.config.RMSNormEps,
	)
	if err != nil {
		_ = session.Close(context.Background())
		g.active = nil
		return nil, err
	}
	g.activeFlow = flow
	var z []float32
	if err := g.worker.Do(context.Background(), func(state *device.State) error {
		stream := torchrng.NewStream(request.Seed)
		defer stream.Close(state)
		var seedErr error
		z, seedErr = routedlm.SeededFlowLatent(stream, state, g.flow, image)
		return seedErr
	}); err != nil {
		_ = flow.Close(context.Background())
		_ = session.Close(context.Background())
		g.active, g.activeFlow = nil, nil
		return nil, err
	}
	return &generationPlan{request: request, image: image, schedule: schedule, z: z, session: session, flow: flow}, nil
}

func (g *Generator) integrate(plan *generationPlan) (generationFeatures, error) {
	if plan == nil || plan.session == nil || plan.session != g.active || plan.flow == nil || plan.flow != g.activeFlow || len(plan.schedule) != plan.request.Steps+1 {
		return generationFeatures{}, errors.New("sensenova recipe: invalid generation plan")
	}
	defer func() {
		_ = plan.flow.Close(context.Background())
		_ = plan.session.Close(context.Background())
		plan.session, plan.flow, g.active, g.activeFlow = nil, nil, nil, nil
	}()
	z := plan.z
	for step := range plan.request.Steps {
		started := time.Now()
		planar, err := latentimage.UnpackPlanarF32(
			z, g.flow.VisionChannels, plan.image.TokenHeight, plan.image.TokenWidth,
			plan.image.TokenPatch, latentimage.PatchChannelsLast,
		)
		if err != nil {
			return generationFeatures{}, err
		}
		hidden, err := plan.flow.VisionEmbedTokens(context.Background(), planar)
		if err != nil {
			return generationFeatures{}, fmt.Errorf("sensenova recipe: step %d vision embedding: %w", step, err)
		}
		g.stats.Vision += time.Since(started)
		started = time.Now()
		timestep := plan.schedule[step]
		condition, err := plan.flow.FlowConditionRow(
			context.Background(), timestep, g.flow.NormalizedNoiseScale(plan.image.NoiseScale),
		)
		if err != nil {
			return generationFeatures{}, fmt.Errorf("sensenova recipe: step %d condition: %w", step, err)
		}
		if err := routedlm.AddConditionRows(hidden, condition); err != nil {
			return generationFeatures{}, err
		}
		g.stats.Condition += time.Since(started)
		started = time.Now()
		branches, _, err := plan.session.Run(context.Background(), hidden, nil, nil)
		if err != nil {
			return generationFeatures{}, fmt.Errorf("sensenova recipe: step %d body: %w", step, err)
		}
		g.stats.Body += time.Since(started)
		started = time.Now()
		velocities := make([][]float32, len(branches))
		for branch := range branches {
			velocities[branch], err = plan.flow.FlowHeadVelocity(context.Background(), branches[branch], z, timestep)
			if err != nil {
				return generationFeatures{}, fmt.Errorf("sensenova recipe: step %d branch %d flow head: %w", step, branch, err)
			}
		}
		g.stats.Flow += time.Since(started)
		started = time.Now()
		guided, err := routedlm.GuidedFlowVelocity(velocities, []float32{plan.request.CFGScale, 1 - plan.request.CFGScale})
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

func (g *Generator) decode(features generationFeatures) (latentimage.EncodedImage, error) {
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
	return workflowruntime.RegisterPipeline(
		runtime, modelID,
		modelrecipe.ModuleRoutedImagePrepare, generator.prepare,
		modelrecipe.ModuleRoutedImageIntegrate, generator.integrate,
		modelrecipe.ModuleRoutedImageDecode, generator.decode,
		latentimage.PNGContent,
	)
}
