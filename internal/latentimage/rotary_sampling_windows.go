//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"path/filepath"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/media"
	"overgo/internal/safetensors"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/torchrng"
)

// samplingScoresInRegisters says a rotary sampling's BF16 attention keeps its scores in registers
// (tensor.AttentionAttributes.ScoresInRegisters): the native kernel took 28.9 ms a layer of MOSS at 12432 tokens, 44%
// of a pass (moss-attention). A test reads the native kernel's sampling beside it.
var samplingScoresInRegisters = true

// samplingInt8 says a rotary sampling multiplies the joint blocks' projections in int8 as comfy_kitchen's int8_linear
// does where the transformer's file holds them int8 ConvRot (heldInConvRot), as the MOSS Ultra workflow's does
// (moss-int8): decoded to BF16, they took 0.76 s of a pass at 2368 x 1344 (moss-speed/do), and drew by other arithmetic
// than ComfyUI's. A test reads the BF16 sampling beside it.
var samplingInt8 = true

// heldInConvRot is the joint projections a transformer directory's file holds in int8 ConvRot over groups of 256
// columns (convRotProjections), as the MOSS Ultra workflow's file holds each of its layers' four.
func heldInConvRot(directory string, layers int) (map[string]bool, error) {
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	return convRotProjections(source, layers)
}

// runRotarySampling retains latents through guidance and Euler updates.
// The caller owns the session and returned latent, and observations alone download intermediate values.
// Where kept holds the programs a session's last sampling left it, their transformer weights stay on the card for this
// one, which binds them by their file and name as they were, and the last's programs are released once this one's first
// holds them, before its adapters are read, so an adapter is read again each sampling; this sampling's programs are
// kept in their place for the next (moss-resident: each request had copied the transformer's 9.5 GB to the card again,
// about 33 s at 2368 x 1344). A nil kept releases them as the sampling ends.
func runRotarySampling(ctx context.Context, runtime *graphruntime.ResidentSession, kept *[]*graphruntime.ResidentProgram, modelRoot string, spec TransformerSpec, height, width int, initial, positive, negative []float32, settings rotarySamplingSpec, observe func(string, []float32) error) (result *residentLatent, err error) {
	if err = settings.validate(spec.Layers); err != nil {
		return nil, err
	}
	if spec.TextHidden <= 0 || len(positive)%spec.TextHidden != 0 || len(negative)%spec.TextHidden != 0 {
		return nil, errors.New("rotary sampling: conditioning widths differ")
	}
	var int8 map[string]bool
	if samplingInt8 {
		if int8, err = heldInConvRot(filepath.Join(modelRoot, "transformer"), spec.Layers); err != nil {
			return nil, err
		}
	}
	var programs, previous []*graphruntime.ResidentProgram
	if kept != nil {
		previous, *kept = *kept, nil
	}
	defer func() {
		released := previous
		if err != nil || kept == nil {
			released = append(released, programs...)
		} else {
			*kept = programs
		}
		err = errors.Join(err, runtime.Release(context.WithoutCancel(ctx), released...))
		if err != nil && result != nil {
			err = errors.Join(err, result.outputs.Release(context.WithoutCancel(ctx)))
			result = nil
		}
	}()
	type branch struct {
		plan    *rotaryDenoiserProgram
		program *graphruntime.ResidentProgram
	}
	branches := map[string]branch{}
	load := func(name string, conditioning []float32, skip []int, weights ...float32) (branch, error) {
		if existing, ok := branches[name]; ok {
			return existing, nil
		}
		p, e := compileRotaryDenoiserProgram(spec, mossRefinementEpsilon, height, width, len(conditioning)/spec.TextHidden, skip, dtype.BF16, rotaryOptions{eightBit: settings.eightBit, registers: samplingScoresInRegisters, int8: int8, adapters: settings.adapters}, weights...)
		if e != nil {
			return branch{}, e
		}
		feeds, e := p.feeds(initial, conditioning, float32(settings.schedule.Timesteps[0]))
		if e != nil {
			return branch{}, e
		}
		program, e := compileResidentWeights(ctx, runtime, "rotary sampling "+name, filepath.Join(modelRoot, "transformer"), p.weightInputs, []*tensor.Tensor{p.latent}, p.output)
		if e != nil {
			return branch{}, e
		}
		programs = append(programs, program)
		// The last sampling's programs go once this one holds the transformer's weights, before its adapters are read.
		if len(previous) > 0 {
			released := previous
			previous = nil
			if e := runtime.Release(ctx, released...); e != nil {
				return branch{}, e
			}
		}
		for index, adapter := range settings.adapters {
			if e := bindAdapterInputs(ctx, runtime, program, "rotary sampling "+name, adapter, p.adapterInputs[index]); e != nil {
				return branch{}, e
			}
			feeds[p.strengths[index]] = reference.Value{Shape: p.strengths[index].Shape, Data: []float32{settings.strengths[index]}}
		}
		for _, node := range append([]*tensor.Tensor{p.context, p.rope, p.unit, p.valueScale}, p.strengths...) {
			if node == nil {
				continue
			}
			if _, required := program.Graph.InputSlot(node); !required {
				continue
			}
			data := feeds[node].Data
			identity := fmt.Sprintf("sampling:%x", sha256.Sum256(driver.Bytes(data)))
			if e := runtime.BindF32(ctx, program, identity, node, data); e != nil {
				return branch{}, e
			}
		}
		v := branch{p, program}
		branches[name] = v
		return v, nil
	}
	// Each branch scales the values of its prompt's rows by that prompt's weights.
	conditional, e := load("positive", positive, nil, settings.weights...)
	if e != nil {
		return nil, e
	}
	shape := conditional.plan.latent.Shape
	// update is a compiled Euler update and the nodes of its plan.
	type update struct {
		plan    *flowGuidanceProgram
		program *graphruntime.ResidentProgram
	}
	compileUpdate := func(name string, guidance float32) (update, error) {
		p, e := compileFlowGuidanceProgram(shape, guidance)
		if e != nil {
			return update{}, e
		}
		dynamic := []*tensor.Tensor{p.latent, p.positive}
		if p.negative != nil {
			dynamic = append(dynamic, p.negative)
		}
		program, e := runtime.Compile(ctx, name, dynamic, p.velocity, p.next)
		if e != nil {
			return update{}, e
		}
		programs = append(programs, program)
		return update{p, program}, nil
	}
	guided, e := compileUpdate("rotary guidance and Euler", settings.guidance)
	if e != nil {
		return nil, e
	}
	// A step outside the guidance interval, or one whose velocity the host guided (CFG-Zero*'s projection), steps by the
	// update at a scale of one; the reference recipe compiles none.
	plain := guided
	if guided.plan.negative != nil && settings.varies != (sampling.Guidance{}) {
		if plain, e = compileUpdate("rotary Euler", tensor.UnitScale); e != nil {
			return nil, e
		}
	}
	b := tensor.NewBuilder()
	input := b.Input("initial_latent", dtype.F32, shape)
	uploaded := b.Scale(input, tensor.UnitScale)
	upload, e := runtime.Compile(ctx, "rotary latent upload", nil, uploaded)
	if e != nil {
		return nil, e
	}
	programs = append(programs, upload)
	// hold holds what the host made on the device: the initial latent, a velocity the host guided, a latent the host
	// stepped.
	hold := func(data []float32) (*residentLatent, error) {
		held, e := runtime.Retain(ctx, upload, map[*tensor.Tensor]reference.Value{input: {Shape: shape, Data: data}}, nil)
		if e != nil {
			return nil, e
		}
		value, ok := held.Value(uploaded)
		if !ok {
			held.Release(context.WithoutCancel(ctx))
			return nil, errors.New("rotary sampling: uploaded latent missing")
		}
		return &residentLatent{outputs: held, node: uploaded, value: value}, nil
	}
	current, e := hold(initial)
	if e != nil {
		return nil, e
	}
	// APG carries its momentum over the generation's steps.
	adaptive := &sampling.AdaptiveGuider{Adaptive: settings.varies.Adaptive, Scale: float64(settings.guidance)}
	// hosted guides a step on the host and holds the guided velocity on the device: as CFG-Zero* does, the
	// unconditioned prediction rescaled by its projection onto the conditioned one over the whole sample
	// (sampling.Guider), or as APG does, which denoises the step's latent, a velocity taking it to the latent less
	// sigma times it.
	hosted := func(at sampling.Guider, sigma float64, positive, negative *residentLatent) (*residentLatent, error) {
		conditioned, e := positive.outputs.CopyToHost(ctx, positive.node)
		if e != nil {
			return nil, e
		}
		unconditioned, e := negative.outputs.CopyToHost(ctx, negative.node)
		if e != nil {
			return nil, e
		}
		velocity := make([]float32, len(conditioned.Data))
		if at.Guidance.Adaptive.On() {
			sample, e := current.outputs.CopyToHost(ctx, current.node)
			if e == nil {
				e = adaptive.Combine(velocity, conditioned.Data, unconditioned.Data, sample.Data, sigma)
			}
			if e != nil {
				return nil, e
			}
		} else if e := at.Combine(velocity, conditioned.Data, unconditioned.Data); e != nil {
			return nil, e
		}
		return hold(velocity)
	}
	// ER-SDE (sampling.ERSDE) takes each step on the host from the step's data prediction, which the update gives at a
	// delta of −σ, the Euler step to σ = 0, and draws its fresh noise as ComfyUI's er_sde does, from torch's CUDA
	// generator seeded with the request's seed (moss-ultra-sections/er-sde).
	var solver *sampling.ERSDE
	var noise *torchrng.Stream
	if settings.sampler == sampling.SamplerERSDE {
		sigmas := make([]float32, len(settings.schedule.Sigmas))
		for i, sigma := range settings.schedule.Sigmas {
			sigmas[i] = float32(sigma)
		}
		if solver, e = sampling.NewERSDE(sigmas, sampling.ERSDEStages); e != nil {
			return nil, e
		}
		noise = torchrng.NewStream(settings.seed)
		defer func() { err = errors.Join(err, runtime.Do(context.WithoutCancel(ctx), noise.Close)) }()
	}
	// stepped is ER-SDE's step from the current latent and the step's data prediction, held on the device.
	stepped := func(index int, prediction *residentLatent) (*residentLatent, error) {
		x, e := current.outputs.CopyToHost(ctx, current.node)
		if e != nil {
			return nil, e
		}
		denoised, e := prediction.outputs.CopyToHost(ctx, prediction.node)
		if e != nil {
			return nil, e
		}
		var fresh []float32
		if solver.Draws(index) {
			fresh = make([]float32, len(x.Data))
			if e := runtime.Do(ctx, func(state *device.State) error { return noise.FillHost(state, fresh) }); e != nil {
				return nil, e
			}
		}
		next, e := solver.Step(x.Data, denoised.Data, fresh)
		if e != nil {
			return nil, e
		}
		return hold(next)
	}
	defer func() {
		if current != nil {
			err = errors.Join(err, current.outputs.Release(context.WithoutCancel(ctx)))
		}
	}()
	if observe != nil {
		if e := observe("initial_latents", initial); e != nil {
			return nil, e
		}
	}
	// controlAt is what a widened embedder reads at a step of sigma: the control within its window, zeros outside it and
	// where the request has none, as ComfyUI's MOSS feeds its concat latent (concat_cond).
	var zeros []float32
	controlAt := func(node *tensor.Tensor, sigma float64) []float32 {
		if settings.control != nil && sigma >= settings.controlLow && sigma <= settings.controlHigh {
			return settings.control
		}
		if zeros == nil {
			elements, _ := node.Shape.Elements()
			zeros = make([]float32, elements)
		}
		return zeros
	}
	forwardIndex := 0
	sampler, e := settings.schedule.sampler()
	if e != nil {
		return nil, e
	}
	guider := sampling.Guider{Scale: float64(settings.guidance), Guidance: settings.varies}
	for index, time := range settings.schedule.Timesteps {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		// A step the guidance zeroes gives no velocity: the latent stays, and neither pass runs.
		at := guider.At(index, settings.schedule.Sigmas[index])
		if at.Zeroed() {
			if solver != nil {
				if e := solver.Skip(); e != nil {
					return nil, e
				}
			}
			continue
		}
		embedding, e := (media.SinusoidalProgram{Dimensions: spec.TimestepEmbed, FrequencyBase: rotaryTimestepBase, InputScale: float64(tensor.UnitScale)}).Encode32(time)
		if e != nil {
			return nil, e
		}
		advanceErr := func() error {
			run := func(v branch) (*residentLatent, error) {
				inputs := map[*tensor.Tensor]reference.Value{v.plan.embedding: {Shape: v.plan.embedding.Shape, Data: embedding}}
				if v.plan.control != nil {
					inputs[v.plan.control] = reference.Value{Shape: v.plan.control.Shape, Data: controlAt(v.plan.control, settings.schedule.Sigmas[index])}
				} else if settings.control != nil {
					return nil, errors.New("rotary sampling: the request's control needs an adapter whose embedder reads one (moss-control)")
				}
				out, e := runtime.Retain(ctx, v.program, inputs, []driver.DevicePtr{current.value.Pointer})
				if e != nil {
					return nil, e
				}
				value, ok := out.Value(v.plan.output)
				if !ok {
					out.Release(context.WithoutCancel(ctx))
					return nil, errors.New("rotary sampling: prediction missing")
				}
				if observe != nil {
					host, e := out.CopyToHost(ctx, v.plan.output)
					if e == nil {
						e = observe(fmt.Sprintf("prediction.%d", forwardIndex), host.Data)
					}
					if e != nil {
						out.Release(context.WithoutCancel(ctx))
						return nil, e
					}
				}
				forwardIndex++
				return &residentLatent{outputs: out, node: v.plan.output, value: value}, nil
			}
			pos, e := run(conditional)
			if e != nil {
				return e
			}
			defer pos.outputs.Release(context.WithoutCancel(ctx))
			// Outside the guidance interval the step steps by the conditioned prediction alone.
			step, pointers := plain, []driver.DevicePtr{current.value.Pointer, pos.value.Pointer}
			if at.Unconditional() {
				name, skip := "negative", settings.override.At(index, settings.schedule.Steps)
				if len(skip) > 0 {
					name = "negative-skipped"
				}
				v, e := load(name, negative, skip, settings.negativeWeights...)
				if e != nil {
					return e
				}
				neg, e := run(v)
				if e != nil {
					return e
				}
				defer neg.outputs.Release(context.WithoutCancel(ctx))
				step, pointers = guided, append(pointers, neg.value.Pointer)
				if at.Guidance.Projected || at.Guidance.Adaptive.On() {
					velocity, e := hosted(at, settings.schedule.Sigmas[index], pos, neg)
					if e != nil {
						return e
					}
					defer velocity.outputs.Release(context.WithoutCancel(ctx))
					step, pointers = plain, []driver.DevicePtr{current.value.Pointer, velocity.value.Pointer}
				}
			}
			// Under ER-SDE the update steps to σ = 0, the step's data prediction, from which the host steps.
			delta, reached := sampler.Delta(index), "latents"
			if solver != nil {
				delta, reached = -float32(settings.schedule.Sigmas[index]), "denoised"
			}
			next, e := runtime.Retain(ctx, step.program, map[*tensor.Tensor]reference.Value{step.plan.delta: {Shape: step.plan.delta.Shape, Data: []float32{delta}}}, pointers)
			if e != nil {
				return e
			}
			value, ok := next.Value(step.plan.next)
			if !ok {
				next.Release(context.WithoutCancel(ctx))
				return errors.New("rotary sampling: next latent missing")
			}
			if observe != nil {
				for _, item := range []struct {
					name string
					node *tensor.Tensor
				}{{fmt.Sprintf("velocity.%d", index), step.plan.velocity}, {fmt.Sprintf("%s.%d", reached, index), step.plan.next}} {
					host, e := next.CopyToHost(ctx, item.node)
					if e == nil {
						e = observe(item.name, host.Data)
					}
					if e != nil {
						next.Release(context.WithoutCancel(ctx))
						return e
					}
				}
			}
			following := &residentLatent{outputs: next, node: step.plan.next, value: value}
			if solver != nil {
				prediction := following
				following, e = stepped(index, prediction)
				if e = errors.Join(e, prediction.outputs.Release(context.WithoutCancel(ctx))); e != nil {
					if following != nil {
						following.outputs.Release(context.WithoutCancel(ctx))
					}
					return e
				}
			}
			if e := current.outputs.Release(context.WithoutCancel(ctx)); e != nil {
				following.outputs.Release(context.WithoutCancel(ctx))
				return e
			}
			current = following
			return nil
		}()
		if advanceErr != nil {
			return nil, advanceErr
		}
	}
	result, current = current, nil
	return result, nil
}
