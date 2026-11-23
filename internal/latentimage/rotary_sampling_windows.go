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
	"overgo/internal/samplingcuda"
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
// does where the transformer's file holds them int8 ConvRot (declaredInt8), as the MOSS Ultra workflow's does
// (moss-int8): decoded to BF16, they took 0.76 s of a pass at 2368 x 1344 (moss-speed/do), and drew by other arithmetic
// than ComfyUI's. A test reads the BF16 sampling beside it.
var samplingInt8 = true

// samplingInt8Attention says a rotary sampling's joint layers take comfy_kitchen's int8 attention where the
// transformer's file declares it for them (int8AttentionLayers), as ComfyUI runs them and the MOSS Ultra workflow's file
// declares for its 42 layers (moss-attention-pass): the BF16 attention in registers and its copies of K and V took 29.75
// ms of a text to image pass at 1024 x 576, the int8 attention 17.69 ms against ComfyUI's 11.53, and each section's
// final latent lies nearer its recording. A test reads the BF16 attention beside it.
var samplingInt8Attention = true

// samplingRefinesOnce says a rotary sampling refines each branch's text once, by its text program
// (rotaryTextProgram), and binds the rows to the branch's passes, which had each refined them (SQA-422). A test reads
// the passes that refine it beside it.
var samplingRefinesOnce = true

// declaredInt8 is what a transformer directory's file holds and declares in int8, its file opened once: the joint
// projections it holds in int8 ConvRot over groups of 256 columns (convRotProjections), as the MOSS Ultra workflow's
// file holds each of its layers' four, where projections asks for them, and the joint layers it declares
// comfy_kitchen's int8 attention for (int8AttentionLayers), where attention asks for them.
func declaredInt8(directory string, layers int, projections, attention bool) (held map[string]bool, declared map[int]bool, err error) {
	if !projections && !attention {
		return nil, nil, nil
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, nil, err
	}
	defer source.Close()
	if projections {
		if held, err = convRotProjections(source, layers); err != nil {
			return nil, nil, err
		}
	}
	if attention {
		if declared, err = int8AttentionLayers(source, layers); err != nil {
			return nil, nil, err
		}
	}
	return held, declared, nil
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
	int8, int8Attention, err := declaredInt8(filepath.Join(modelRoot, "transformer"), spec.Layers, samplingInt8, samplingInt8Attention)
	if err != nil {
		return nil, err
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
		options := rotaryOptions{eightBit: settings.eightBit, registers: samplingScoresInRegisters, int8: int8, int8Attention: int8Attention, adapters: settings.adapters, refinedText: samplingRefinesOnce}
		text := len(conditioning) / spec.TextHidden
		p, e := compileRotaryDenoiserProgram(spec, mossRefinementEpsilon, height, width, text, skip, dtype.BF16, options, weights...)
		if e != nil {
			return branch{}, e
		}
		feeds, e := p.feeds(initial, conditioning, float32(settings.schedule.Timesteps[0]))
		if e != nil {
			return branch{}, e
		}
		// The latent and the control a widened embedder reads are on the card, each pass reading them where they lie.
		dynamic := []*tensor.Tensor{p.latent}
		if p.control != nil {
			dynamic = append(dynamic, p.control)
		}
		program, e := compileResidentWeights(ctx, runtime, "rotary sampling "+name, filepath.Join(modelRoot, "transformer"), p.weightInputs, dynamic, p.output)
		if e != nil {
			return branch{}, e
		}
		programs = append(programs, program)
		if e := bindMerged(ctx, runtime, program, "rotary sampling "+name, filepath.Join(modelRoot, "transformer"), p.merged, settings.adapters, settings.strengths); e != nil {
			return branch{}, e
		}
		// The branch's text program is compiled beside its passes, before the last sampling's programs go, so the refiner's
		// weights a session holds are bound as they were.
		var refiner *rotaryTextProgram
		var refining *graphruntime.ResidentProgram
		if p.refined != nil {
			if refiner, e = compileRotaryTextProgram(spec, mossRefinementEpsilon, height, width, text, dtype.BF16, options); e != nil {
				return branch{}, e
			}
			if refining, e = compileResidentWeights(ctx, runtime, "rotary text "+name, filepath.Join(modelRoot, "transformer"), refiner.weightInputs, nil, refiner.output); e != nil {
				return branch{}, e
			}
			programs = append(programs, refining)
			if e := bindMerged(ctx, runtime, refining, "rotary text "+name, filepath.Join(modelRoot, "transformer"), refiner.merged, settings.adapters, settings.strengths); e != nil {
				return branch{}, e
			}
		}
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
			if refining != nil {
				if e := bindAdapterInputs(ctx, runtime, refining, "rotary text "+name, adapter, refiner.adapterInputs[index]); e != nil {
					return branch{}, e
				}
			}
		}
		// The branch's text is refined once, over its conditioning and the text's rotary rows, and each pass reads the rows.
		if refining != nil {
			inputs := map[*tensor.Tensor]reference.Value{refiner.context: {Shape: refiner.context.Shape, Data: conditioning}, refiner.rope: feeds[p.rope]}
			for index, strength := range refiner.strengths {
				inputs[strength] = feeds[p.strengths[index]]
			}
			out, e := runtime.Retain(ctx, refining, inputs, nil)
			if e != nil {
				return branch{}, e
			}
			rows, e := out.CopyToHost(ctx, refiner.output)
			out.Release(context.WithoutCancel(ctx))
			if e != nil {
				return branch{}, e
			}
			feeds[p.refined] = reference.Value{Shape: p.refined.Shape, Data: rows.Data}
		}
		for _, node := range append([]*tensor.Tensor{p.context, p.rope, p.unit, p.valueScale, p.refined}, p.strengths...) {
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
	// holder holds values of a shape on the device: copied from the host where data is given, else from the device
	// pointer from, by the program a copy of each compiles to.
	holder := func(name string, shape tensor.Shape, onDevice bool) (func(data []float32, from driver.DevicePtr) (*residentLatent, error), error) {
		b := tensor.NewBuilder()
		input := b.Input(name, dtype.F32, shape)
		held := b.Scale(input, tensor.UnitScale)
		var dynamic []*tensor.Tensor
		if onDevice {
			dynamic = []*tensor.Tensor{input}
		}
		program, e := runtime.Compile(ctx, "rotary "+name, dynamic, held)
		if e != nil {
			return nil, e
		}
		programs = append(programs, program)
		return func(data []float32, from driver.DevicePtr) (*residentLatent, error) {
			var inputs map[*tensor.Tensor]reference.Value
			var pointers []driver.DevicePtr
			if onDevice {
				pointers = []driver.DevicePtr{from}
			} else {
				inputs = map[*tensor.Tensor]reference.Value{input: {Shape: shape, Data: data}}
			}
			out, e := runtime.Retain(ctx, program, inputs, pointers)
			if e != nil {
				return nil, e
			}
			value, ok := out.Value(held)
			if !ok {
				out.Release(context.WithoutCancel(ctx))
				return nil, errors.New("rotary sampling: held value missing")
			}
			return &residentLatent{outputs: out, node: held, value: value}, nil
		}, nil
	}
	upload, e := holder("latent upload", shape, false)
	if e != nil {
		return nil, e
	}
	duplicate, e := holder("latent copy", shape, true)
	if e != nil {
		return nil, e
	}
	// hold holds what the host made on the device: the initial latent, a velocity the host guided.
	hold := func(data []float32) (*residentLatent, error) { return upload(data, 0) }
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
	// held are the values a sampling holds on the device beside its latent, each released as the sampling ends: ER-SDE's
	// last prediction, its divided differences and its noise, and the control's.
	var held []*residentLatent
	defer func() {
		for _, value := range held {
			if value != nil {
				err = errors.Join(err, value.outputs.Release(context.WithoutCancel(ctx)))
			}
		}
	}()
	// ER-SDE (sampling.ERSDE) takes each step from the step's data prediction, which the update gives at a delta of −σ,
	// the Euler step to σ = 0, on the card as the host steps (samplingcuda.ERSDEDevice, moss-step-copies), and draws its
	// fresh noise there as ComfyUI's er_sde does, from torch's CUDA generator seeded with the request's seed
	// (moss-ultra-sections/er-sde): the latent, the predictions, their divided differences and the noise stay on the card.
	var solver *sampling.ERSDE
	var noise *torchrng.Stream
	var stepper samplingcuda.ERSDEDevice
	// lastPrediction is the last prediction, kept for the next step; firsts hold the divided differences, the step's written
	// to firsts[written] and the last's read from the other; fresh holds the step's noise.
	var lastPrediction, fresh *residentLatent
	var firsts [tensor.PairedExtent]*residentLatent
	written := 0
	elements, e := shape.Elements()
	if e != nil {
		return nil, e
	}
	count := int(elements)
	if settings.sampler == sampling.SamplerERSDE {
		sigmas := make([]float32, len(settings.schedule.Sigmas))
		for i, sigma := range settings.schedule.Sigmas {
			sigmas[i] = float32(sigma)
		}
		if solver, e = sampling.NewERSDE(sigmas, sampling.ERSDEStages); e != nil {
			return nil, e
		}
		noise = torchrng.NewStream(settings.seed)
		defer func() {
			err = errors.Join(err, runtime.Do(context.WithoutCancel(ctx), func(state *device.State) error {
				return errors.Join(noise.Close(state), stepper.Close(state))
			}))
		}()
		defer func() { held = append(held, lastPrediction, fresh, firsts[0], firsts[1]) }()
	}
	// forget lets the last prediction go, as a skipped step makes the solver's order start again.
	forget := func() error {
		if lastPrediction == nil {
			return nil
		}
		released := lastPrediction
		lastPrediction = nil
		return released.outputs.Release(context.WithoutCancel(ctx))
	}
	// buffer is a device value of the latent's shape the step writes into, made once from the current latent.
	buffer := func(value **residentLatent) error {
		if *value != nil {
			return nil
		}
		made, e := duplicate(nil, current.value.Pointer)
		*value = made
		return e
	}
	// stepped is ER-SDE's step from the current latent and the step's data prediction, which it keeps as the last
	// prediction or, for a step to 0, gives as the next latent.
	stepped := func(prediction *residentLatent) (*residentLatent, error) {
		plan, e := solver.Plan()
		if e != nil {
			return nil, errors.Join(e, prediction.outputs.Release(context.WithoutCancel(ctx)))
		}
		if plan.ToZero {
			return prediction, forget()
		}
		next, e := duplicate(nil, current.value.Pointer)
		if e == nil {
			e = errors.Join(buffer(&fresh), buffer(&firsts[0]), buffer(&firsts[1]))
		}
		if e != nil {
			if next != nil {
				e = errors.Join(e, next.outputs.Release(context.WithoutCancel(ctx)))
			}
			return nil, errors.Join(e, prediction.outputs.Release(context.WithoutCancel(ctx)))
		}
		var last driver.DevicePtr
		if lastPrediction != nil {
			last = lastPrediction.value.Pointer
		}
		e = runtime.Do(ctx, func(state *device.State) error {
			if e := noise.FillDevice(state, fresh.value.Pointer, count); e != nil {
				return e
			}
			return stepper.Step(state, plan, next.value.Pointer, prediction.value.Pointer, fresh.value.Pointer, last,
				firsts[written].value.Pointer, firsts[1-written].value.Pointer, count)
		})
		if plan.Order >= 2 {
			written = 1 - written
		}
		e = errors.Join(e, forget())
		lastPrediction = prediction
		if e != nil {
			return nil, errors.Join(e, next.outputs.Release(context.WithoutCancel(ctx)))
		}
		return next, nil
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
	// controlAt is what a widened embedder reads at a step of sigma, on the device: the control within its window, zeros
	// outside it and where the request has none, as ComfyUI's MOSS feeds its concat latent (concat_cond), each copied to
	// the card once a sampling.
	var controls [tensor.PairedExtent]*residentLatent
	var controlUpload func(data []float32, from driver.DevicePtr) (*residentLatent, error)
	defer func() { held = append(held, controls[0], controls[1]) }()
	controlAt := func(node *tensor.Tensor, sigma float64) (driver.DevicePtr, error) {
		at, data := 0, settings.control
		if settings.control == nil || sigma < settings.controlLow || sigma > settings.controlHigh {
			elements, e := node.Shape.Elements()
			if e != nil {
				return 0, e
			}
			at, data = 1, make([]float32, elements)
		}
		if controls[at] == nil {
			if controlUpload == nil {
				var e error
				if controlUpload, e = holder("control upload", node.Shape, false); e != nil {
					return 0, e
				}
			}
			var e error
			if controls[at], e = controlUpload(data, 0); e != nil {
				return 0, e
			}
		}
		return controls[at].value.Pointer, nil
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
				if e := errors.Join(solver.Skip(), forget()); e != nil {
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
				pointers := []driver.DevicePtr{current.value.Pointer}
				if v.plan.control != nil {
					control, e := controlAt(v.plan.control, settings.schedule.Sigmas[index])
					if e != nil {
						return nil, e
					}
					pointers = append(pointers, control)
				} else if settings.control != nil {
					return nil, errors.New("rotary sampling: the request's control needs an adapter whose embedder reads one (moss-control)")
				}
				out, e := runtime.Retain(ctx, v.program, inputs, pointers)
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
				if following, e = stepped(following); e != nil {
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
