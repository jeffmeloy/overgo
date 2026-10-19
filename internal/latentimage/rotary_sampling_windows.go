//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"path/filepath"

	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// runRotarySampling retains latents through guidance and Euler updates.
// The caller owns the session and returned latent, and observations alone download intermediate values.
func runRotarySampling(ctx context.Context, runtime *graphruntime.ResidentSession, modelRoot string, spec TransformerSpec, height, width int, initial, positive, negative []float32, settings rotarySamplingSpec, observe func(string, []float32) error) (result *residentLatent, err error) {
	if err = settings.validate(spec.Layers); err != nil {
		return nil, err
	}
	if spec.TextHidden <= 0 || len(positive)%spec.TextHidden != 0 || len(negative)%spec.TextHidden != 0 {
		return nil, errors.New("rotary sampling: conditioning widths differ")
	}
	var programs []*graphruntime.ResidentProgram
	defer func() {
		err = errors.Join(err, runtime.Release(context.WithoutCancel(ctx), programs...))
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
	load := func(name string, conditioning []float32, skip []int) (branch, error) {
		if existing, ok := branches[name]; ok {
			return existing, nil
		}
		p, e := compileRotaryDenoiserProgram(spec, mossRefinementEpsilon, height, width, len(conditioning)/spec.TextHidden, skip, dtype.BF16)
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
		for _, node := range []*tensor.Tensor{p.context, p.rope, p.unit} {
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
	conditional, e := load("positive", positive, nil)
	if e != nil {
		return nil, e
	}
	shape := conditional.plan.latent.Shape
	step, e := compileFlowGuidanceProgram(shape, settings.guidance)
	if e != nil {
		return nil, e
	}
	dynamic := []*tensor.Tensor{step.latent, step.positive}
	if step.negative != nil {
		dynamic = append(dynamic, step.negative)
	}
	update, e := runtime.Compile(ctx, "rotary guidance and Euler", dynamic, step.velocity, step.next)
	if e != nil {
		return nil, e
	}
	programs = append(programs, update)
	b := tensor.NewBuilder()
	input := b.Input("initial_latent", dtype.F32, shape)
	uploaded := b.Scale(input, tensor.UnitScale)
	upload, e := runtime.Compile(ctx, "rotary latent upload", nil, uploaded)
	if e != nil {
		return nil, e
	}
	programs = append(programs, upload)
	retained, e := runtime.Retain(ctx, upload, map[*tensor.Tensor]reference.Value{input: {Shape: shape, Data: initial}}, nil)
	if e != nil {
		return nil, e
	}
	value, ok := retained.Value(uploaded)
	if !ok {
		retained.Release(context.WithoutCancel(ctx))
		return nil, errors.New("rotary sampling: uploaded latent missing")
	}
	current := &residentLatent{outputs: retained, node: uploaded, value: value}
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
	forwardIndex := 0
	for index, time := range settings.schedule.Timesteps {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		embedding, e := (media.SinusoidalProgram{Dimensions: spec.TimestepEmbed, FrequencyBase: rotaryTimestepBase, InputScale: float64(tensor.UnitScale)}).Encode32(time)
		if e != nil {
			return nil, e
		}
		advanceErr := func() error {
			run := func(v branch) (*residentLatent, error) {
				out, e := runtime.Retain(ctx, v.program, map[*tensor.Tensor]reference.Value{v.plan.embedding: {Shape: v.plan.embedding.Shape, Data: embedding}}, []driver.DevicePtr{current.value.Pointer})
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
			pointers := []driver.DevicePtr{current.value.Pointer, pos.value.Pointer}
			if step.negative != nil {
				name, skip := "negative", []int(nil)
				if settings.skipped(index) {
					name, skip = "negative-skipped", settings.skip
				}
				v, e := load(name, negative, skip)
				if e != nil {
					return e
				}
				neg, e := run(v)
				if e != nil {
					return e
				}
				defer neg.outputs.Release(context.WithoutCancel(ctx))
				pointers = append(pointers, neg.value.Pointer)
			}
			next, e := runtime.Retain(ctx, update, map[*tensor.Tensor]reference.Value{step.delta: {Shape: step.delta.Shape, Data: []float32{float32(settings.schedule.Deltas[index])}}}, pointers)
			if e != nil {
				return e
			}
			value, ok := next.Value(step.next)
			if !ok {
				next.Release(context.WithoutCancel(ctx))
				return errors.New("rotary sampling: next latent missing")
			}
			if observe != nil {
				for _, item := range []struct {
					name string
					node *tensor.Tensor
				}{{fmt.Sprintf("velocity.%d", index), step.velocity}, {fmt.Sprintf("latents.%d", index), step.next}} {
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
			if e := current.outputs.Release(context.WithoutCancel(ctx)); e != nil {
				next.Release(context.WithoutCancel(ctx))
				return e
			}
			current = &residentLatent{outputs: next, node: step.next, value: value}
			return nil
		}()
		if advanceErr != nil {
			return nil, advanceErr
		}
	}
	result, current = current, nil
	return result, nil
}
