package latentimage

import (
	"errors"
	"overgo/internal/checked"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type rotarySamplingSpec struct {
	schedule FlowSchedule
	guidance float32
	// varies is how the guidance varies over the steps (sampling.Guider.At); its zero value guides every step alike.
	varies sampling.Guidance
	// override is what the negative branch changes of the transformer's blocks (sampling.BlockOverride).
	override sampling.BlockOverride
	// weights are the conditioned branch's text rows' attention-value weights (weightedSpans), none for a prompt that
	// weights nothing.
	weights []float32
}

func (s rotarySamplingSpec) validate(layers int) error {
	if !checked.NonNegativeFinite32(s.guidance) || s.schedule.Steps <= 0 || len(s.schedule.Timesteps) != s.schedule.Steps || len(s.schedule.Deltas) != s.schedule.Steps || len(s.schedule.Sigmas) <= s.schedule.Steps {
		return errors.New("rotary sampling: invalid guidance or schedule")
	}
	if err := s.override.Validate(layers); err != nil {
		return err
	}
	if err := s.varies.Validate(); err != nil {
		return err
	}
	for i, time := range s.schedule.Timesteps {
		if !checked.Finite32(float32(time)) || !checked.Finite32(float32(s.schedule.Deltas[i])) || s.schedule.Deltas[i] > 0 {
			return errors.New("rotary sampling: invalid timestep or Euler delta")
		}
	}
	return nil
}

// negativeOverride is the block override a rotary profile's negative branch carries: the blocks it skips over the whole
// schedule (MOSS's block 22).
func (r rotaryPolicy) negativeOverride() sampling.BlockOverride {
	return sampling.BlockOverride{Blocks: r.NegativeSkip, Stop: 1}
}

type flowGuidanceProgram struct {
	latent, positive, negative, delta, velocity, next *tensor.Tensor
}

func compileFlowGuidanceProgram(shape tensor.Shape, guidance float32) (*flowGuidanceProgram, error) {
	if !checked.NonNegativeFinite32(guidance) {
		return nil, errors.New("flow guidance: scale must be finite and nonnegative")
	}
	b := tensor.NewBuilder()
	p := &flowGuidanceProgram{}
	p.latent = b.Input("latent", dtype.F32, shape)
	p.positive = b.Input("positive", dtype.F32, shape)
	p.delta = b.Input("delta", dtype.F32, tensor.MustShape(tensor.SingletonExtent))
	p.velocity = p.positive
	if guidance != tensor.UnitScale {
		p.negative = b.Input("negative", dtype.F32, shape)
		difference := b.Add(p.positive, b.Scale(p.negative, -tensor.UnitScale))
		p.velocity = b.Add(p.negative, b.Scale(difference, guidance))
	}
	p.next = b.Add(p.latent, b.Multiply(p.velocity, p.delta))
	return p, b.Err()
}
