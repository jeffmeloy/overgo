package latentimage

import (
	"errors"
	"overgo/internal/checked"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type rotarySamplingSpec struct {
	schedule    FlowSchedule
	guidance    float32
	skip        []int
	start, stop float64
}

func (s rotarySamplingSpec) validate(layers int) error {
	if !checked.NonNegativeFinite32(s.guidance) || !checked.UnitInterval64(s.start) || !checked.UnitInterval64(s.stop) || s.start > s.stop || s.schedule.Steps <= 0 || len(s.schedule.Timesteps) != s.schedule.Steps || len(s.schedule.Deltas) != s.schedule.Steps {
		return errors.New("rotary sampling: invalid guidance, skip interval or schedule")
	}
	for _, index := range s.skip {
		if !checked.ValidIndex(index, layers) {
			return errors.New("rotary sampling: skip index outside transformer depth")
		}
	}
	for i, time := range s.schedule.Timesteps {
		if !checked.Finite32(float32(time)) || !checked.Finite32(float32(s.schedule.Deltas[i])) || s.schedule.Deltas[i] > 0 {
			return errors.New("rotary sampling: invalid timestep or Euler delta")
		}
	}
	return nil
}

func (s rotarySamplingSpec) skipped(step int) bool {
	fraction := float64(step) / float64(s.schedule.Steps)
	return len(s.skip) > 0 && s.start <= fraction && fraction < s.stop
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
