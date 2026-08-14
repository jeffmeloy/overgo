package latentimage

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

const (
	timestepFrequencyBase = 1e4
	timestepInputScale    = 1e3
)

type timestepProgram struct {
	Input        *tensor.Tensor
	Embedding    *tensor.Tensor
	Modulation   *tensor.Tensor
	weightInputs map[string]*tensor.Tensor
	dim          int
}

func compileTimestepProgram(spec TransformerSpec, storage dtype.Type) (*timestepProgram, error) {
	if spec.TimestepEmbed <= 0 || spec.TimestepEmbed%2 != 0 || spec.Hidden <= 0 || spec.ModFieldsOr6() <= 0 {
		return nil, fmt.Errorf(
			"timestep program: invalid geometry embed=%d hidden=%d fields=%d",
			spec.TimestepEmbed, spec.Hidden, spec.ModFieldsOr6(),
		)
	}
	if storage != dtype.F32 && storage != dtype.BF16 {
		return nil, fmt.Errorf("timestep program: weight type %s unsupported", storage)
	}
	builder := tensor.NewBuilder()
	setBuilderMatmulCompute(builder, storage)
	program := &timestepProgram{
		weightInputs: make(map[string]*tensor.Tensor), dim: spec.TimestepEmbed,
	}
	binder := weightBinder{builder: builder, inputs: program.weightInputs, matmulType: storage}
	hidden := uint64(spec.Hidden)
	fields := uint64(spec.ModFieldsOr6())
	program.Input = builder.Input(
		"timestep_sinusoid", dtype.F32, tensor.MustShape(uint64(spec.TimestepEmbed), 1),
	)
	first := builder.GELUTanhExact(builder.Add(
		builder.MulMat(binder.input("time_embed.linear_1.weight", uint64(spec.TimestepEmbed), hidden), program.Input),
		binder.input("time_embed.linear_1.bias", hidden),
	))
	program.Embedding = builder.Add(
		builder.MulMat(binder.input("time_embed.linear_2.weight", hidden, hidden), first),
		binder.input("time_embed.linear_2.bias", hidden),
	)
	program.Modulation = builder.Add(
		builder.MulMat(
			binder.input("time_mod_proj.weight", hidden, fields*hidden),
			builder.GELUTanhExact(program.Embedding),
		),
		binder.input("time_mod_proj.bias", fields*hidden),
	)
	if err := builder.Err(); err != nil {
		return nil, fmt.Errorf("timestep program: %w", err)
	}
	return program, nil
}

func (p *timestepProgram) sinusoid(sigma float64) ([]float32, error) {
	if p == nil || p.dim <= 0 || math.IsNaN(sigma) || math.IsInf(sigma, 0) {
		return nil, fmt.Errorf("timestep program: invalid sigma %g", sigma)
	}
	half := p.dim / 2
	values := make([]float32, p.dim)
	for index := range half {
		frequency := math.Exp(-math.Log(timestepFrequencyBase) * float64(index) / float64(half))
		angle := sigma * timestepInputScale * frequency
		values[index] = float32(math.Cos(angle))
		values[half+index] = float32(math.Sin(angle))
	}
	return values, nil
}
