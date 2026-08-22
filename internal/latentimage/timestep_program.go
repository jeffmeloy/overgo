package latentimage

import (
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type timestepProgram struct {
	Input        *tensor.Tensor
	Embedding    *tensor.Tensor
	Modulation   *tensor.Tensor
	weightInputs map[string]*tensor.Tensor
	encoding     media.SinusoidalProgram
}

func compileTimestepProgram(spec TransformerSpec, storage dtype.Type, sinusoid media.SinusoidalProgram) (*timestepProgram, error) {
	fieldsCount := spec.ModFieldsOr6()
	if !checked.PositiveInts(spec.TimestepEmbed, spec.Hidden, fieldsCount) || !checked.EvenInt(spec.TimestepEmbed) {
		return nil, fmt.Errorf(
			"timestep program: invalid geometry embed=%d hidden=%d fields=%d",
			spec.TimestepEmbed, spec.Hidden, spec.ModFieldsOr6(),
		)
	}
	switch storage {
	case dtype.F32, dtype.BF16:
	default:
		return nil, fmt.Errorf("timestep program: weight type %s unsupported", storage)
	}
	if err := sinusoid.Validate(); err != nil {
		return nil, fmt.Errorf("timestep program: %w", err)
	}
	if !checked.Equal(sinusoid.Dimensions, spec.TimestepEmbed) {
		return nil, fmt.Errorf("timestep program: sinusoid dimensions=%d incompatible with embed=%d", sinusoid.Dimensions, spec.TimestepEmbed)
	}
	builder := tensor.NewBuilder()
	setBuilderMatmulCompute(builder, storage)
	program := &timestepProgram{
		weightInputs: make(map[string]*tensor.Tensor), encoding: sinusoid,
	}
	binder := tensor.WeightInputs{Builder: builder, Inputs: program.weightInputs, MatrixType: storage}
	hidden := uint64(spec.Hidden)
	fields := uint64(fieldsCount)
	program.Input = builder.Input(
		"timestep_sinusoid", dtype.F32, tensor.MustShape(uint64(spec.TimestepEmbed), tensor.SingletonExtent),
	)
	first := builder.GELUTanhExact(builder.Add(
		builder.MulMat(binder.Input("time_embed.linear_1.weight", uint64(spec.TimestepEmbed), hidden), program.Input),
		binder.Input("time_embed.linear_1.bias", hidden),
	))
	program.Embedding = builder.Add(
		builder.MulMat(binder.Input("time_embed.linear_2.weight", hidden, hidden), first),
		binder.Input("time_embed.linear_2.bias", hidden),
	)
	program.Modulation = builder.Add(
		builder.MulMat(
			binder.Input("time_mod_proj.weight", hidden, fields*hidden),
			builder.GELUTanhExact(program.Embedding),
		),
		binder.Input("time_mod_proj.bias", fields*hidden),
	)
	if err := builder.Err(); err != nil {
		return nil, fmt.Errorf("timestep program: %w", err)
	}
	return program, nil
}

func (p *timestepProgram) sinusoid(sigma float64) ([]float32, error) {
	if p == nil {
		return nil, fmt.Errorf("timestep program: invalid sigma %g", sigma)
	}
	return p.encoding.Encode32(sigma)
}
