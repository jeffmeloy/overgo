package projector

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func (r *Gemma4Runner) encodeCUDAWithTrace(
	ctx context.Context,
	input Gemma4Image,
	trace gemma4Trace,
) (Gemma4Output, error) {
	rows := input.GridH * input.GridW
	if len(input.PixelValues) != rows*r.spec.PatchWidth || len(input.Positions) != rows*2 {
		return Gemma4Output{}, errors.New("projector: Gemma 4 input shape is inconsistent")
	}
	pixels := make([]float32, len(input.PixelValues))
	patchArea := r.spec.ModelPatch * r.spec.ModelPatch
	for row := 0; row < rows; row++ {
		source := input.PixelValues[row*r.spec.PatchWidth:]
		destination := pixels[row*r.spec.PatchWidth:]
		for pixel := 0; pixel < patchArea; pixel++ {
			for channel := 0; channel < 3; channel++ {
				destination[channel*patchArea+pixel] = source[pixel*3+channel]
			}
		}
	}

	builder := tensor.NewBuilder()
	pixelInput := builder.Input(
		visionInputTensor, dtype.F32, tensor.MustShape(uint64(r.spec.PatchWidth), uint64(rows)),
	)
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	weight := graph.weight
	affineNorm := normalizationPlan{kind: normalizationAffineLayer, epsilon: r.spec.LayerNormEpsilon, bf16: true}
	rmsNorm := normalizationPlan{kind: normalizationRMS, epsilon: r.spec.RMSNormEpsilon, bf16: true}
	pixelsNode := builder.BF16Round(pixelInput)
	ln1 := affineNorm.graph(builder, pixelsNode, weight("v.patch_norm.1.weight"), weight("v.patch_norm.1.bias"))
	patchDense := builder.BF16Round(builder.Add(
		builder.MulMat(weight(visionPatchWeightTensor), ln1), weight(visionPatchBiasTensor),
	))
	ln2 := affineNorm.graph(builder, patchDense, weight("v.patch_norm.2.weight"), weight("v.patch_norm.2.bias"))
	position := builder.Reshape(
		weight(visionPositionWeightTensor), uint64(r.spec.Hidden), uint64(r.spec.PositionCount*2),
	)
	xRows := make([]uint32, rows)
	yRows := make([]uint32, rows)
	for row := 0; row < rows; row++ {
		x, y := input.Positions[row*2], input.Positions[row*2+1]
		if x < 0 || y < 0 || x >= r.spec.PositionCount || y >= r.spec.PositionCount {
			return Gemma4Output{}, fmt.Errorf("projector: position %d,%d exceeds table", x, y)
		}
		xRows[row] = uint32(x)
		yRows[row] = uint32(r.spec.PositionCount + y)
	}
	positionSum := builder.BF16Round(builder.Add(builder.GetRows(position, xRows), builder.GetRows(position, yRows)))
	positioned := builder.BF16Round(builder.Add(ln2, positionSum))
	posNorm := affineNorm.graph(builder, positioned, weight("v.patch_norm.3.weight"), weight("v.patch_norm.3.bias"))
	preProjection := rmsNorm.graph(builder, posNorm, nil, nil)
	embeddings := builder.BF16Round(builder.MulMat(weight(multimodalInputProjection), preProjection))
	graph.hostFeeds[pixelInput] = reference.Value{Shape: pixelInput.Shape, Data: pixels}
	outputs := []*tensor.Tensor{embeddings}
	if trace != nil {
		outputs = []*tensor.Tensor{ln1, patchDense, ln2, posNorm, preProjection, embeddings}
	}
	results, err := graph.execute(outputs...)
	if err != nil {
		return Gemma4Output{}, fmt.Errorf("projector: execute Gemma 4 CUDA graph: %w", err)
	}
	if trace != nil {
		stages := []string{
			"patch_ln1", "patch_dense", "patch_ln2", "pos_norm", "pre_projection_norm", "embedding_projection",
		}
		for index, name := range stages {
			traceGemma4(trace, name, results[outputs[index]].Data)
		}
	}
	return Gemma4Output{
		Embeddings: results[embeddings], GridH: input.GridH, GridW: input.GridW,
	}, nil
}

func (r *Gemma4Runner) encodeAudioCUDA(
	ctx context.Context,
	frames []float32,
	rows int,
	spec Gemma4AudioSpec,
) (Gemma4AudioOutput, error) {
	builder := tensor.NewBuilder()
	input := builder.Input(
		"audio_frames", dtype.F32, tensor.MustShape(uint64(spec.SamplesPerToken), uint64(rows)),
	)
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	projection := graph.weight("mm.a.input_projection.weight")
	normed := (normalizationPlan{kind: normalizationRMS, epsilon: spec.RMSNormEpsilon, bf16: true}).graph(
		builder, builder.BF16Round(input), nil, nil,
	)
	embeddings := builder.BF16Round(builder.MulMat(projection, normed))
	graph.hostFeeds[input] = reference.Value{Shape: input.Shape, Data: frames}
	results, err := graph.execute(embeddings)
	if err != nil {
		return Gemma4AudioOutput{}, fmt.Errorf("projector: execute Gemma 4 audio CUDA graph: %w", err)
	}
	return Gemma4AudioOutput{Embeddings: results[embeddings]}, nil
}
