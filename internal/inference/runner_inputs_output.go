package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

func (r *Runner) preparePerLayerInputs(
	ctx context.Context,
	activation reference.Value,
	rows []uint32,
) ([]reference.Value, error) {
	if !r.program.Model.ProjectedInput().PerLayerEmbeddings {
		return nil, nil
	}
	if r.weights.PerLayerTokenEmbedding == nil || r.weights.PerLayerModelProjection == nil ||
		r.weights.PerLayerProjectionNorm == nil {
		return nil, errors.New("inference: per-layer input weights are incomplete")
	}
	selected, err := r.gatherTensor(ctx, *r.weights.PerLayerTokenEmbedding, rows)
	if err != nil {
		return nil, fmt.Errorf("inference: load per-layer embeddings: %w", err)
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("per_layer.input", activation)
	selectedInput := runtime.input("per_layer.selected", selected)
	projection, err := runtime.weight(*r.weights.PerLayerModelProjection)
	if err != nil {
		return nil, err
	}
	norm, err := runtime.weight(*r.weights.PerLayerProjectionNorm)
	if err != nil {
		return nil, err
	}
	outputs, err := r.program.Model.BuildPerLayerInputs(
		runtime.builder, input, selectedInput, projection, norm,
	)
	if err != nil {
		return nil, err
	}
	results, err := runtime.execute(outputs...)
	if err != nil {
		return nil, err
	}
	values := make([]reference.Value, len(outputs))
	for index, output := range outputs {
		values[index] = results[output]
	}
	return values, nil
}

func (r *Runner) gatherTensor(
	ctx context.Context,
	info gguf.TensorInfo,
	indices []uint32,
) (reference.Value, error) {
	if !r.hasPreloadedWeights() {
		value, err := model.LoadHostRows(ctx, r.file, info, indices)
		if err != nil {
			return reference.Value{}, err
		}
		return r.applyLoRAEmbeddingSelection(info.Name, indices, value)
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	table, err := runtime.weight(info)
	if err != nil {
		return reference.Value{}, err
	}
	output := runtime.builder.GetRows(table, indices)
	if err := runtime.builder.Err(); err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *Runner) addPositionEmbeddings(
	ctx context.Context,
	activation reference.Value,
	positions []uint32,
) (reference.Value, error) {
	if r.weights.PositionEmbedding == nil {
		return activation, nil
	}
	if err := validateLearnedPositions(positions, r.spec.ContextLength); err != nil {
		return reference.Value{}, err
	}
	positionRows, err := r.gatherTensor(ctx, *r.weights.PositionEmbedding, positions)
	if err != nil {
		return reference.Value{}, fmt.Errorf("inference: load position embeddings: %w", err)
	}
	if !positionRows.Shape.Equal(activation.Shape) ||
		validateStateValue(positionRows) != nil || validateStateValue(activation) != nil {
		return reference.Value{}, errors.New("inference: position embedding shape differs from token embeddings")
	}
	for index := range activation.Data {
		activation.Data[index] += positionRows.Data[index]
	}
	return activation, nil
}

func (r *Runner) addTokenTypeEmbedding(
	ctx context.Context,
	activation reference.Value,
) (reference.Value, error) {
	if r.weights.TokenTypeEmbedding == nil {
		return activation, nil
	}
	typeRow, err := r.gatherTensor(ctx, *r.weights.TokenTypeEmbedding, []uint32{0})
	if err != nil {
		return reference.Value{}, fmt.Errorf("inference: load token-type embedding: %w", err)
	}
	width, tokens, valid := activation.MatrixExtents()
	if !valid || !tensor.IsMatrix(typeRow.Shape, uint64(width), tensor.SingletonExtent) ||
		validateStateValue(typeRow) != nil {
		return reference.Value{}, errors.New("inference: token-type embedding shape is incompatible")
	}
	for token := range tokens {
		start := token * width
		for index, value := range typeRow.Data {
			activation.Data[start+index] += value
		}
	}
	return activation, nil
}

func (r *Runner) applyTokenEmbeddingNorm(
	ctx context.Context,
	activation reference.Value,
) (reference.Value, error) {
	if r.weights.TokenEmbeddingNorm == nil {
		return activation, nil
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("token_embd_norm.input", activation)
	weightInput, err := runtime.weight(*r.weights.TokenEmbeddingNorm)
	if err != nil {
		return reference.Value{}, err
	}
	var biasInput *tensor.Tensor
	if r.weights.TokenEmbeddingNormBias != nil {
		biasInput, err = runtime.weight(*r.weights.TokenEmbeddingNormBias)
		if err != nil {
			return reference.Value{}, err
		}
	}
	output := r.program.Model.Normalization().Apply(runtime.builder, input, weightInput, biasInput)
	if err := runtime.builder.Err(); err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *Runner) logits(
	ctx context.Context,
	outputInfo gguf.TensorInfo,
	hidden []float32,
) ([]float32, error) {
	if !r.hasPreloadedWeights() {
		logits, err := model.DotRows(
			ctx,
			r.file,
			outputInfo,
			hidden,
		)
		if err != nil {
			return nil, err
		}
		r.applyLoRALogits(outputInfo.Name, hidden, logits)
		if err := addOutputBias(logits, r.outputBias); err != nil {
			return nil, err
		}
		scaleLogits(logits, r.spec.OutputLogitMultiplier())
		return r.finalizeLogits(logits), nil
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	table, err := runtime.weight(outputInfo)
	if err != nil {
		return nil, err
	}
	inputShape := tensor.MustShape(uint64(len(hidden)), 1)
	inputValue, err := reference.NewValue(inputShape, hidden)
	if err != nil {
		return nil, err
	}
	input := runtime.input("logits.input", inputValue)
	output := runtime.builder.MulMat(table, input)
	if r.weights.OutputBias != nil {
		bias, biasErr := runtime.weight(*r.weights.OutputBias)
		if biasErr != nil {
			return nil, biasErr
		}
		output = runtime.builder.Add(output, bias)
	}
	if scale := r.spec.OutputLogitMultiplier(); scale != 1 {
		output = runtime.builder.Scale(output, scale)
	}
	if err := runtime.builder.Err(); err != nil {
		return nil, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return nil, err
	}
	return r.finalizeLogits(results[output].Data), nil
}

func scaleLogits(logits []float32, scale float32) {
	if scale == 1 {
		return
	}
	for index := range logits {
		logits[index] *= scale
	}
}

func applyLogitSoftcap(logits []float32, cap float32) []float32 {
	if cap <= 0 {
		return logits
	}
	for index, value := range logits {
		logits[index] = cap * float32(math.Tanh(float64(value/cap)))
	}
	return logits
}

func (r *Runner) finalizeLogits(logits []float32) []float32 {
	logits = applyLogitSoftcap(logits, r.spec.FinalLogitSoftcap)
	if len(r.outputExclusions) == 0 || r.spec.VocabularySize == 0 {
		return logits
	}
	vocabulary := int(r.spec.VocabularySize)
	if len(logits)%vocabulary != 0 {
		return logits
	}
	for base := 0; base < len(logits); base += vocabulary {
		for _, span := range r.outputExclusions {
			start, end := int(span.Start), min(int(span.End), vocabulary)
			for token := start; token < end; token++ {
				logits[base+token] = -math.MaxFloat32
			}
		}
	}
	return logits
}

func addOutputBias(logits, bias []float32) error {
	if len(bias) == 0 {
		return nil
	}
	if len(logits)%len(bias) != 0 {
		return fmt.Errorf(
			"inference: %d logits are not divisible by output bias length %d",
			len(logits),
			len(bias),
		)
	}
	for index := range logits {
		logits[index] += bias[index%len(bias)]
	}
	return nil
}
