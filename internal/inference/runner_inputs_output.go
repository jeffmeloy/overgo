package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

const (
	chameleonTextTokenStart = 4
	chameleonTextTokenEnd   = 8196
)

func (r *Runner) loadEmbeddings(ctx context.Context, rows []uint32) (reference.Value, error) {
	return r.loadRows(ctx, r.weights.TokenEmbedding, rows)
}

func (r *Runner) prepareGemma4PerLayerInputs(
	ctx context.Context,
	activation reference.Value,
	rows []uint32,
) ([]reference.Value, error) {
	if !r.profile().Has(model.ArchitecturePerLayerEmbeddings) ||
		r.spec.EmbeddingPerLayer == 0 {
		return nil, nil
	}
	if r.weights.PerLayerTokenEmbedding == nil || r.weights.PerLayerModelProjection == nil ||
		r.weights.PerLayerProjectionNorm == nil {
		return nil, errors.New("inference: Gemma per-layer weights are incomplete")
	}
	selected, err := r.loadRows(ctx, *r.weights.PerLayerTokenEmbedding, rows)
	if err != nil {
		return nil, fmt.Errorf("inference: load Gemma per-layer embeddings: %w", err)
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("gemma4.per_layer.input", activation)
	selectedInput := runtime.input("gemma4.per_layer.selected", selected)
	projection, err := runtime.weight(*r.weights.PerLayerModelProjection)
	if err != nil {
		return nil, err
	}
	norm, err := runtime.weight(*r.weights.PerLayerProjectionNorm)
	if err != nil {
		return nil, err
	}
	outputs, err := model.BuildGemma4PerLayerInputs(
		runtime.builder, input, selectedInput, projection, norm, r.spec,
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

func (r *Runner) loadRows(
	ctx context.Context,
	info gguf.TensorInfo,
	rows []uint32,
) (reference.Value, error) {
	if !r.hasPreloadedWeights() {
		value, err := model.LoadHostRows(ctx, r.file, info, rows)
		if err != nil {
			return reference.Value{}, err
		}
		return r.applyLoRAEmbeddingRows(info.Name, rows, value)
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	table, err := runtime.weight(info)
	if err != nil {
		return reference.Value{}, err
	}
	output := runtime.builder.GetRows(table, rows)
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
	positionRows, err := r.loadRows(ctx, *r.weights.PositionEmbedding, positions)
	if err != nil {
		return reference.Value{}, fmt.Errorf("inference: load position embeddings: %w", err)
	}
	if positionRows.Shape != activation.Shape || len(positionRows.Data) != len(activation.Data) {
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
	typeRow, err := r.loadRows(ctx, *r.weights.TokenTypeEmbedding, []uint32{0})
	if err != nil {
		return reference.Value{}, fmt.Errorf("inference: load token-type embedding: %w", err)
	}
	width := int(activation.Shape.Dims[0])
	if typeRow.Shape.Rank != 2 || typeRow.Shape.Dims[0] != uint64(width) ||
		typeRow.Shape.Dims[1] != 1 || len(typeRow.Data) != width {
		return reference.Value{}, errors.New("inference: token-type embedding shape is incompatible")
	}
	for token := 0; token < int(activation.Shape.Dims[1]); token++ {
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
	output := model.ApplyNormalization(runtime.builder, input, weightInput, biasInput, r.spec)
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
			1024,
		)
		if err != nil {
			return nil, err
		}
		if err := r.applyLoRALogits(outputInfo.Name, hidden, logits); err != nil {
			return nil, err
		}
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
	if !r.profile().Has(model.ArchitectureDiscreteImageTokens) || r.spec.VocabularySize == 0 {
		return logits
	}
	vocabulary := int(r.spec.VocabularySize)
	if len(logits)%vocabulary != 0 {
		return logits
	}
	end := chameleonTextTokenEnd
	if end > vocabulary {
		end = vocabulary
	}
	for base := 0; base < len(logits); base += vocabulary {
		for token := chameleonTextTokenStart; token < end; token++ {
			logits[base+token] = -math.MaxFloat32
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
