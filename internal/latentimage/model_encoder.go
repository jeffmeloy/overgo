package latentimage

import (
	"fmt"
	"overgo/internal/gguf"
	"overgo/internal/hfgguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// modelEncoderProgram binds a full-prompt layer range to the shared model plan.
// The terminal range includes final normalization and every range starts with zero recurrent state.
type modelEncoderProgram struct {
	input, output *tensor.Tensor
	boundaries    map[string]*tensor.Tensor
	weights       tensor.WeightBindings
	feeds         map[*tensor.Tensor]reference.Value
}

func compileModelEncoder(catalog hfgguf.LanguageCatalog, tokens, first, end int) (*modelEncoderProgram, error) {
	if catalog.Output != hfgguf.LanguageHidden || catalog.Spec.Arithmetic != model.ArithmeticTransformersBF16 {
		return nil, fmt.Errorf("model encoder requires a native BF16 hidden-state catalog")
	}
	plan, err := model.CompileModelPlanWithProfile(catalog.Spec, catalog.Weights, catalog.Spec.Profile())
	if err != nil {
		return nil, err
	}
	if tokens <= 0 || first < 0 || end < first || end > plan.LayerCount() || first == end && end != plan.LayerCount() {
		return nil, fmt.Errorf("model encoder refuses tokens=%d range=[%d,%d) for %d layers", tokens, first, end, plan.LayerCount())
	}
	p := &modelEncoderProgram{boundaries: map[string]*tensor.Tensor{}, feeds: map[*tensor.Tensor]reference.Value{}}
	b := tensor.NewBuilder()
	bind := func(info *gguf.TensorInfo) (*tensor.Tensor, error) {
		if info == nil || info.Name == "" {
			return nil, nil
		}
		storage := dtype.F32
		switch info.Type {
		case gguf.DTypeBF16:
			storage = dtype.BF16
		case gguf.DTypeF32:
		default:
			return nil, fmt.Errorf("model encoder weight %s has unsupported storage %v", info.Name, info.Type)
		}
		shape, err := tensor.NewShape(info.Shape[:info.Dimensions]...)
		if err != nil {
			return nil, err
		}
		node := b.Input(info.Name, storage, shape)
		p.weights = append(p.weights, node)
		return node, b.Err()
	}
	p.input = b.Input("conditioning.hidden", dtype.F32, tensor.MustShape(uint64(catalog.Spec.EmbeddingLength), uint64(tokens)))
	current := p.input
	positions := make([]uint32, tokens)
	for i := range positions {
		positions[i] = uint32(i)
	}
	for layer := first; layer < end; layer++ {
		info := catalog.Weights.Layers[layer]
		weights := model.LayerGraphWeights{}
		if err := model.BindLayerGraphInventory(&info, &weights, func(_ model.LayerBindingSlot, info *gguf.TensorInfo) (*tensor.Tensor, error) { return bind(info) }); err != nil {
			return nil, err
		}
		context := model.CachedBlockContext{Builder: b, Input: current, Positions: positions, Layer: uint32(layer), Recurrent: info.Recurrent, CacheWrite: tensor.CacheWriteConcat, Sequences: 1}
		if info.Recurrent {
			schema, err := plan.CacheSchema(uint32(layer), 1)
			if err != nil {
				return nil, err
			}
			zero := func(label string, shape tensor.Shape) *tensor.Tensor {
				node := b.Input(fmt.Sprintf("conditioning.%d.%s", layer, label), dtype.F32, shape)
				n, _ := shape.Elements()
				p.feeds[node] = reference.Value{Shape: shape, Data: make([]float32, n)}
				return node
			}
			context.PastKey = zero("convolution", schema.Primary.Key.Value.Shape)
			context.PastValue = zero("state", schema.Primary.Value.Value.Shape)
		}
		program, err := plan.LayerProgram(uint32(layer))
		if err != nil {
			return nil, err
		}
		result, err := program.Build(context, weights)
		if err != nil {
			return nil, err
		}
		current = result.Output
		p.boundaries[fmt.Sprintf("layer.%d", layer)] = current
	}
	if end == plan.LayerCount() {
		weight, err := bind(&catalog.Weights.OutputNorm)
		if err != nil {
			return nil, err
		}
		current = plan.Normalization().Apply(b, current, weight, nil)
		p.boundaries["final_norm"] = current
	}
	p.output = current
	return p, b.Err()
}
