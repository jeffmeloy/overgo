package inference

import (
	"fmt"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

const (
	cacheKeyInputName   = "cache_key"
	cacheValueInputName = "cache_value"
)

type layerGraphCacheInputs struct {
	key        *tensor.Tensor
	value      *tensor.Tensor
	indexerKey *tensor.Tensor
	convState  *tensor.Tensor
	ssmState   *tensor.Tensor
	states     map[model.CacheStateName]*tensor.Tensor
}

func (r *Runner) hostLayerCacheInputs(
	builder *tensor.Builder,
	layer int,
	info model.LayerWeights,
	plan model.LayerPlan,
	past *LayerCache,
	feeds map[*tensor.Tensor]reference.Value,
) (layerGraphCacheInputs, error) {
	result := layerGraphCacheInputs{states: make(map[model.CacheStateName]*tensor.Tensor)}
	name := func(suffix string) string { return fmt.Sprintf("blk.%d.%s", layer, suffix) }
	input := func(suffix string, value reference.Value) *tensor.Tensor {
		node := builder.Input(name(suffix), dtype.F32, value.Shape)
		feeds[node] = value
		return node
	}
	zero := func(suffix string, shape tensor.Shape) *tensor.Tensor {
		elements, _ := shape.Elements()
		return input(suffix, reference.Value{Shape: shape, Data: make([]float32, int(elements))})
	}
	schema, err := model.CacheSchemaForPlan(r.spec, plan, info, 1)
	if err != nil {
		return layerGraphCacheInputs{}, err
	}
	if past != nil {
		result.key = input(cacheKeyInputName, past.Key)
		result.value = input(cacheValueInputName, past.Value)
		for stateName, state := range past.States {
			node := input(string(stateName), state.Value)
			switch stateName {
			case model.CacheStateIndexerKey:
				result.indexerKey = node
			case model.CacheStateConvolution:
				result.convState = node
			case model.CacheStateSSM:
				result.ssmState = node
			default:
				result.states[stateName] = node
			}
		}
	} else if schema.Primary.Key.Mode == model.CacheStateFixed {
		result.key = zero(cacheKeyInputName, schema.Primary.Key.Value.Shape)
		result.value = zero(cacheValueInputName, schema.Primary.Value.Value.Shape)
	}
	for stateName, stateSchema := range schema.States {
		if stateSchema.Mode != model.CacheStateFixed {
			continue
		}
		switch stateName {
		case model.CacheStateConvolution:
			if result.convState == nil {
				result.convState = zero(string(stateName), stateSchema.Value.Shape)
			}
		case model.CacheStateSSM:
			if result.ssmState == nil {
				result.ssmState = zero(string(stateName), stateSchema.Value.Shape)
			}
		}
	}
	return result, nil
}
