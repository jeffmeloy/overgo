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
	key    *tensor.Tensor
	value  *tensor.Tensor
	states model.CacheStates[*tensor.Tensor]
}

type layerCacheSource[T any] struct {
	primary model.CachePair[T]
	states  model.CacheStates[T]
}

func bindLayerGraphCacheInputs[T any](
	schema model.LayerCacheSchema,
	past *layerCacheSource[T],
	input func(string, T) *tensor.Tensor,
	zero func(string, tensor.Shape) *tensor.Tensor,
) layerGraphCacheInputs {
	result := layerGraphCacheInputs{states: make(model.CacheStates[*tensor.Tensor])}
	if past != nil {
		result.key = input(cacheKeyInputName, past.primary.Key)
		result.value = input(cacheValueInputName, past.primary.Value)
		for stateName, state := range past.states {
			result.states[stateName] = model.CacheState[*tensor.Tensor]{
				Mode: state.Mode, Value: input(string(stateName), state.Value),
			}
		}
	} else if schema.Primary.Key.Mode == model.CacheStateFixed {
		result.key = zero(cacheKeyInputName, schema.Primary.Key.Value.Shape)
		result.value = zero(cacheValueInputName, schema.Primary.Value.Value.Shape)
	}
	for stateName, stateSchema := range schema.States {
		if !stateSchema.Value.ZeroInitial || result.states[stateName].Value != nil {
			continue
		}
		result.states[stateName] = model.CacheState[*tensor.Tensor]{
			Mode: stateSchema.Mode, Value: zero(string(stateName), stateSchema.Value.Shape),
		}
	}
	return result
}

func (r *Runner) hostLayerCacheInputs(
	builder *tensor.Builder,
	layer int,
	info model.LayerWeights,
	plan model.LayerPlan,
	past *LayerCache,
	feeds map[*tensor.Tensor]reference.Value,
) (layerGraphCacheInputs, error) {
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
	var source *layerCacheSource[reference.Value]
	if past != nil {
		source = &layerCacheSource[reference.Value]{
			primary: model.NewCachePair(past.Key, past.Value), states: past.States,
		}
	}
	return bindLayerGraphCacheInputs(schema, source, input, zero), nil
}
