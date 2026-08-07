package inference

import (
	"fmt"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
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
		return input(suffix, reference.ZeroValue(shape))
	}
	_, schema, err := r.cacheSchema(layer, 1)
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
