package model

import (
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensorcatalog"
)

const (
	attentionNormWeightTensor     = "attn_norm.weight"
	attentionQueryWeightTensor    = "attn_q.weight"
	attentionKeyWeightTensor      = "attn_k.weight"
	attentionValueWeightTensor    = "attn_v.weight"
	attentionOutputWeightTensor   = "attn_output.weight"
	attentionQueryNormTensor      = "attn_q_norm.weight"
	attentionKeyNormTensor        = "attn_k_norm.weight"
	feedForwardNormWeightTensor   = "ffn_norm.weight"
	feedForwardGateWeightTensor   = "ffn_gate.weight"
	feedForwardUpWeightTensor     = "ffn_up.weight"
	feedForwardDownWeightTensor   = "ffn_down.weight"
	postAttentionNormWeightTensor = "post_attention_norm.weight"
	tokenEmbeddingWeightTensor    = "token_embd.weight"
	outputNormWeightTensor        = "output_norm.weight"
	outputWeightTensor            = "output.weight"
)

// tensorBinding: compiled catalog slot.
type tensorBinding struct {
	name        string
	shapes      [][]uint64
	rank        uint32
	nonempty    bool
	storages    []dtype.Type
	destination *gguf.TensorInfo
	pointer     **gguf.TensorInfo
	optional    bool
	index       int
	present     bool
}

func optionalRelationalTensorPointer(
	name string,
	destination **gguf.TensorInfo,
	rank uint32,
	nonempty bool,
	storages ...dtype.Type,
) tensorBinding {
	return tensorBinding{
		name: name, rank: rank, nonempty: nonempty, storages: storages,
		pointer: destination, optional: true,
	}
}

func requiredTensor(name string, destination *gguf.TensorInfo, shape ...uint64) tensorBinding {
	return tensorBinding{name: name, shapes: [][]uint64{shape}, destination: destination}
}

func optionalTensor(name string, destination *gguf.TensorInfo, shape ...uint64) tensorBinding {
	return tensorBinding{name: name, shapes: [][]uint64{shape}, destination: destination, optional: true}
}

func requiredTensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorBinding {
	return tensorBinding{name: name, shapes: [][]uint64{shape}, pointer: destination}
}

func optionalTensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorBinding {
	return tensorBinding{name: name, shapes: [][]uint64{shape}, pointer: destination, optional: true}
}

func optionalF32TensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorBinding {
	return optionalStoredTensorPointer(name, destination, dtype.F32, shape...)
}

func requiredF32TensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorBinding {
	return requiredStoredTensorPointer(name, destination, dtype.F32, shape...)
}

func requiredStoredTensorPointer(
	name string,
	destination **gguf.TensorInfo,
	storage dtype.Type,
	shape ...uint64,
) tensorBinding {
	return tensorBinding{name: name, shapes: [][]uint64{shape}, storages: []dtype.Type{storage}, pointer: destination}
}

func optionalStoredTensorPointer(
	name string,
	destination **gguf.TensorInfo,
	storage dtype.Type,
	shape ...uint64,
) tensorBinding {
	requirement := requiredStoredTensorPointer(name, destination, storage, shape...)
	requirement.optional = true
	return requirement
}

func requiredTensorPointerShapes(
	name string,
	destination **gguf.TensorInfo,
	shapes ...[]uint64,
) tensorBinding {
	return tensorBinding{name: name, shapes: shapes, pointer: destination}
}

func bindTensorProgram(
	catalog weightCatalog,
	prefix string,
	program []tensorBinding,
) error {
	for bindingIndex := range program {
		binding := &program[bindingIndex]
		if (binding.destination == nil) == (binding.pointer == nil) {
			return fmt.Errorf("tensor binding %q has invalid destination", prefix+binding.name)
		}
		name := prefix + binding.name
		itemIndex, ok := catalog.tensors[name]
		if !ok {
			if binding.optional {
				continue
			}
			return fmt.Errorf("required tensor %q is missing", name)
		}
		item := catalog.items[itemIndex]
		catalogRequirement := tensorcatalog.Requirement{
			Name: name, Shapes: binding.shapes, Rank: binding.rank,
			NonEmpty: binding.nonempty, Storages: binding.storages,
		}
		if err := tensorcatalog.ValidateInfo(item, catalogRequirement); err != nil {
			return err
		}
		binding.index, binding.present = itemIndex, true
	}
	for bindingIndex := range program {
		binding := &program[bindingIndex]
		if !binding.present {
			continue
		}
		if binding.destination != nil {
			*binding.destination = catalog.items[binding.index]
		} else {
			*binding.pointer = &catalog.items[binding.index]
		}
	}
	return nil
}

func validateTensorInfo(
	item gguf.TensorInfo,
	storages []dtype.Type,
	shapes ...[]uint64,
) error {
	return tensorcatalog.ValidateInfo(item, tensorcatalog.Requirement{
		Name: item.Name, Shapes: shapes, Storages: storages,
	})
}
