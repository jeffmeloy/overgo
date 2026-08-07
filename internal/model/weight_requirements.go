package model

import (
	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensorcatalog"
)

// tensorRequirement: ordered catalog binding
type tensorRequirement struct {
	name        string
	shape       []uint64
	destination *gguf.TensorInfo
	pointer     **gguf.TensorInfo
	optional    bool
	f32         bool
}

func requiredTensor(name string, destination *gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return tensorRequirement{name: name, shape: shape, destination: destination}
}

func optionalTensor(name string, destination *gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return tensorRequirement{name: name, shape: shape, destination: destination, optional: true}
}

func requiredTensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return tensorRequirement{name: name, shape: shape, pointer: destination}
}

func optionalTensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return tensorRequirement{name: name, shape: shape, pointer: destination, optional: true}
}

func optionalF32TensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return tensorRequirement{name: name, shape: shape, pointer: destination, optional: true, f32: true}
}

func requiredF32TensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return tensorRequirement{name: name, shape: shape, pointer: destination, f32: true}
}

func loadTensorRequirements(
	load weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	requirements []tensorRequirement,
) error {
	for _, requirement := range requirements {
		name := prefix + requirement.name
		if requirement.optional {
			if _, ok := tensors[name]; !ok {
				continue
			}
		}
		item, err := load(name, requirement.shape...)
		if err != nil {
			return err
		}
		catalogRequirement := tensorcatalog.Requirement{
			Name: name, Shape: requirement.shape, Storage: dtype.F32, CheckStorage: requirement.f32,
		}
		if err := tensorcatalog.ValidateInfo(item, catalogRequirement); err != nil {
			return err
		}
		if requirement.destination != nil {
			*requirement.destination = item
		} else {
			*requirement.pointer = &item
		}
	}
	return nil
}
