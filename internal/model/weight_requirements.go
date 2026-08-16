package model

import (
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensorcatalog"
)

// tensorRequirement: ordered catalog binding
type tensorRequirement struct {
	name        string
	shapes      [][]uint64
	rank        uint32
	nonempty    bool
	storages    []dtype.Type
	destination *gguf.TensorInfo
	pointer     **gguf.TensorInfo
	optional    bool
}

func optionalRelationalTensorPointer(
	name string,
	destination **gguf.TensorInfo,
	rank uint32,
	nonempty bool,
	storages ...dtype.Type,
) tensorRequirement {
	return tensorRequirement{
		name: name, rank: rank, nonempty: nonempty, storages: storages,
		pointer: destination, optional: true,
	}
}

func requiredTensor(name string, destination *gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return tensorRequirement{name: name, shapes: [][]uint64{shape}, destination: destination}
}

func optionalTensor(name string, destination *gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return tensorRequirement{name: name, shapes: [][]uint64{shape}, destination: destination, optional: true}
}

func requiredTensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return tensorRequirement{name: name, shapes: [][]uint64{shape}, pointer: destination}
}

func optionalTensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return tensorRequirement{name: name, shapes: [][]uint64{shape}, pointer: destination, optional: true}
}

func optionalF32TensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return optionalStoredTensorPointer(name, destination, dtype.F32, shape...)
}

func requiredF32TensorPointer(name string, destination **gguf.TensorInfo, shape ...uint64) tensorRequirement {
	return requiredStoredTensorPointer(name, destination, dtype.F32, shape...)
}

func requiredStoredTensorPointer(
	name string,
	destination **gguf.TensorInfo,
	storage dtype.Type,
	shape ...uint64,
) tensorRequirement {
	return tensorRequirement{name: name, shapes: [][]uint64{shape}, storages: []dtype.Type{storage}, pointer: destination}
}

func optionalStoredTensorPointer(
	name string,
	destination **gguf.TensorInfo,
	storage dtype.Type,
	shape ...uint64,
) tensorRequirement {
	requirement := requiredStoredTensorPointer(name, destination, storage, shape...)
	requirement.optional = true
	return requirement
}

func requiredTensorPointerShapes(
	name string,
	destination **gguf.TensorInfo,
	shapes ...[]uint64,
) tensorRequirement {
	return tensorRequirement{name: name, shapes: shapes, pointer: destination}
}

func requiredTensorShapes(
	name string,
	destination *gguf.TensorInfo,
	shapes ...[]uint64,
) tensorRequirement {
	return tensorRequirement{name: name, shapes: shapes, destination: destination}
}

func loadTensorRequirements(
	catalog weightCatalog,
	prefix string,
	requirements []tensorRequirement,
) error {
	for _, requirement := range requirements {
		name := prefix + requirement.name
		item, ok := catalog.tensor(name)
		if !ok {
			if requirement.optional {
				continue
			}
			return fmt.Errorf("required tensor %q is missing", name)
		}
		catalogRequirement := tensorcatalog.Requirement{
			Name: name, Shapes: requirement.shapes, Rank: requirement.rank,
			NonEmpty: requirement.nonempty, Storages: requirement.storages,
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

func validateTensorInfo(
	item gguf.TensorInfo,
	storages []dtype.Type,
	shapes ...[]uint64,
) error {
	return tensorcatalog.ValidateInfo(item, tensorcatalog.Requirement{
		Name: item.Name, Shapes: shapes, Storages: storages,
	})
}
