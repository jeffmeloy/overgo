package tensorcatalog

import (
	"fmt"
	"slices"
	"sort"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

type Requirement struct {
	Name     string
	Shapes   [][]uint64
	Optional bool
	Storages []dtype.Type
}

func ValidateInfo(info gguf.TensorInfo, requirement Requirement) error {
	if info.Dimensions > uint32(len(info.Shape)) {
		return fmt.Errorf("tensor %q rank %d exceeds stored shape", info.Name, info.Dimensions)
	}
	shape := info.Shape[:info.Dimensions]
	shapeOK := false
	for _, expected := range requirement.Shapes {
		if slices.Equal(shape, expected) {
			shapeOK = true
			break
		}
	}
	if !shapeOK {
		return fmt.Errorf("tensor %q shape %v, want one of %v", info.Name, shape, requirement.Shapes)
	}
	if len(requirement.Storages) > 0 && !slices.Contains(requirement.Storages, info.Type) {
		return fmt.Errorf("tensor %q storage %s, want one of %v", info.Name, info.Type, requirement.Storages)
	}
	return nil
}

func Validate(
	tensors map[string]gguf.TensorInfo,
	prefix string,
	requirements []Requirement,
) error {
	ordered := slices.Clone(requirements)
	sort.SliceStable(ordered, func(left, right int) bool { return ordered[left].Name < ordered[right].Name })
	for _, requirement := range ordered {
		name := prefix + requirement.Name
		info, ok := tensors[name]
		if !ok {
			if requirement.Optional {
				continue
			}
			return fmt.Errorf("missing tensor %q", name)
		}
		requirement.Name = name
		if err := ValidateInfo(info, requirement); err != nil {
			return err
		}
	}
	return nil
}
