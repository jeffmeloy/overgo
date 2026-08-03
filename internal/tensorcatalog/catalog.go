package tensorcatalog

import (
	"fmt"
	"slices"
	"sort"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/dtype"
)

type Requirement struct {
	Name         string
	Shape        []uint64
	Optional     bool
	Storage      dtype.Type
	CheckStorage bool
}

func ValidateInfo(info gguf.TensorInfo, requirement Requirement) error {
	if info.Dimensions > uint32(len(info.Shape)) || int(info.Dimensions) != len(requirement.Shape) {
		return fmt.Errorf("tensor %q rank %d, want %d", info.Name, info.Dimensions, len(requirement.Shape))
	}
	if !slices.Equal(info.Shape[:info.Dimensions], requirement.Shape) {
		return fmt.Errorf("tensor %q shape %v, want %v", info.Name, info.Shape[:info.Dimensions], requirement.Shape)
	}
	if requirement.CheckStorage && info.Type != requirement.Storage {
		return fmt.Errorf("tensor %q must use %s storage", info.Name, requirement.Storage)
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
