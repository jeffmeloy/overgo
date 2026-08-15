package tensorcatalog

import (
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func Shape(shapes map[string][]int, name string, rank int) ([]int, error) {
	shape, ok := shapes[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", name)
	}
	if len(shape) != rank {
		return nil, fmt.Errorf("tensor %q rank %d, want %d", name, len(shape), rank)
	}
	return shape, nil
}

// IndexedCount validates contiguous prefix+index+suffix names.
func IndexedCount(shapes map[string][]int, prefix, suffix string) (int, error) {
	var indices []int
	for name := range shapes {
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, suffix) {
			continue
		}
		indexText := strings.TrimSuffix(strings.TrimPrefix(name, prefix), suffix)
		index, err := strconv.Atoi(indexText)
		if err != nil || index < 0 {
			return 0, fmt.Errorf("malformed indexed tensor %q", name)
		}
		indices = append(indices, index)
	}
	if len(indices) == 0 {
		return 0, fmt.Errorf("no indexed tensors under %q", prefix)
	}
	sort.Ints(indices)
	for expected, index := range indices {
		if index != expected {
			return 0, fmt.Errorf("tensor indices under %q are not contiguous from zero: %v", prefix, indices)
		}
	}
	return len(indices), nil
}

type Requirement struct {
	Name     string
	Shapes   [][]uint64
	Rank     uint32
	NonEmpty bool
	Optional bool
	Storages []dtype.Type
}

func ValidateInfo(info gguf.TensorInfo, requirement Requirement) error {
	if info.Dimensions > uint32(len(info.Shape)) {
		return fmt.Errorf("tensor %q rank %d exceeds stored shape", info.Name, info.Dimensions)
	}
	shape := info.Shape[:info.Dimensions]
	if requirement.Rank > 0 && info.Dimensions != requirement.Rank {
		return fmt.Errorf("tensor %q rank %d, want %d", info.Name, info.Dimensions, requirement.Rank)
	}
	if requirement.NonEmpty {
		for _, extent := range shape {
			if extent == 0 {
				return fmt.Errorf("tensor %q shape %v is empty", info.Name, shape)
			}
		}
	}
	if len(requirement.Shapes) > 0 {
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
