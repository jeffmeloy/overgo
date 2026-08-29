package tensorcatalog

import (
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

func Shape(shapes map[string][]int, name string, rank int) ([]int, error) {
	return Dimensions(shapes, name, rank)
}

// Dimensions returns one catalog entry with an exact rank.
func Dimensions(catalog map[string][]int, name string, rank int) ([]int, error) {
	dimensions, ok := catalog[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", name)
	}
	if !checked.Equal(len(dimensions), rank) {
		return nil, fmt.Errorf("tensor %q rank %d, want %d", name, len(dimensions), rank)
	}
	return dimensions, nil
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
	slices.Sort(indices)
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
	Ranks    []uint32
	NonEmpty bool
	Optional bool
	Storages []dtype.Type
}

type EqualAxes [2]uint32

type FixedAxis struct {
	Axis     uint32
	Extent   uint64
	Optional bool
}

func ValidateRelations(info gguf.TensorInfo, equal []EqualAxes, fixed []FixedAxis) error {
	shape := info.Extents()
	for _, relation := range equal {
		left, right := relation[0], relation[1]
		if left >= uint32(len(shape)) || right >= uint32(len(shape)) {
			return fmt.Errorf("tensor %q relation axes %d,%d exceed rank %d", info.Name, left, right, len(shape))
		}
		if shape[left] != shape[right] {
			return fmt.Errorf("tensor %q axes %d,%d differ: %d,%d", info.Name, left, right, shape[left], shape[right])
		}
	}
	for _, constraint := range fixed {
		if constraint.Axis >= uint32(len(shape)) {
			if constraint.Optional {
				continue
			}
			return fmt.Errorf("tensor %q axis %d exceeds rank %d", info.Name, constraint.Axis, len(shape))
		}
		if shape[constraint.Axis] != constraint.Extent {
			return fmt.Errorf("tensor %q axis %d extent %d, want %d", info.Name, constraint.Axis, shape[constraint.Axis], constraint.Extent)
		}
	}
	return nil
}

func ValidateInfo(info gguf.TensorInfo, requirement Requirement) error {
	if info.Dimensions > uint32(len(info.Shape)) {
		return fmt.Errorf("tensor %q rank %d exceeds stored shape", info.Name, info.Dimensions)
	}
	shape := info.Shape[:info.Dimensions]
	if requirement.Rank > 0 && info.Dimensions != requirement.Rank {
		return fmt.Errorf("tensor %q rank %d, want %d", info.Name, info.Dimensions, requirement.Rank)
	}
	if len(requirement.Ranks) > 0 && !slices.Contains(requirement.Ranks, info.Dimensions) {
		return fmt.Errorf("tensor %q rank %d, want one of %v", info.Name, info.Dimensions, requirement.Ranks)
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
