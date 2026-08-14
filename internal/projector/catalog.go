package projector

import (
	"fmt"
	"maps"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/tensorcatalog"
)

func validateProjectorTensorShapes(file *gguf.File, required map[string][]uint64) error {
	tensors := make(map[string]gguf.TensorInfo, len(required))
	requirements := make([]tensorcatalog.Requirement, 0, len(required))
	for name, shape := range required {
		if info, ok := file.Tensor(name); ok {
			tensors[name] = info
		}
		requirements = append(requirements, tensorcatalog.Requirement{Name: name, Shapes: [][]uint64{shape}})
	}
	if err := tensorcatalog.Validate(tensors, "", requirements); err != nil {
		return fmt.Errorf("projector: %w", err)
	}
	return nil
}

func validateProjectorTensorCatalog(
	file *gguf.File,
	required map[string][]uint64,
	hostOnly ...string,
) ([]string, error) {
	if err := validateProjectorTensorShapes(file, required); err != nil {
		return nil, err
	}
	names := slices.Sorted(maps.Keys(required))
	if len(hostOnly) == 0 {
		return names, nil
	}
	excluded := make(map[string]struct{}, len(hostOnly))
	for _, name := range hostOnly {
		excluded[name] = struct{}{}
	}
	return slices.DeleteFunc(names, func(name string) bool {
		_, ok := excluded[name]
		return ok
	}), nil
}

func addOptionalProjectorTensor(
	file *gguf.File,
	required map[string][]uint64,
	name string,
	shape []uint64,
) {
	if hasTensor(file, name) {
		required[name] = shape
	}
}

func addOptionalProjectorPair(
	file *gguf.File,
	required map[string][]uint64,
	first, second string,
	shape []uint64,
) error {
	hasFirst, hasSecond := hasTensor(file, first), hasTensor(file, second)
	if hasFirst != hasSecond {
		return fmt.Errorf("projector: tensors %q and %q must be paired", first, second)
	}
	if hasFirst {
		required[first], required[second] = shape, shape
	}
	return nil
}

func addStandardVisionLayerCatalog(
	file *gguf.File,
	required map[string][]uint64,
	layers, hidden, intermediate int,
	fusedQKV []bool,
) {
	hiddenShape := []uint64{uint64(hidden)}
	for layer := range layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for name, shape := range map[string][]uint64{
			"attn_out.weight": {uint64(hidden), uint64(hidden)},
			"ffn_up.weight":   {uint64(hidden), uint64(intermediate)},
			"ffn_down.weight": {uint64(intermediate), uint64(hidden)},
			"ln1.weight":      hiddenShape,
			"ln1.bias":        hiddenShape,
			"ln2.weight":      hiddenShape,
			"ln2.bias":        hiddenShape,
		} {
			required[prefix+name] = shape
		}
		if fusedQKV[layer] {
			required[prefix+"attn_qkv.weight"] = []uint64{uint64(hidden), uint64(3 * hidden)}
			addOptionalProjectorTensor(file, required, prefix+"attn_qkv.bias", []uint64{uint64(3 * hidden)})
		} else {
			for _, part := range []string{"q", "k", "v"} {
				required[prefix+"attn_"+part+".weight"] = []uint64{uint64(hidden), uint64(hidden)}
				addOptionalProjectorTensor(file, required, prefix+"attn_"+part+".bias", hiddenShape)
			}
		}
		addOptionalProjectorTensor(file, required, prefix+"attn_out.bias", hiddenShape)
		addOptionalProjectorTensor(file, required, prefix+"ffn_up.bias", []uint64{uint64(intermediate)})
		addOptionalProjectorTensor(file, required, prefix+"ffn_down.bias", hiddenShape)
	}
}
