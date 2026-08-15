package projector

import (
	"fmt"
	"maps"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/tensorcatalog"
)

const (
	visionPatchWeightTensor    = "v.patch_embd.weight"
	visionPatchWeightTensor1   = visionPatchWeightTensor + ".1"
	visionPatchBiasTensor      = "v.patch_embd.bias"
	visionPositionWeightTensor = "v.position_embd.weight"
	visionPreNormWeightTensor  = "v.pre_ln.weight"
	visionPreNormBiasTensor    = "v.pre_ln.bias"
	visionPostNormWeightTensor = "v.post_ln.weight"
	visionPostNormBiasTensor   = "v.post_ln.bias"
)

type tensorPresence uint8

const (
	tensorOptional tensorPresence = iota
	tensorRequired
)

func addSpatialVisionEmbeddingCatalog(
	file *gguf.File,
	required map[string][]uint64,
	spec visionBackboneSpec,
	positions int,
	bias tensorPresence,
) {
	required[visionPatchWeightTensor] = []uint64{
		uint64(spec.PatchSize), uint64(spec.PatchSize), rgbChannelCount, uint64(spec.Hidden),
	}
	required[visionPositionWeightTensor] = []uint64{uint64(spec.Hidden), uint64(positions)}
	addProjectorTensor(file, required, visionPatchBiasTensor, []uint64{uint64(spec.Hidden)}, bias)
}

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

func addProjectorTensor(
	file *gguf.File,
	required map[string][]uint64,
	name string,
	shape []uint64,
	presence tensorPresence,
) {
	if presence == tensorRequired {
		required[name] = shape
	} else {
		addOptionalProjectorTensor(file, required, name, shape)
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

func addOptionalVisionNormCatalog(file *gguf.File, required map[string][]uint64, hidden int) error {
	shape := []uint64{uint64(hidden)}
	if err := addOptionalProjectorPair(file, required, visionPreNormWeightTensor, visionPreNormBiasTensor, shape); err != nil {
		return err
	}
	return addOptionalProjectorPair(file, required, visionPostNormWeightTensor, visionPostNormBiasTensor, shape)
}

func addStandardVisionLayerCatalog(
	file *gguf.File,
	required map[string][]uint64,
	layers, hidden, intermediate int,
	fusedQKV []bool,
	defaultFused bool,
	bias tensorPresence,
) {
	for layer := range layers {
		fused := defaultFused
		if fusedQKV != nil {
			fused = fusedQKV[layer]
		}
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		hiddenShape := []uint64{uint64(hidden)}
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
		if fused {
			required[prefix+"attn_qkv.weight"] = []uint64{uint64(hidden), uint64(3 * hidden)}
			addProjectorTensor(file, required, prefix+"attn_qkv.bias", []uint64{uint64(3 * hidden)}, bias)
		} else {
			for _, part := range []string{"q", "k", "v"} {
				required[prefix+"attn_"+part+".weight"] = []uint64{uint64(hidden), uint64(hidden)}
				addProjectorTensor(file, required, prefix+"attn_"+part+".bias", hiddenShape, bias)
			}
		}
		addProjectorTensor(file, required, prefix+"attn_out.bias", hiddenShape, bias)
		addProjectorTensor(file, required, prefix+"ffn_up.bias", []uint64{uint64(intermediate)}, bias)
		addProjectorTensor(file, required, prefix+"ffn_down.bias", hiddenShape, bias)
	}
}
