//overgo:runtime-inputs caller

package projector

import (
	"fmt"
	"maps"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/tensorcatalog"
)

const (
	visionPatchWeightTensor      = "v.patch_embd.weight"
	visionPatchWeightTensor1     = visionPatchWeightTensor + ".1"
	visionPatchBiasTensor        = "v.patch_embd.bias"
	visionPositionWeightTensor   = "v.position_embd.weight"
	visionPreNormWeightTensor    = "v.pre_ln.weight"
	visionPreNormBiasTensor      = "v.pre_ln.bias"
	visionPostNormWeightTensor   = "v.post_ln.weight"
	visionPostNormBiasTensor     = "v.post_ln.bias"
	projectionFirstWeightTensor  = "mm.0.weight"
	projectionFirstBiasTensor    = "mm.0.bias"
	projectionSecondWeightTensor = "mm.2.weight"
	projectionSecondBiasTensor   = "mm.2.bias"
	multimodalProjectionWeight   = "mm.model.fc.weight"
	multimodalProjectionBias     = "mm.model.fc.bias"
	multimodalInputProjection    = "mm.input_projection.weight"
	visionInputTensor            = "pixel_values"
	visionClassEmbeddingTensor   = "v.class_embd"
	visionImageNewlineTensor     = "v.image_newline"
	visionViewSeparatorTensor    = "v.view_seperator"
)

type tensorPresence uint8

type tensorPair struct {
	weight string
	bias   string
}

var splitAttentionTensors = [...]tensorPair{
	{weight: "attn_q.weight", bias: "attn_q.bias"},
	{weight: "attn_k.weight", bias: "attn_k.bias"},
	{weight: "attn_v.weight", bias: "attn_v.bias"},
}

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
		uint64(spec.PatchSize), uint64(spec.PatchSize), media.RGBChannels, uint64(spec.Hidden),
	}
	required[visionPositionWeightTensor] = []uint64{uint64(spec.Hidden), uint64(positions)}
	addProjectorTensor(file, required, visionPatchBiasTensor, []uint64{uint64(spec.Hidden)}, bias)
}

func validateProjectorTensorCatalog(
	file *gguf.File,
	required map[string][]uint64,
	hostOnly ...string,
) ([]string, error) {
	names := slices.Sorted(maps.Keys(required))
	for _, name := range names {
		info, ok := file.Tensor(name)
		if !ok {
			return nil, fmt.Errorf("projector: missing tensor %q", name)
		}
		requirement := tensorcatalog.Requirement{Name: name, Shapes: [][]uint64{required[name]}}
		if err := tensorcatalog.ValidateInfo(info, requirement); err != nil {
			return nil, fmt.Errorf("projector: %w", err)
		}
	}
	return slices.DeleteFunc(names, func(name string) bool {
		return slices.Contains(hostOnly, name)
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

func addTwoLayerProjectionCatalog(
	file *gguf.File,
	required map[string][]uint64,
	input, hidden, output int,
	bias tensorPresence,
) {
	required[projectionFirstWeightTensor] = []uint64{uint64(input), uint64(hidden)}
	required[projectionSecondWeightTensor] = []uint64{uint64(hidden), uint64(output)}
	addProjectorTensor(file, required, projectionFirstBiasTensor, []uint64{uint64(hidden)}, bias)
	addProjectorTensor(file, required, projectionSecondBiasTensor, []uint64{uint64(output)}, bias)
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
		required[prefix+"attn_out.weight"] = []uint64{uint64(hidden), uint64(hidden)}
		required[prefix+"ffn_up.weight"] = []uint64{uint64(hidden), uint64(intermediate)}
		required[prefix+"ffn_down.weight"] = []uint64{uint64(intermediate), uint64(hidden)}
		required[prefix+"ln1.weight"] = hiddenShape
		required[prefix+"ln1.bias"] = hiddenShape
		required[prefix+"ln2.weight"] = hiddenShape
		required[prefix+"ln2.bias"] = hiddenShape
		if fused {
			required[prefix+"attn_qkv.weight"] = []uint64{uint64(hidden), uint64(3 * hidden)}
			addProjectorTensor(file, required, prefix+"attn_qkv.bias", []uint64{uint64(3 * hidden)}, bias)
		} else {
			for _, names := range splitAttentionTensors {
				required[prefix+names.weight] = []uint64{uint64(hidden), uint64(hidden)}
				addProjectorTensor(file, required, prefix+names.bias, hiddenShape, bias)
			}
		}
		addProjectorTensor(file, required, prefix+"attn_out.bias", hiddenShape, bias)
		addProjectorTensor(file, required, prefix+"ffn_up.bias", []uint64{uint64(intermediate)}, bias)
		addProjectorTensor(file, required, prefix+"ffn_down.bias", hiddenShape, bias)
	}
}
