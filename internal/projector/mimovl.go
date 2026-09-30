package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/tensor"
)

const mimoVLProjectorType = "mimovl"

type spatialWindowMode int8

const (
	windowModeGlobal spatialWindowMode = iota - tensor.SingletonExtent
	windowModeRow
	windowModeColumn
)

type miMoVLSpec struct {
	visionBackboneSpec
	ProjectionDim      int
	MergerIntermediate int
	KVHeads            int
	HeadDim            int
	MergeSize          int
	WindowSize         int
	MinPixels          int
	MaxPixels          int
	WindowModes        []spatialWindowMode
}

type miMoVLInput gridImage
type miMoVLOutput gridOutput

type miMoVLRunner struct {
	projectorResources
	spec miMoVLSpec
}

func (r *miMoVLRunner) Spec() miMoVLSpec {
	if r == nil {
		return miMoVLSpec{}
	}
	return r.spec
}

func readMiMoVLSpec(file *gguf.File) (miMoVLSpec, error) {
	useSiLU, err := metadataBool(file, visionUseSiLUKey)
	if err != nil {
		return miMoVLSpec{}, err
	}
	if !useSiLU {
		return miMoVLSpec{}, errors.New("projector: MiMo-VL SiLU is disabled")
	}
	spec := miMoVLSpec{}
	if err := readRotaryVisionBackbone(file, mimoVLProjectorType, &spec.ProjectionDim, &spec.visionBackboneSpec); err != nil {
		return miMoVLSpec{}, err
	}
	if err := readProjectionNorm(file, &spec.ProjectionNormEpsilon); err != nil {
		return miMoVLSpec{}, err
	}
	if err := readMetadataIntFields(file,
		metadataIntField{visionKVHeadCountKey, &spec.KVHeads},
		metadataIntField{visionSpatialMergeKey, &spec.MergeSize},
		metadataIntField{visionWindowSizeKey, &spec.WindowSize},
		metadataIntField{visionMinPixelsKey, &spec.MinPixels},
		metadataIntField{visionMaxPixelsKey, &spec.MaxPixels},
	); err != nil {
		return miMoVLSpec{}, err
	}
	modeValues, err := metadataInts(file, "clip.vision.wa_pattern_mode", tensorRequired, false)
	if err != nil {
		return miMoVLSpec{}, err
	}
	modes := make([]spatialWindowMode, len(modeValues))
	for index, value := range modeValues {
		mode := spatialWindowMode(value)
		switch mode {
		case windowModeGlobal, windowModeRow, windowModeColumn:
		default:
			return miMoVLSpec{}, fmt.Errorf("projector: MiMo-VL window mode %d at layer %d is invalid", value, index)
		}
		modes[index] = mode
	}
	qkv, ok := file.Tensor("v.blk.0.attn_qkv.weight")
	if !ok || qkv.Dimensions != tensor.PairedExtent {
		return miMoVLSpec{}, errors.New("projector: MiMo-VL fused QKV tensor is unavailable")
	}
	if !checked.PositiveInts(spec.Heads, spec.KVHeads) {
		return miMoVLSpec{}, errors.New("projector: MiMo-VL fused QKV width is invalid")
	}
	kvWidth, ok := checked.Mul64(uint64(spec.KVHeads), tensor.PairedExtent)
	denominator, ok := checked.Add64(uint64(spec.Heads), kvWidth)
	headDim, ok := checked.DivExact64(qkv.Shape[tensor.SingletonExtent], denominator)
	spec.HeadDim, ok = checked.Int(headDim)
	if !ok {
		return miMoVLSpec{}, errors.New("projector: MiMo-VL fused QKV width is invalid")
	}
	merger, ok := file.Tensor(projectionFirstWeightTensor)
	spec.MergerIntermediate, ok = matrixRowsInt(merger, ok)
	if !ok {
		return miMoVLSpec{}, errors.New("projector: MiMo-VL merger tensor is unavailable")
	}
	spec.WindowModes = modes
	if err := spec.validate(); err != nil {
		return miMoVLSpec{}, err
	}
	return spec, nil
}

func (s miMoVLSpec) validate() error {
	if err := s.visionBackboneSpec.validateRotary(); err != nil {
		return err
	}
	_, groupsOK := checked.DivExactInt(s.Heads, s.KVHeads)
	_, rotaryOK := checked.DivExactInt(s.HeadDim, tensor.PairedExtent*tensor.PairedExtent)
	if !checked.PositiveInts(s.ProjectionDim, s.MergerIntermediate, s.KVHeads, s.HeadDim, s.MergeSize, s.WindowSize, s.MinPixels) ||
		s.MaxPixels < s.MinPixels || !checked.PositiveFinite32(s.ProjectionNormEpsilon) ||
		!groupsOK || !rotaryOK || len(s.WindowModes) != s.Layers {
		return fmt.Errorf("projector: invalid MiMo-VL metadata: %+v", s)
	}
	return nil
}

func validateMiMoVLCatalog(file *gguf.File, spec miMoVLSpec) ([]string, error) {
	qWidth := spec.Heads * spec.HeadDim
	kvWidth := spec.KVHeads * spec.HeadDim
	required := map[string][]uint64{
		visionPatchWeightTensor:    {uint64(spec.PatchSize), uint64(spec.PatchSize), media.RGBChannels, uint64(spec.Hidden)},
		visionPatchWeightTensor1:   {uint64(spec.PatchSize), uint64(spec.PatchSize), media.RGBChannels, uint64(spec.Hidden)},
		visionPostNormWeightTensor: {uint64(spec.Hidden)},
	}
	addTwoLayerProjectionCatalog(file, required, spec.Hidden*spec.MergeSize*spec.MergeSize, spec.MergerIntermediate, spec.ProjectionDim, tensorOptional)
	addOptionalProjectorTensor(file, required, visionPostNormBiasTensor, []uint64{uint64(spec.Hidden)})
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for name, shape := range map[string][]uint64{
			"attn_qkv.weight": {uint64(spec.Hidden), uint64(qWidth + 2*kvWidth)},
			"attn_qkv.bias":   {uint64(qWidth + 2*kvWidth)},
			"attn_out.weight": {uint64(qWidth), uint64(spec.Hidden)},
			"ffn_up.weight":   {uint64(spec.Hidden), uint64(spec.Intermediate)},
			"ffn_up.bias":     {uint64(spec.Intermediate)},
			"ffn_gate.weight": {uint64(spec.Hidden), uint64(spec.Intermediate)},
			"ffn_gate.bias":   {uint64(spec.Intermediate)},
			"ffn_down.weight": {uint64(spec.Intermediate), uint64(spec.Hidden)},
			"ffn_down.bias":   {uint64(spec.Hidden)},
			"ln1.weight":      {uint64(spec.Hidden)},
			"ln2.weight":      {uint64(spec.Hidden)},
		} {
			required[prefix+name] = shape
		}
		if spec.WindowModes[layer] != windowModeGlobal {
			required[prefix+"attn_sinks"] = []uint64{uint64(spec.Heads)}
		}
		for name, width := range map[string]int{
			"attn_out.bias": spec.Hidden, "ln1.bias": spec.Hidden, "ln2.bias": spec.Hidden,
		} {
			addOptionalProjectorTensor(file, required, prefix+name, []uint64{uint64(width)})
		}
	}
	return validateProjectorTensorCatalog(file, required)
}

func preprocessMiMoVLImage(source image.Image, spec miMoVLSpec) (miMoVLInput, error) {
	if err := spec.validate(); err != nil {
		return miMoVLInput{}, err
	}
	preprocessSpec := qwen3VLSpec{
		visionBackboneSpec: spec.visionBackboneSpec,
		MergerIntermediate: spec.MergerIntermediate, OutputHidden: spec.ProjectionDim, MergeSize: spec.MergeSize,
	}
	input, err := preprocessQwen3VLFrames([]image.Image{source, source}, preprocessSpec, qwen3VLPreprocessOptions{
		MinPixels: spec.MinPixels, MaxPixels: spec.MaxPixels,
	})
	if err != nil {
		return miMoVLInput{}, err
	}
	return miMoVLInput{PixelValues: input.PixelValues, GridH: input.GridH, GridW: input.GridW}, nil
}

func (r *miMoVLRunner) EncodeImage(ctx context.Context, source image.Image) (miMoVLOutput, error) {
	if r == nil || r.file == nil {
		return miMoVLOutput{}, errRunnerClosed
	}
	input, err := preprocessMiMoVLImage(source, r.spec)
	if err != nil {
		return miMoVLOutput{}, err
	}
	return r.encodeGraph(ctx, input)
}

func columnMajorPatchOrder(rows, columns, merge int) []int {
	order := make([]int, tensor.FirstOffset, rows*columns*merge*merge)
	for column := range columns {
		for row := range rows {
			unit := row*columns + column
			for patch := range merge * merge {
				order = append(order, unit*merge*merge+patch)
			}
		}
	}
	return order
}

func inversePermutation(order []int) []int {
	inverse := make([]int, len(order))
	for destination, source := range order {
		inverse[source] = destination
	}
	return inverse
}

func reorderInts(input []int, order []int) []int {
	output := make([]int, len(input))
	for destination, source := range order {
		output[destination] = input[source]
	}
	return output
}
