package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

	"overgo/internal/gguf"
)

const (
	mimoVLProjectorType   = "mimovl"
	mimoVLPostNormEpsilon = 1e-6
)

type MiMoVLSpec struct {
	visionBackboneSpec
	ProjectionDim      int
	MergerIntermediate int
	KVHeads            int
	HeadDim            int
	MergeSize          int
	WindowSize         int
	MinPixels          int
	MaxPixels          int
	WindowModes        []int
}

type MiMoVLInput gridImage
type MiMoVLOutput gridOutput

type MiMoVLRunner struct {
	projectorResources
	spec MiMoVLSpec
}

func (r *MiMoVLRunner) Spec() MiMoVLSpec {
	if r == nil {
		return MiMoVLSpec{}
	}
	return r.spec
}

func ReadMiMoVLSpec(file *gguf.File) (MiMoVLSpec, error) {
	useSiLU, err := metadataBool(file, "clip.use_silu")
	if err != nil {
		return MiMoVLSpec{}, err
	}
	if !useSiLU {
		return MiMoVLSpec{}, errors.New("projector: MiMo-VL SiLU is disabled")
	}
	spec := MiMoVLSpec{}
	if err := readVisionBackbone(file, mimoVLProjectorType, &spec.ProjectionDim, &spec.visionBackboneSpec); err != nil {
		return MiMoVLSpec{}, err
	}
	if err := readMetadataIntFields(file,
		metadataIntField{"clip.vision.attention.head_count_kv", &spec.KVHeads},
		metadataIntField{"clip.vision.spatial_merge_size", &spec.MergeSize},
		metadataIntField{"clip.vision.window_size", &spec.WindowSize},
		metadataIntField{"clip.vision.image_min_pixels", &spec.MinPixels},
		metadataIntField{"clip.vision.image_max_pixels", &spec.MaxPixels},
	); err != nil {
		return MiMoVLSpec{}, err
	}
	modes, err := granite4MetadataInts(file, "clip.vision.wa_pattern_mode", true)
	if err != nil {
		return MiMoVLSpec{}, err
	}
	qkv, ok := file.Tensor("v.blk.0.attn_qkv.weight")
	if !ok || qkv.Dimensions != 2 {
		return MiMoVLSpec{}, errors.New("projector: MiMo-VL fused QKV tensor is unavailable")
	}
	denominator := spec.Heads + 2*spec.KVHeads
	if denominator <= 0 || qkv.Shape[1]%uint64(denominator) != 0 {
		return MiMoVLSpec{}, errors.New("projector: MiMo-VL fused QKV width is invalid")
	}
	merger, ok := file.Tensor("mm.0.weight")
	if !ok || merger.Dimensions != 2 {
		return MiMoVLSpec{}, errors.New("projector: MiMo-VL merger tensor is unavailable")
	}
	spec.HeadDim = int(qkv.Shape[1]) / denominator
	spec.MergerIntermediate = int(merger.Shape[1])
	spec.WindowModes = modes
	if err := spec.validate(); err != nil {
		return MiMoVLSpec{}, err
	}
	return spec, nil
}

func (s MiMoVLSpec) validate() error {
	if err := s.visionBackboneSpec.validate(); err != nil {
		return err
	}
	if s.ProjectionDim <= 0 || s.MergerIntermediate <= 0 || s.KVHeads <= 0 || s.HeadDim <= 0 ||
		s.MergeSize != 2 || s.WindowSize <= 0 || s.MinPixels <= 0 || s.MaxPixels < s.MinPixels ||
		s.Heads%s.KVHeads != 0 || s.HeadDim%4 != 0 || len(s.WindowModes) != s.Layers {
		return fmt.Errorf("projector: invalid MiMo-VL metadata: %+v", s)
	}
	for layer, mode := range s.WindowModes {
		if mode < -1 || mode > 1 {
			return fmt.Errorf("projector: MiMo-VL window mode %d at layer %d is invalid", mode, layer)
		}
	}
	return nil
}

func validateMiMoVLCatalog(file *gguf.File, spec MiMoVLSpec) ([]string, error) {
	qWidth := spec.Heads * spec.HeadDim
	kvWidth := spec.KVHeads * spec.HeadDim
	required := map[string][]uint64{
		"v.patch_embd.weight":   {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.patch_embd.weight.1": {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.post_ln.weight":      {uint64(spec.Hidden)},
		"mm.0.weight":           {uint64(spec.Hidden * 4), uint64(spec.MergerIntermediate)},
		"mm.2.weight":           {uint64(spec.MergerIntermediate), uint64(spec.ProjectionDim)},
	}
	for name, width := range map[string]int{
		"v.post_ln.bias": spec.Hidden,
		"mm.0.bias":      spec.MergerIntermediate,
		"mm.2.bias":      spec.ProjectionDim,
	} {
		addOptionalProjectorTensor(file, required, name, []uint64{uint64(width)})
	}
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
		if spec.WindowModes[layer] != -1 {
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

func PreprocessMiMoVLImage(source image.Image, spec MiMoVLSpec) (MiMoVLInput, error) {
	if source == nil {
		return MiMoVLInput{}, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return MiMoVLInput{}, err
	}
	preprocessSpec := Qwen3VLSpec{
		visionBackboneSpec: spec.visionBackboneSpec,
		MergerIntermediate: spec.MergerIntermediate, OutputHidden: spec.ProjectionDim, MergeSize: spec.MergeSize,
	}
	input, err := preprocessQwen3VLFrames([]image.Image{source, source}, preprocessSpec, Qwen3VLPreprocessOptions{
		MinPixels: spec.MinPixels, MaxPixels: spec.MaxPixels, MaxAspectRatio: defaultVisionMaxAspectRatio,
	})
	if err != nil {
		return MiMoVLInput{}, err
	}
	return MiMoVLInput{PixelValues: input.PixelValues, GridH: input.GridH, GridW: input.GridW}, nil
}

func (r *MiMoVLRunner) EncodeImage(ctx context.Context, source image.Image) (MiMoVLOutput, error) {
	if r == nil || r.file == nil {
		return MiMoVLOutput{}, errRunnerClosed
	}
	input, err := PreprocessMiMoVLImage(source, r.spec)
	if err != nil {
		return MiMoVLOutput{}, err
	}
	return r.encode(ctx, input)
}

func (r *MiMoVLRunner) encode(ctx context.Context, input MiMoVLInput) (MiMoVLOutput, error) {
	return r.encodeGraph(ctx, input)
}

func mimoVLColumnOrder(rows, columns, merge int) []int {
	order := make([]int, 0, rows*columns*merge*merge)
	for column := 0; column < columns; column++ {
		for row := 0; row < rows; row++ {
			unit := row*columns + column
			for patch := 0; patch < merge*merge; patch++ {
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

func reorderRows(input []float32, order []int, width int) []float32 {
	output := make([]float32, len(input))
	for destination, source := range order {
		copy(output[destination*width:(destination+1)*width], input[source*width:(source+1)*width])
	}
	return output
}

func reorderInts(input []int, order []int) []int {
	output := make([]int, len(input))
	for destination, source := range order {
		output[destination] = input[source]
	}
	return output
}
