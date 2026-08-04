package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/reference"
)

const (
	mimoVLProjectorType   = "mimovl"
	mimoVLPostNormEpsilon = 1e-6
)

type MiMoVLSpec struct {
	ImageSize          int
	PatchSize          int
	Hidden             int
	Intermediate       int
	ProjectionDim      int
	MergerIntermediate int
	Layers             int
	Heads              int
	KVHeads            int
	HeadDim            int
	MergeSize          int
	WindowSize         int
	MinPixels          int
	MaxPixels          int
	LayerNormEpsilon   float32
	ImageMean          [3]float32
	ImageStd           [3]float32
	WindowModes        []int
}

type MiMoVLInput struct {
	PixelValues []float32
	GridH       int
	GridW       int
}

type MiMoVLOutput struct {
	Embeddings reference.Value
	GridH      int
	GridW      int
	MergeSize  int
}

type MiMoVLRunner struct {
	file *gguf.File
	spec MiMoVLSpec
	cuda *projectorCUDA
}

type MiMoVLOpenOptions struct {
	CUDA          bool
	DeviceOrdinal int
}

func OpenMiMoVL(path string) (*MiMoVLRunner, error) {
	return OpenMiMoVLWithOptions(path, MiMoVLOpenOptions{})
}

func OpenMiMoVLWithOptions(path string, options MiMoVLOpenOptions) (*MiMoVLRunner, error) {
	return openProjectorResource(path, func(file *gguf.File) (*MiMoVLRunner, error) {
		spec, err := ReadMiMoVLSpec(file)
		if err != nil {
			return nil, err
		}
		catalog, err := validateMiMoVLCatalog(file, spec)
		if err != nil {
			return nil, err
		}
		runner := &MiMoVLRunner{file: file, spec: spec}
		if options.CUDA {
			runner.cuda, err = openProjectorCUDA(context.Background(), file, catalog, nil, options.DeviceOrdinal)
			if err != nil {
				return nil, fmt.Errorf("projector: initialize MiMo-VL CUDA: %w", err)
			}
		}
		return runner, nil
	})
}

func (r *MiMoVLRunner) Close() error {
	if r == nil {
		return nil
	}
	return closeProjectorResources(&r.file, &r.cuda)
}

func (r *MiMoVLRunner) Spec() MiMoVLSpec {
	if r == nil {
		return MiMoVLSpec{}
	}
	return r.spec
}

func ReadMiMoVLSpec(file *gguf.File) (MiMoVLSpec, error) {
	if file == nil {
		return MiMoVLSpec{}, errors.New("projector: GGUF file is nil")
	}
	architecture, err := metadataString(file, "general.architecture")
	if err != nil {
		return MiMoVLSpec{}, err
	}
	if architecture != "clip" {
		return MiMoVLSpec{}, fmt.Errorf("projector: architecture %q is not clip", architecture)
	}
	projectorType, err := metadataString(file, "clip.projector_type")
	if err != nil {
		return MiMoVLSpec{}, err
	}
	if projectorType != mimoVLProjectorType {
		return MiMoVLSpec{}, fmt.Errorf("projector: type %q is not %s", projectorType, mimoVLProjectorType)
	}
	hasVision, err := metadataBool(file, "clip.has_vision_encoder")
	if err != nil {
		return MiMoVLSpec{}, err
	}
	if !hasVision {
		return MiMoVLSpec{}, errors.New("projector: vision encoder is disabled")
	}
	useSiLU, err := metadataBool(file, "clip.use_silu")
	if err != nil {
		return MiMoVLSpec{}, err
	}
	if !useSiLU {
		return MiMoVLSpec{}, errors.New("projector: MiMo-VL SiLU is disabled")
	}
	keys := []string{
		"clip.vision.image_size", "clip.vision.patch_size", "clip.vision.embedding_length",
		"clip.vision.feed_forward_length", "clip.vision.projection_dim", "clip.vision.block_count",
		"clip.vision.attention.head_count", "clip.vision.attention.head_count_kv",
		"clip.vision.spatial_merge_size", "clip.vision.window_size",
		"clip.vision.image_min_pixels", "clip.vision.image_max_pixels",
	}
	values := make([]int, len(keys))
	for index, key := range keys {
		value, valueErr := metadataUint32(file, key)
		if valueErr != nil {
			return MiMoVLSpec{}, valueErr
		}
		values[index] = int(value)
	}
	epsilon, err := metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return MiMoVLSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", 3)
	if err != nil {
		return MiMoVLSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", 3)
	if err != nil {
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
	denominator := values[6] + 2*values[7]
	if denominator <= 0 || qkv.Shape[1]%uint64(denominator) != 0 {
		return MiMoVLSpec{}, errors.New("projector: MiMo-VL fused QKV width is invalid")
	}
	merger, ok := file.Tensor("mm.0.weight")
	if !ok || merger.Dimensions != 2 {
		return MiMoVLSpec{}, errors.New("projector: MiMo-VL merger tensor is unavailable")
	}
	spec := MiMoVLSpec{
		ImageSize: values[0], PatchSize: values[1], Hidden: values[2], Intermediate: values[3],
		ProjectionDim: values[4], Layers: values[5], Heads: values[6], KVHeads: values[7],
		MergeSize: values[8], WindowSize: values[9], MinPixels: values[10], MaxPixels: values[11],
		HeadDim: int(qkv.Shape[1]) / denominator, MergerIntermediate: int(merger.Shape[1]),
		LayerNormEpsilon: epsilon, WindowModes: modes,
	}
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	if err := spec.validate(); err != nil {
		return MiMoVLSpec{}, err
	}
	return spec, nil
}

func (s MiMoVLSpec) validate() error {
	if s.ImageSize <= 0 || s.PatchSize <= 0 || s.Hidden <= 0 || s.Intermediate <= 0 || s.ProjectionDim <= 0 ||
		s.MergerIntermediate <= 0 || s.Layers <= 0 || s.Heads <= 0 || s.KVHeads <= 0 || s.HeadDim <= 0 ||
		s.MergeSize != 2 || s.WindowSize <= 0 || s.MinPixels <= 0 || s.MaxPixels < s.MinPixels ||
		s.Heads%s.KVHeads != 0 || s.HeadDim%4 != 0 || s.ImageSize%s.PatchSize != 0 ||
		len(s.WindowModes) != s.Layers || s.LayerNormEpsilon <= 0 {
		return fmt.Errorf("projector: invalid MiMo-VL metadata: %+v", s)
	}
	for layer, mode := range s.WindowModes {
		if mode < -1 || mode > 1 {
			return fmt.Errorf("projector: MiMo-VL window mode %d at layer %d is invalid", mode, layer)
		}
	}
	for channel := range s.ImageStd {
		if s.ImageStd[channel] <= 0 || !finite32(s.ImageMean[channel]) || !finite32(s.ImageStd[channel]) {
			return fmt.Errorf("projector: invalid MiMo-VL normalization channel %d", channel)
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
		ImageSize: spec.ImageSize, PatchSize: spec.PatchSize, Hidden: spec.Hidden,
		Intermediate: spec.Intermediate, MergerIntermediate: spec.MergerIntermediate,
		OutputHidden: spec.ProjectionDim, Layers: spec.Layers, Heads: spec.Heads,
		MergeSize: spec.MergeSize, LayerNormEpsilon: spec.LayerNormEpsilon,
		ImageMean: spec.ImageMean, ImageStd: spec.ImageStd,
	}
	input, err := preprocessQwen3VLFrames([]image.Image{source, source}, preprocessSpec, Qwen3VLPreprocessOptions{
		MinPixels: spec.MinPixels, MaxPixels: spec.MaxPixels, MaxAspectRatio: 200,
	})
	if err != nil {
		return MiMoVLInput{}, err
	}
	return MiMoVLInput{PixelValues: input.PixelValues, GridH: input.GridH, GridW: input.GridW}, nil
}

func (r *MiMoVLRunner) EncodeImage(ctx context.Context, source image.Image) (MiMoVLOutput, error) {
	if r == nil || r.file == nil {
		return MiMoVLOutput{}, errors.New("projector: runner is closed")
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
