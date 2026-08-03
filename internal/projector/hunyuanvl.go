package projector

import (
	"context"
	"errors"
	"fmt"
	"image"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/reference"
)

const hunyuanVLProjectorType = "hunyuanvl"

type HunyuanVLSpec struct {
	ImageSize        int
	PatchSize        int
	Hidden           int
	Intermediate     int
	OutputHidden     int
	Layers           int
	Heads            int
	MergeSize        int
	MinPixels        int
	MaxPixels        int
	ConvIntermediate int
	ProjectorInput   int
	LayerNormEpsilon float32
	ImageMean        [3]float32
	ImageStd         [3]float32
	PreLayerNorm     bool
	PostLayerNorm    bool
	FusedQKV         []bool
}

type HunyuanVLPreprocessOptions struct {
	MinPixels      int
	MaxPixels      int
	MaxAspectRatio int
}

type HunyuanVLImage struct {
	PixelValues []float32
	GridH       int
	GridW       int
}

type HunyuanVLOutput struct {
	Embeddings reference.Value
	GridH      int
	GridW      int
	MergeSize  int
}

type HunyuanVLRunner struct {
	file *gguf.File
	spec HunyuanVLSpec
	cuda *hunyuanVLCUDA
}

type HunyuanVLOpenOptions struct {
	CUDA          bool
	DeviceOrdinal int
}

func OpenHunyuanVL(path string) (*HunyuanVLRunner, error) {
	return OpenHunyuanVLWithOptions(path, HunyuanVLOpenOptions{})
}

func OpenHunyuanVLWithOptions(path string, options HunyuanVLOpenOptions) (*HunyuanVLRunner, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(cause error) (*HunyuanVLRunner, error) {
		_ = file.Close()
		return nil, cause
	}
	spec, err := ReadHunyuanVLSpec(file)
	if err != nil {
		return fail(err)
	}
	if err := validateHunyuanVLCatalog(file, spec); err != nil {
		return fail(err)
	}
	runner := &HunyuanVLRunner{file: file, spec: spec}
	if options.CUDA {
		runner.cuda, err = openHunyuanVLCUDA(context.Background(), file, spec, options.DeviceOrdinal)
		if err != nil {
			return fail(fmt.Errorf("projector: initialize Hunyuan-VL CUDA: %w", err))
		}
	}
	return runner, nil
}

func (r *HunyuanVLRunner) Close() error {
	if r == nil {
		return nil
	}
	return closeProjectorResources(&r.file, &r.cuda)
}

func (r *HunyuanVLRunner) Spec() HunyuanVLSpec {
	if r == nil {
		return HunyuanVLSpec{}
	}
	return r.spec
}

func ReadHunyuanVLSpec(file *gguf.File) (HunyuanVLSpec, error) {
	if file == nil {
		return HunyuanVLSpec{}, errors.New("projector: GGUF file is nil")
	}
	architecture, err := metadataString(file, "general.architecture")
	if err != nil {
		return HunyuanVLSpec{}, err
	}
	if architecture != "clip" {
		return HunyuanVLSpec{}, fmt.Errorf("projector: architecture %q is not clip", architecture)
	}
	projectorType, err := metadataString(file, "clip.projector_type")
	if err != nil {
		return HunyuanVLSpec{}, err
	}
	if projectorType != hunyuanVLProjectorType {
		return HunyuanVLSpec{}, fmt.Errorf("projector: type %q is not %s", projectorType, hunyuanVLProjectorType)
	}
	hasVision, err := metadataBool(file, "clip.has_vision_encoder")
	if err != nil {
		return HunyuanVLSpec{}, err
	}
	if !hasVision {
		return HunyuanVLSpec{}, errors.New("projector: vision encoder is disabled")
	}
	values := make([]int, 7)
	for index, key := range []string{
		"clip.vision.image_size", "clip.vision.patch_size", "clip.vision.embedding_length",
		"clip.vision.feed_forward_length", "clip.vision.projection_dim", "clip.vision.block_count",
		"clip.vision.attention.head_count",
	} {
		value, valueErr := metadataUint32(file, key)
		if valueErr != nil {
			return HunyuanVLSpec{}, valueErr
		}
		values[index] = int(value)
	}
	merge := 2
	if value, ok, valueErr := optionalMetadataUint32(file, "clip.vision.spatial_merge_size"); valueErr != nil {
		return HunyuanVLSpec{}, valueErr
	} else if ok {
		merge = int(value)
	}
	minPixels, _, err := optionalMetadataUint32(file, "clip.vision.image_min_pixels")
	if err != nil {
		return HunyuanVLSpec{}, err
	}
	maxPixels, _, err := optionalMetadataUint32(file, "clip.vision.image_max_pixels")
	if err != nil {
		return HunyuanVLSpec{}, err
	}
	factor := values[1] * merge
	if minPixels == 0 {
		minPixels = uint32(factor * factor * 256)
	}
	if maxPixels == 0 {
		maxPixels = uint32(factor * factor * 16384)
	}
	epsilon, err := metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return HunyuanVLSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", 3)
	if err != nil {
		return HunyuanVLSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", 3)
	if err != nil {
		return HunyuanVLSpec{}, err
	}
	conv0, ok := file.Tensor("mm.0.weight")
	if !ok || conv0.Dimensions != 4 {
		return HunyuanVLSpec{}, errors.New("projector: Hunyuan-VL first convolution is unavailable or invalid")
	}
	conv2, ok := file.Tensor("mm.2.weight")
	if !ok || conv2.Dimensions != 4 {
		return HunyuanVLSpec{}, errors.New("projector: Hunyuan-VL second convolution is unavailable or invalid")
	}
	spec := HunyuanVLSpec{
		ImageSize: values[0], PatchSize: values[1], Hidden: values[2], Intermediate: values[3],
		OutputHidden: values[4], Layers: values[5], Heads: values[6], MergeSize: merge,
		MinPixels: int(minPixels), MaxPixels: int(maxPixels), ConvIntermediate: int(conv0.Shape[3]),
		ProjectorInput: int(conv2.Shape[3]), LayerNormEpsilon: epsilon,
		PreLayerNorm: hasTensor(file, "v.pre_ln.weight"), PostLayerNorm: hasTensor(file, "v.post_ln.weight"),
		FusedQKV: make([]bool, values[5]),
	}
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	for layer := range spec.FusedQKV {
		spec.FusedQKV[layer] = hasTensor(file, fmt.Sprintf("v.blk.%d.attn_qkv.weight", layer))
	}
	if err := spec.validate(); err != nil {
		return HunyuanVLSpec{}, err
	}
	return spec, nil
}

func optionalMetadataUint32(file *gguf.File, key string) (uint32, bool, error) {
	value, ok := file.MetadataValue(key)
	if !ok {
		return 0, false, nil
	}
	if value.Type != gguf.ValueTypeUint32 {
		return 0, false, fmt.Errorf("projector: metadata %q must be uint32", key)
	}
	result, ok := value.Data.(uint32)
	if !ok {
		return 0, false, fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	return result, true, nil
}

func (s HunyuanVLSpec) validate() error {
	if s.ImageSize <= 0 || s.PatchSize <= 0 || s.Hidden <= 0 || s.Intermediate <= 0 ||
		s.OutputHidden <= 0 || s.Layers <= 0 || s.Heads <= 0 || s.MergeSize <= 0 ||
		s.MinPixels <= 0 || s.MaxPixels < s.MinPixels || s.ConvIntermediate <= 0 || s.ProjectorInput <= 0 ||
		s.Hidden%s.Heads != 0 || s.ImageSize%s.PatchSize != 0 || s.LayerNormEpsilon <= 0 ||
		len(s.FusedQKV) != s.Layers {
		return fmt.Errorf("projector: invalid Hunyuan-VL metadata: %+v", s)
	}
	for channel := range s.ImageStd {
		if s.ImageStd[channel] <= 0 || !finite32(s.ImageMean[channel]) || !finite32(s.ImageStd[channel]) {
			return fmt.Errorf("projector: invalid Hunyuan-VL normalization channel %d", channel)
		}
	}
	return nil
}

func validateHunyuanVLCatalog(file *gguf.File, spec HunyuanVLSpec) error {
	positionSide := spec.ImageSize / spec.PatchSize
	required := map[string][]uint64{
		"v.patch_embd.weight":    {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.position_embd.weight": {uint64(spec.Hidden), uint64(positionSide * positionSide)},
		"mm.pre_norm.weight":     {uint64(spec.Hidden)},
		"mm.0.weight":            {uint64(spec.MergeSize), uint64(spec.MergeSize), uint64(spec.Hidden), uint64(spec.ConvIntermediate)},
		"mm.0.bias":              {uint64(spec.ConvIntermediate)},
		"mm.2.weight":            {1, 1, uint64(spec.ConvIntermediate), uint64(spec.ProjectorInput)},
		"mm.2.bias":              {uint64(spec.ProjectorInput)},
		"v.image_newline":        {uint64(spec.ProjectorInput)},
		"mm.model.fc.weight":     {uint64(spec.ProjectorInput), uint64(spec.OutputHidden)},
		"mm.model.fc.bias":       {uint64(spec.OutputHidden)},
		"mm.image_begin":         {uint64(spec.OutputHidden)}, "mm.image_end": {uint64(spec.OutputHidden)},
		"mm.post_norm.weight": {uint64(spec.OutputHidden)},
	}
	if hasTensor(file, "v.patch_embd.bias") {
		required["v.patch_embd.bias"] = []uint64{uint64(spec.Hidden)}
	}
	for _, prefix := range []string{"v.pre_ln", "v.post_ln"} {
		hasWeight, hasBias := hasTensor(file, prefix+".weight"), hasTensor(file, prefix+".bias")
		if hasWeight != hasBias {
			return fmt.Errorf("projector: tensors %q and %q must be paired", prefix+".weight", prefix+".bias")
		}
		if hasWeight {
			required[prefix+".weight"], required[prefix+".bias"] = []uint64{uint64(spec.Hidden)}, []uint64{uint64(spec.Hidden)}
		}
	}
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for name, shape := range map[string][]uint64{
			"attn_out.weight": {uint64(spec.Hidden), uint64(spec.Hidden)},
			"ffn_up.weight":   {uint64(spec.Hidden), uint64(spec.Intermediate)},
			"ffn_down.weight": {uint64(spec.Intermediate), uint64(spec.Hidden)},
			"ln1.weight":      {uint64(spec.Hidden)}, "ln1.bias": {uint64(spec.Hidden)},
			"ln2.weight": {uint64(spec.Hidden)}, "ln2.bias": {uint64(spec.Hidden)},
		} {
			required[prefix+name] = shape
		}
		if spec.FusedQKV[layer] {
			required[prefix+"attn_qkv.weight"] = []uint64{uint64(spec.Hidden), uint64(3 * spec.Hidden)}
			if hasTensor(file, prefix+"attn_qkv.bias") {
				required[prefix+"attn_qkv.bias"] = []uint64{uint64(3 * spec.Hidden)}
			}
		} else {
			for _, part := range []string{"q", "k", "v"} {
				required[prefix+"attn_"+part+".weight"] = []uint64{uint64(spec.Hidden), uint64(spec.Hidden)}
				if hasTensor(file, prefix+"attn_"+part+".bias") {
					required[prefix+"attn_"+part+".bias"] = []uint64{uint64(spec.Hidden)}
				}
			}
		}
		for _, name := range []string{"attn_out", "ffn_up", "ffn_down"} {
			if hasTensor(file, prefix+name+".bias") {
				width := spec.Hidden
				if name == "ffn_up" {
					width = spec.Intermediate
				}
				required[prefix+name+".bias"] = []uint64{uint64(width)}
			}
		}
	}
	return validateProjectorTensorShapes(file, required)
}

func DefaultHunyuanVLPreprocessOptions(spec HunyuanVLSpec) HunyuanVLPreprocessOptions {
	return HunyuanVLPreprocessOptions{MinPixels: spec.MinPixels, MaxPixels: spec.MaxPixels, MaxAspectRatio: 200}
}

func PreprocessHunyuanVLImage(source image.Image, spec HunyuanVLSpec, options HunyuanVLPreprocessOptions) (HunyuanVLImage, error) {
	if source == nil {
		return HunyuanVLImage{}, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return HunyuanVLImage{}, err
	}
	if options == (HunyuanVLPreprocessOptions{}) {
		options = DefaultHunyuanVLPreprocessOptions(spec)
	}
	bounds := source.Bounds()
	resizedH, resizedW, err := smartResizeAligned(
		bounds.Dy(), bounds.Dx(), spec.PatchSize*spec.MergeSize,
		options.MinPixels, options.MaxPixels, options.MaxAspectRatio,
	)
	if err != nil {
		return HunyuanVLImage{}, err
	}
	resized := resizeImageBicubic(source, resizedW, resizedH)
	gridH, gridW := resizedH/spec.PatchSize, resizedW/spec.PatchSize
	patchArea := spec.PatchSize * spec.PatchSize
	patchWidth := 3 * patchArea
	pixels := make([]float32, gridH*gridW*patchWidth)
	for patchY := 0; patchY < gridH; patchY++ {
		for patchX := 0; patchX < gridW; patchX++ {
			row := (patchY*gridW + patchX) * patchWidth
			for channel := 0; channel < 3; channel++ {
				position := row + channel*patchArea
				for y := 0; y < spec.PatchSize; y++ {
					for x := 0; x < spec.PatchSize; x++ {
						r, g, b, _ := resized.At(patchX*spec.PatchSize+x, patchY*spec.PatchSize+y).RGBA()
						value := [3]uint32{r, g, b}[channel]
						pixels[position] = (float32(value>>8)/255 - spec.ImageMean[channel]) / spec.ImageStd[channel]
						position++
					}
				}
			}
		}
	}
	return HunyuanVLImage{PixelValues: pixels, GridH: gridH, GridW: gridW}, nil
}

func (r *HunyuanVLRunner) EncodeImage(ctx context.Context, source image.Image, options HunyuanVLPreprocessOptions) (HunyuanVLOutput, error) {
	if r == nil || r.file == nil {
		return HunyuanVLOutput{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessHunyuanVLImage(source, r.spec, options)
	if err != nil {
		return HunyuanVLOutput{}, err
	}
	return r.encode(ctx, input)
}

func (r *HunyuanVLRunner) encode(ctx context.Context, input HunyuanVLImage) (HunyuanVLOutput, error) {
	return r.encodeGraph(ctx, input)
}
