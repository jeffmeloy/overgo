package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/reference"
)

const paddleOCRProjectorType = "paddleocr"

type paddleOCRActivation uint8

const (
	paddleOCRGELUQuick paddleOCRActivation = iota
	paddleOCRGELU
	paddleOCRSiLU
)

type PaddleOCRSpec struct {
	ImageSize             int
	PatchSize             int
	Hidden                int
	Intermediate          int
	ProjectorIntermediate int
	OutputHidden          int
	Layers                int
	Heads                 int
	MergeSize             int
	MinPixels             int
	MaxPixels             int
	LayerNormEpsilon      float32
	ImageMean             [3]float32
	ImageStd              [3]float32
	Activation            paddleOCRActivation
	PreLayerNorm          bool
	PostLayerNorm         bool
	FusedQKV              []bool
}

type PaddleOCRPreprocessOptions struct {
	MinPixels      int
	MaxPixels      int
	MaxAspectRatio int
}

type PaddleOCRImage struct {
	PixelValues []float32
	GridH       int
	GridW       int
}

type PaddleOCROutput struct {
	Embeddings reference.Value
	GridH      int
	GridW      int
	MergeSize  int
}

type PaddleOCRRunner struct {
	file *gguf.File
	spec PaddleOCRSpec
	cuda *paddleOCRCuda
}

type PaddleOCROpenOptions struct {
	CUDA          bool
	DeviceOrdinal int
}

func OpenPaddleOCR(path string) (*PaddleOCRRunner, error) {
	return OpenPaddleOCRWithOptions(path, PaddleOCROpenOptions{})
}

func OpenPaddleOCRWithOptions(path string, options PaddleOCROpenOptions) (*PaddleOCRRunner, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(cause error) (*PaddleOCRRunner, error) {
		_ = file.Close()
		return nil, cause
	}
	spec, err := ReadPaddleOCRSpec(file)
	if err != nil {
		return fail(err)
	}
	if err := validatePaddleOCRCatalog(file, spec); err != nil {
		return fail(err)
	}
	runner := &PaddleOCRRunner{file: file, spec: spec}
	if options.CUDA {
		runner.cuda, err = openPaddleOCRCuda(context.Background(), file, spec, options.DeviceOrdinal)
		if err != nil {
			return fail(fmt.Errorf("projector: initialize PaddleOCR CUDA: %w", err))
		}
	}
	return runner, nil
}

func (r *PaddleOCRRunner) Close() error {
	if r == nil {
		return nil
	}
	var closeErr error
	if r.cuda != nil {
		closeErr = r.cuda.Close()
		r.cuda = nil
	}
	if r.file == nil {
		return closeErr
	}
	file := r.file
	r.file = nil
	return errors.Join(closeErr, file.Close())
}

func (r *PaddleOCRRunner) Spec() PaddleOCRSpec {
	if r == nil {
		return PaddleOCRSpec{}
	}
	return r.spec
}

func ReadPaddleOCRSpec(file *gguf.File) (PaddleOCRSpec, error) {
	if file == nil {
		return PaddleOCRSpec{}, errors.New("projector: GGUF file is nil")
	}
	architecture, err := metadataString(file, "general.architecture")
	if err != nil {
		return PaddleOCRSpec{}, err
	}
	if architecture != "clip" {
		return PaddleOCRSpec{}, fmt.Errorf("projector: architecture %q is not clip", architecture)
	}
	projectorType, err := metadataString(file, "clip.projector_type")
	if err != nil {
		return PaddleOCRSpec{}, err
	}
	if projectorType != paddleOCRProjectorType {
		return PaddleOCRSpec{}, fmt.Errorf("projector: type %q is not %s", projectorType, paddleOCRProjectorType)
	}
	hasVision, err := metadataBool(file, "clip.has_vision_encoder")
	if err != nil {
		return PaddleOCRSpec{}, err
	}
	if !hasVision {
		return PaddleOCRSpec{}, errors.New("projector: vision encoder is disabled")
	}
	values := make([]int, 9)
	for index, key := range []string{
		"clip.vision.image_size", "clip.vision.patch_size", "clip.vision.embedding_length",
		"clip.vision.feed_forward_length", "clip.vision.projection_dim", "clip.vision.block_count",
		"clip.vision.attention.head_count", "clip.vision.image_min_pixels", "clip.vision.image_max_pixels",
	} {
		value, valueErr := metadataUint32(file, key)
		if valueErr != nil {
			return PaddleOCRSpec{}, valueErr
		}
		values[index] = int(value)
	}
	epsilon, err := metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return PaddleOCRSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", 3)
	if err != nil {
		return PaddleOCRSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", 3)
	if err != nil {
		return PaddleOCRSpec{}, err
	}
	activation, err := readPaddleOCRActivation(file)
	if err != nil {
		return PaddleOCRSpec{}, err
	}
	merger, ok := file.Tensor("mm.1.weight")
	if !ok || merger.Dimensions != 2 || merger.Shape[1] > uint64(^uint(0)>>1) {
		return PaddleOCRSpec{}, errors.New("projector: PaddleOCR merger tensor is unavailable or invalid")
	}
	spec := PaddleOCRSpec{
		ImageSize: values[0], PatchSize: values[1], Hidden: values[2], Intermediate: values[3],
		OutputHidden: values[4], Layers: values[5], Heads: values[6], MinPixels: values[7], MaxPixels: values[8],
		MergeSize: 2, ProjectorIntermediate: int(merger.Shape[1]), LayerNormEpsilon: epsilon, Activation: activation,
		PreLayerNorm: hasTensor(file, "v.pre_ln.weight"), PostLayerNorm: hasTensor(file, "v.post_ln.weight"),
		FusedQKV: make([]bool, values[5]),
	}
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	for layer := range spec.FusedQKV {
		spec.FusedQKV[layer] = hasTensor(file, fmt.Sprintf("v.blk.%d.attn_qkv.weight", layer))
	}
	if err := spec.validate(); err != nil {
		return PaddleOCRSpec{}, err
	}
	return spec, nil
}

func readPaddleOCRActivation(file *gguf.File) (paddleOCRActivation, error) {
	useGELU, err := optionalMetadataBool(file, "clip.use_gelu")
	if err != nil {
		return 0, err
	}
	useSiLU, err := optionalMetadataBool(file, "clip.use_silu")
	if err != nil {
		return 0, err
	}
	if useGELU && useSiLU {
		return 0, errors.New("projector: PaddleOCR GELU and SiLU are both enabled")
	}
	if useGELU {
		return paddleOCRGELU, nil
	}
	if useSiLU {
		return paddleOCRSiLU, nil
	}
	return paddleOCRGELUQuick, nil
}

func optionalMetadataBool(file *gguf.File, key string) (bool, error) {
	value, ok := file.MetadataValue(key)
	if !ok {
		return false, nil
	}
	if value.Type != gguf.ValueTypeBool {
		return false, fmt.Errorf("projector: metadata %q must be bool", key)
	}
	result, ok := value.Data.(bool)
	if !ok {
		return false, fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	return result, nil
}

func hasTensor(file *gguf.File, name string) bool {
	_, ok := file.Tensor(name)
	return ok
}

func (s PaddleOCRSpec) validate() error {
	if s.ImageSize <= 0 || s.PatchSize <= 0 || s.Hidden <= 0 || s.Intermediate <= 0 ||
		s.ProjectorIntermediate <= 0 || s.OutputHidden <= 0 || s.Layers <= 0 || s.Heads <= 0 ||
		s.MergeSize != 2 || s.MinPixels <= 0 || s.MaxPixels < s.MinPixels || s.Hidden%s.Heads != 0 ||
		(s.Hidden/s.Heads)%4 != 0 || s.ImageSize%s.PatchSize != 0 || s.LayerNormEpsilon <= 0 ||
		len(s.FusedQKV) != s.Layers {
		return fmt.Errorf("projector: invalid PaddleOCR metadata: %+v", s)
	}
	for channel := range s.ImageStd {
		if s.ImageStd[channel] <= 0 || !finite32(s.ImageMean[channel]) || !finite32(s.ImageStd[channel]) {
			return fmt.Errorf("projector: invalid PaddleOCR normalization channel %d", channel)
		}
	}
	return nil
}

func validatePaddleOCRCatalog(file *gguf.File, spec PaddleOCRSpec) error {
	required := map[string][]uint64{
		"v.patch_embd.weight":    {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.position_embd.weight": {uint64(spec.Hidden), uint64((spec.ImageSize / spec.PatchSize) * (spec.ImageSize / spec.PatchSize))},
		"mm.input_norm.weight":   {uint64(spec.Hidden)}, "mm.input_norm.bias": {uint64(spec.Hidden)},
		"mm.1.weight": {uint64(spec.Hidden * 4), uint64(spec.ProjectorIntermediate)},
		"mm.1.bias":   {uint64(spec.ProjectorIntermediate)},
		"mm.2.weight": {uint64(spec.ProjectorIntermediate), uint64(spec.OutputHidden)},
		"mm.2.bias":   {uint64(spec.OutputHidden)},
	}
	addOptionalPair := func(weight, bias string, shape []uint64) error {
		hasWeight, hasBias := hasTensor(file, weight), hasTensor(file, bias)
		if hasWeight != hasBias {
			return fmt.Errorf("projector: tensors %q and %q must be paired", weight, bias)
		}
		if hasWeight {
			required[weight], required[bias] = shape, shape
		}
		return nil
	}
	if err := addOptionalPair("v.pre_ln.weight", "v.pre_ln.bias", []uint64{uint64(spec.Hidden)}); err != nil {
		return err
	}
	if err := addOptionalPair("v.post_ln.weight", "v.post_ln.bias", []uint64{uint64(spec.Hidden)}); err != nil {
		return err
	}
	if hasTensor(file, "v.patch_embd.bias") {
		required["v.patch_embd.bias"] = []uint64{uint64(spec.Hidden)}
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

func DefaultPaddleOCRPreprocessOptions(spec PaddleOCRSpec) PaddleOCRPreprocessOptions {
	return PaddleOCRPreprocessOptions{MinPixels: spec.MinPixels, MaxPixels: spec.MaxPixels, MaxAspectRatio: 200}
}

func PreprocessPaddleOCRImage(source image.Image, spec PaddleOCRSpec, options PaddleOCRPreprocessOptions) (PaddleOCRImage, error) {
	if source == nil {
		return PaddleOCRImage{}, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return PaddleOCRImage{}, err
	}
	if options == (PaddleOCRPreprocessOptions{}) {
		options = DefaultPaddleOCRPreprocessOptions(spec)
	}
	bounds := source.Bounds()
	resizedH, resizedW, err := smartResizeAligned(
		bounds.Dy(), bounds.Dx(), spec.PatchSize*spec.MergeSize,
		options.MinPixels, options.MaxPixels, options.MaxAspectRatio,
	)
	if err != nil {
		return PaddleOCRImage{}, err
	}
	resized := resizeImageBilinear(source, resizedW, resizedH)
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
	return PaddleOCRImage{PixelValues: pixels, GridH: gridH, GridW: gridW}, nil
}

func resizeImageBilinear(source image.Image, width, height int) *image.RGBA {
	bounds := source.Bounds()
	inputW, inputH := bounds.Dx(), bounds.Dy()
	output := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		sourceY := (float64(y)+0.5)*float64(inputH)/float64(height) - 0.5
		y0 := max(0, min(inputH-1, int(math.Floor(sourceY))))
		y1 := min(y0+1, inputH-1)
		wy := sourceY - math.Floor(sourceY)
		if sourceY < 0 {
			wy = 0
		}
		for x := 0; x < width; x++ {
			sourceX := (float64(x)+0.5)*float64(inputW)/float64(width) - 0.5
			x0 := max(0, min(inputW-1, int(math.Floor(sourceX))))
			x1 := min(x0+1, inputW-1)
			wx := sourceX - math.Floor(sourceX)
			if sourceX < 0 {
				wx = 0
			}
			corners := [4][4]uint32{}
			corners[0][0], corners[0][1], corners[0][2], corners[0][3] = source.At(bounds.Min.X+x0, bounds.Min.Y+y0).RGBA()
			corners[1][0], corners[1][1], corners[1][2], corners[1][3] = source.At(bounds.Min.X+x1, bounds.Min.Y+y0).RGBA()
			corners[2][0], corners[2][1], corners[2][2], corners[2][3] = source.At(bounds.Min.X+x0, bounds.Min.Y+y1).RGBA()
			corners[3][0], corners[3][1], corners[3][2], corners[3][3] = source.At(bounds.Min.X+x1, bounds.Min.Y+y1).RGBA()
			weights := [4]float64{(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy}
			index := y*output.Stride + x*4
			for channel := 0; channel < 3; channel++ {
				value := 0.0
				for corner := range corners {
					value += weights[corner] * float64(corners[corner][channel]>>8)
				}
				output.Pix[index+channel] = clampUint8(value)
			}
			output.Pix[index+3] = 255
		}
	}
	return output
}

func (r *PaddleOCRRunner) EncodeImage(ctx context.Context, source image.Image, options PaddleOCRPreprocessOptions) (PaddleOCROutput, error) {
	if r == nil || r.file == nil {
		return PaddleOCROutput{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessPaddleOCRImage(source, r.spec, options)
	if err != nil {
		return PaddleOCROutput{}, err
	}
	return r.encode(ctx, input)
}

func (r *PaddleOCRRunner) encode(ctx context.Context, input PaddleOCRImage) (PaddleOCROutput, error) {
	return r.encodeGraph(ctx, input)
}

func paddleOCRGrid(height, width int) ([]int, []int) {
	rows := make([]int, height*width)
	columns := make([]int, height*width)
	for index := range rows {
		rows[index], columns[index] = index/width, index%width
	}
	return rows, columns
}
