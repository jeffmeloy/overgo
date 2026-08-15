package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"overgo/internal/gguf"
)

const (
	paddleOCRProjectorType    = "paddleocr"
	paddleOCRInputNormEpsilon = 1e-5
)

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

type PaddleOCROutput gridOutput

type PaddleOCRRunner struct {
	projectorResources
	spec      PaddleOCRSpec
	attention visionAttentionPlan
}

func openPaddleOCR(ctx context.Context, file *gguf.File, options OpenOptions) (*PaddleOCRRunner, error) {
	return buildCatalogProjector(ctx, file, options, "PaddleOCR", nil,
		ReadPaddleOCRSpec, validatePaddleOCRCatalog,
		func(file *gguf.File, spec PaddleOCRSpec, cuda *projectorCUDA) *PaddleOCRRunner {
			return &PaddleOCRRunner{projectorResources: projectorResources{file: file, cuda: cuda}, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads)}
		})
}

func (r *PaddleOCRRunner) Spec() PaddleOCRSpec {
	if r == nil {
		return PaddleOCRSpec{}
	}
	return r.spec
}

func ReadPaddleOCRSpec(file *gguf.File) (PaddleOCRSpec, error) {
	if err := validateVisionProjector(file, "clip.projector_type", paddleOCRProjectorType); err != nil {
		return PaddleOCRSpec{}, err
	}
	spec := PaddleOCRSpec{}
	if err := readMetadataIntFields(file,
		metadataIntField{"clip.vision.image_size", &spec.ImageSize},
		metadataIntField{"clip.vision.patch_size", &spec.PatchSize},
		metadataIntField{"clip.vision.embedding_length", &spec.Hidden},
		metadataIntField{"clip.vision.feed_forward_length", &spec.Intermediate},
		metadataIntField{"clip.vision.projection_dim", &spec.OutputHidden},
		metadataIntField{"clip.vision.block_count", &spec.Layers},
		metadataIntField{"clip.vision.attention.head_count", &spec.Heads},
		metadataIntField{"clip.vision.image_min_pixels", &spec.MinPixels},
		metadataIntField{"clip.vision.image_max_pixels", &spec.MaxPixels},
	); err != nil {
		return PaddleOCRSpec{}, err
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
	spec.MergeSize = 2
	spec.ProjectorIntermediate = int(merger.Shape[1])
	spec.LayerNormEpsilon = epsilon
	spec.Activation = activation
	spec.PreLayerNorm = hasTensor(file, "v.pre_ln.weight")
	spec.PostLayerNorm = hasTensor(file, "v.post_ln.weight")
	spec.FusedQKV = make([]bool, spec.Layers)
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

func validatePaddleOCRCatalog(file *gguf.File, spec PaddleOCRSpec) ([]string, error) {
	required := map[string][]uint64{
		"v.patch_embd.weight":    {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.position_embd.weight": {uint64(spec.Hidden), uint64((spec.ImageSize / spec.PatchSize) * (spec.ImageSize / spec.PatchSize))},
		"mm.input_norm.weight":   {uint64(spec.Hidden)}, "mm.input_norm.bias": {uint64(spec.Hidden)},
		"mm.1.weight": {uint64(spec.Hidden * 4), uint64(spec.ProjectorIntermediate)},
		"mm.1.bias":   {uint64(spec.ProjectorIntermediate)},
		"mm.2.weight": {uint64(spec.ProjectorIntermediate), uint64(spec.OutputHidden)},
		"mm.2.bias":   {uint64(spec.OutputHidden)},
	}
	if err := addOptionalProjectorPair(file, required, "v.pre_ln.weight", "v.pre_ln.bias", []uint64{uint64(spec.Hidden)}); err != nil {
		return nil, err
	}
	if err := addOptionalProjectorPair(file, required, "v.post_ln.weight", "v.post_ln.bias", []uint64{uint64(spec.Hidden)}); err != nil {
		return nil, err
	}
	addOptionalProjectorTensor(file, required, "v.patch_embd.bias", []uint64{uint64(spec.Hidden)})
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, spec.FusedQKV)
	return validateProjectorTensorCatalog(file, required)
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
					value += weights[corner] * float64(corners[corner][channel]>>rgba16To8Shift)
				}
				output.Pix[index+channel] = clampUint8(value)
			}
			output.Pix[index+3] = opaqueAlpha
		}
	}
	return output
}

func (r *PaddleOCRRunner) EncodeImage(ctx context.Context, source image.Image, options RasterPatchOptions) (PaddleOCROutput, error) {
	if r == nil || r.file == nil {
		return PaddleOCROutput{}, errors.New("projector: runner is closed")
	}
	if err := r.spec.validate(); err != nil {
		return PaddleOCROutput{}, err
	}
	input, err := preprocessRasterPatches(source, rasterPatchPlan{
		patchSize: r.spec.PatchSize, mergeSize: r.spec.MergeSize,
		defaultBudget: pixelBudget{MinPixels: r.spec.MinPixels, MaxPixels: r.spec.MaxPixels, MaxAspectRatio: defaultVisionMaxAspectRatio},
		mean:          r.spec.ImageMean, std: r.spec.ImageStd, interpolation: rasterBilinear,
	}, options)
	if err != nil {
		return PaddleOCROutput{}, err
	}
	return r.encode(ctx, input)
}

func (r *PaddleOCRRunner) encode(ctx context.Context, input RasterPatchImage) (PaddleOCROutput, error) {
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
