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

type PaddleOCRSpec struct {
	visionBackboneSpec
	ProjectorIntermediate int
	OutputHidden          int
	MergeSize             int
	MinPixels             int
	MaxPixels             int
	Activation            visionActivation
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

func (r *PaddleOCRRunner) Spec() PaddleOCRSpec {
	if r == nil {
		return PaddleOCRSpec{}
	}
	return r.spec
}

func ReadPaddleOCRSpec(file *gguf.File) (PaddleOCRSpec, error) {
	spec := PaddleOCRSpec{}
	if err := readRotaryVisionBackbone(file, paddleOCRProjectorType, &spec.OutputHidden, &spec.visionBackboneSpec); err != nil {
		return PaddleOCRSpec{}, err
	}
	if err := readMetadataIntFields(file,
		metadataIntField{visionMinPixelsKey, &spec.MinPixels},
		metadataIntField{visionMaxPixelsKey, &spec.MaxPixels},
		metadataIntField{visionSpatialMergeKey, &spec.MergeSize},
	); err != nil {
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
	spec.ProjectorIntermediate = int(merger.Shape[1])
	spec.Activation = activation
	spec.PreLayerNorm = hasTensor(file, visionPreNormWeightTensor)
	spec.PostLayerNorm = hasTensor(file, visionPostNormWeightTensor)
	spec.FusedQKV = make([]bool, spec.Layers)
	for layer := range spec.FusedQKV {
		spec.FusedQKV[layer] = hasTensor(file, fmt.Sprintf("v.blk.%d.attn_qkv.weight", layer))
	}
	if err := spec.validate(); err != nil {
		return PaddleOCRSpec{}, err
	}
	return spec, nil
}

func readPaddleOCRActivation(file *gguf.File) (visionActivation, error) {
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
		return visionGELU, nil
	}
	if useSiLU {
		return visionSiLU, nil
	}
	return visionQuickGELU, nil
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
	if err := s.visionBackboneSpec.validateRotary(); err != nil {
		return err
	}
	if s.ProjectorIntermediate <= 0 || s.OutputHidden <= 0 || s.MergeSize <= 0 ||
		s.MinPixels <= 0 || s.MaxPixels < s.MinPixels || (s.Hidden/s.Heads)%visionRoPEComponentCount != 0 || len(s.FusedQKV) != s.Layers {
		return fmt.Errorf("projector: invalid PaddleOCR metadata: %+v", s)
	}
	return nil
}

func validatePaddleOCRCatalog(file *gguf.File, spec PaddleOCRSpec) ([]string, error) {
	required := map[string][]uint64{
		"mm.input_norm.weight": {uint64(spec.Hidden)}, "mm.input_norm.bias": {uint64(spec.Hidden)},
		"mm.1.weight": {uint64(spec.Hidden * spec.MergeSize * spec.MergeSize), uint64(spec.ProjectorIntermediate)},
		"mm.1.bias":   {uint64(spec.ProjectorIntermediate)},
		"mm.2.weight": {uint64(spec.ProjectorIntermediate), uint64(spec.OutputHidden)},
		"mm.2.bias":   {uint64(spec.OutputHidden)},
	}
	positionSide := spec.ImageSize / spec.PatchSize
	addSpatialVisionEmbeddingCatalog(file, required, spec.visionBackboneSpec, positionSide*positionSide, tensorOptional)
	if err := addOptionalVisionNormCatalog(file, required, spec.Hidden); err != nil {
		return nil, err
	}
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, spec.FusedQKV, false, tensorOptional)
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
		return PaddleOCROutput{}, errRunnerClosed
	}
	plan := r.spec.visionBackboneSpec.rasterPlan(r.spec.MergeSize, r.spec.MinPixels, r.spec.MaxPixels, rasterBilinear)
	return encodeRasterPatches(ctx, source, options, plan, r.spec.validate, r.encodeGraph)
}

func paddleOCRGrid(height, width int) ([]int, []int) {
	rows := make([]int, height*width)
	columns := make([]int, height*width)
	for index := range rows {
		rows[index], columns[index] = index/width, index%width
	}
	return rows, columns
}
