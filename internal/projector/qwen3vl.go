package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const qwen3VLProjectorType = "qwen3vl_merger"

var deepstackTensorSuffixes = [...]string{"norm.weight", "norm.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}

type Qwen3VLSpec struct {
	visionBackboneSpec
	MergerIntermediate int
	OutputHidden       int
	MergeSize          int
	DeepstackLayers    []bool
}

type Qwen3VLPreprocessOptions pixelBudget

type Qwen3VLImage struct {
	PixelValues []float32
	GridT       int
	GridH       int
	GridW       int
}

type Qwen3VLOutput struct {
	Embeddings          reference.Value
	DeepstackEmbeddings []reference.Value
	GridT               int
	GridH               int
	GridW               int
	MergeSize           int
}

type Qwen3VLRunner struct {
	projectorResources
	spec Qwen3VLSpec
}

func DefaultQwen3VLPreprocessOptions() Qwen3VLPreprocessOptions {
	return Qwen3VLPreprocessOptions{
		MinPixels: 256 * 256,
		MaxPixels: 4096 * 4096,
	}
}

func DefaultQwen3VLVideoPreprocessOptions() Qwen3VLPreprocessOptions {
	return Qwen3VLPreprocessOptions{
		MinPixels: 56 * 56,
		MaxPixels: 3584 * 3584,
	}
}

func (r *Qwen3VLRunner) Spec() Qwen3VLSpec {
	if r == nil {
		return Qwen3VLSpec{}
	}
	return r.spec
}

func ReadQwen3VLSpec(file *gguf.File) (Qwen3VLSpec, error) {
	if useGELU, geluErr := metadataBool(file, "clip.use_gelu"); geluErr != nil {
		return Qwen3VLSpec{}, geluErr
	} else if !useGELU {
		return Qwen3VLSpec{}, errors.New("projector: Qwen3VL GELU is disabled")
	}
	deepstackLayers, err := readMetadataBoolArray(file, "clip.vision.is_deepstack_layers", false)
	if err != nil {
		return Qwen3VLSpec{}, err
	}
	spec := Qwen3VLSpec{}
	if err := readRotaryVisionBackbone(file, qwen3VLProjectorType, &spec.OutputHidden, &spec.visionBackboneSpec); err != nil {
		return Qwen3VLSpec{}, err
	}
	if err := readMetadataIntFields(file,
		metadataIntField{visionSpatialMergeKey, &spec.MergeSize},
	); err != nil {
		return Qwen3VLSpec{}, err
	}
	tensorDeepstack := make([]bool, spec.Layers)
	for layer := range tensorDeepstack {
		prefix := fmt.Sprintf("v.deepstack.%d.", layer)
		count := tensor.FirstOffset
		for _, suffix := range deepstackTensorSuffixes {
			if _, ok := file.Tensor(prefix + suffix); ok {
				count++
			}
		}
		if count != tensor.FirstOffset && count != len(deepstackTensorSuffixes) {
			return Qwen3VLSpec{}, fmt.Errorf("projector: deepstack layer %d has %d of %d tensors", layer, count, len(deepstackTensorSuffixes))
		}
		tensorDeepstack[layer] = count == len(deepstackTensorSuffixes)
	}
	if len(deepstackLayers) == tensor.FirstOffset {
		for _, enabled := range tensorDeepstack {
			if enabled {
				deepstackLayers = tensorDeepstack
				break
			}
		}
	} else if len(deepstackLayers) == spec.Layers {
		for layer := range tensorDeepstack {
			if deepstackLayers[layer] != tensorDeepstack[layer] {
				return Qwen3VLSpec{}, fmt.Errorf("projector: deepstack metadata differs from layer %d tensors", layer)
			}
		}
	}
	spec.DeepstackLayers = deepstackLayers
	merger, ok := file.Tensor(projectionFirstWeightTensor)
	spec.MergerIntermediate, ok = matrixRowsInt(merger, ok)
	if !ok {
		return Qwen3VLSpec{}, errors.New("projector: merger input tensor is unavailable or invalid")
	}
	if err := spec.validate(); err != nil {
		return Qwen3VLSpec{}, err
	}
	return spec, nil
}

func (s Qwen3VLSpec) validate() error {
	if err := s.visionBackboneSpec.validateRotary(); err != nil {
		return err
	}
	headWidth, headsOK := checked.DivExactInt(s.Hidden, s.Heads)
	_, rotaryOK := checked.DivExactInt(headWidth, tensor.PairedExtent*tensor.PairedExtent)
	if !checked.PositiveInts(s.MergerIntermediate, s.OutputHidden, s.MergeSize) || !headsOK || !rotaryOK {
		return fmt.Errorf("projector: invalid Qwen3VL metadata: %+v", s)
	}
	if len(s.DeepstackLayers) != tensor.FirstOffset && len(s.DeepstackLayers) != s.Layers {
		return fmt.Errorf("projector: deepstack flags = %d, want %d", len(s.DeepstackLayers), s.Layers)
	}
	return nil
}

func validateQwen3VLCatalog(file *gguf.File, spec Qwen3VLSpec) ([]string, error) {
	required := map[string][]uint64{
		visionPatchWeightTensor1:   {uint64(spec.PatchSize), uint64(spec.PatchSize), media.RGBChannels, uint64(spec.Hidden)},
		visionPostNormWeightTensor: {uint64(spec.Hidden)},
		visionPostNormBiasTensor:   {uint64(spec.Hidden)},
	}
	addTwoLayerProjectionCatalog(file, required, spec.Hidden*spec.MergeSize*spec.MergeSize, spec.MergerIntermediate, spec.OutputHidden, tensorRequired)
	positionSide := spec.ImageSize / spec.PatchSize
	addSpatialVisionEmbeddingCatalog(file, required, spec.visionBackboneSpec, positionSide*positionSide, tensorRequired)
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, nil, true, tensorRequired)
	for layer := range spec.Layers {
		if len(spec.DeepstackLayers) > layer && spec.DeepstackLayers[layer] {
			deepstackPrefix := fmt.Sprintf("v.deepstack.%d.", layer)
			mergedWidth := uint64(spec.Hidden * spec.MergeSize * spec.MergeSize)
			required[deepstackPrefix+"norm.weight"] = []uint64{mergedWidth}
			required[deepstackPrefix+"norm.bias"] = []uint64{mergedWidth}
			required[deepstackPrefix+"fc1.weight"] = []uint64{mergedWidth, mergedWidth}
			required[deepstackPrefix+"fc1.bias"] = []uint64{mergedWidth}
			required[deepstackPrefix+"fc2.weight"] = []uint64{mergedWidth, uint64(spec.OutputHidden)}
			required[deepstackPrefix+"fc2.bias"] = []uint64{uint64(spec.OutputHidden)}
		}
	}
	return validateProjectorTensorCatalog(file, required)
}

func PreprocessQwen3VLImage(source image.Image, spec Qwen3VLSpec, options Qwen3VLPreprocessOptions) (Qwen3VLImage, error) {
	if source == nil {
		return Qwen3VLImage{}, errors.New("projector: image is nil")
	}
	return preprocessQwen3VLFrames([]image.Image{source, source}, spec, options)
}

func PreprocessQwen3VLFrames(frames []image.Image, spec Qwen3VLSpec, options Qwen3VLPreprocessOptions) (Qwen3VLImage, error) {
	if len(frames) == tensor.FirstOffset {
		return Qwen3VLImage{}, errors.New("projector: video has no frames")
	}
	if options == (Qwen3VLPreprocessOptions{}) {
		options = DefaultQwen3VLVideoPreprocessOptions()
	}
	padded := slices.Clone(frames)
	if len(padded)%tensor.PairedExtent != tensor.FirstOffset {
		padded = append(padded, padded[len(padded)-tensor.SingletonExtent])
	}
	return preprocessQwen3VLFrames(padded, spec, options)
}

func preprocessQwen3VLFrames(frames []image.Image, spec Qwen3VLSpec, options Qwen3VLPreprocessOptions) (Qwen3VLImage, error) {
	if err := spec.validate(); err != nil {
		return Qwen3VLImage{}, err
	}
	if len(frames) == tensor.FirstOffset || len(frames)%tensor.PairedExtent != tensor.FirstOffset || frames[tensor.FirstOffset] == nil {
		return Qwen3VLImage{}, errors.New("projector: temporal frames must form non-empty pairs")
	}
	bounds := frames[tensor.FirstOffset].Bounds()
	height, width := bounds.Dy(), bounds.Dx()
	resizedH, resizedW, err := pixelBudget(options).resize(
		height, width, spec.PatchSize*spec.MergeSize,
	)
	if err != nil {
		return Qwen3VLImage{}, err
	}
	planes := make([][][]float32, len(frames))
	for frameIndex, frame := range frames {
		if frame == nil || frame.Bounds().Dx() != width || frame.Bounds().Dy() != height {
			return Qwen3VLImage{}, fmt.Errorf("projector: video frame %d geometry differs", frameIndex)
		}
		resized := media.ResizeBicubic(frame, resizedW, resizedH)
		planes[frameIndex] = make([][]float32, media.RGBChannels)
		for channel := range planes[frameIndex] {
			planes[frameIndex][channel] = make([]float32, resizedH*resizedW)
		}
		for y := range resizedH {
			for x := range resizedW {
				r, g, b, _ := resized.At(x, y).RGBA()
				values := [media.RGBChannels]uint32{r, g, b}
				for channel := range values {
					value := media.NormalizedRGBAChannel(values[channel])
					planes[frameIndex][channel][y*resizedW+x] = (value - spec.ImageMean[channel]) / spec.ImageStd[channel]
				}
			}
		}
	}
	gridH, gridW := resizedH/spec.PatchSize, resizedW/spec.PatchSize
	gridT := len(frames) / tensor.PairedExtent
	patchArea := spec.PatchSize * spec.PatchSize
	patchDim := tensor.PairedExtent * media.RGBChannels * patchArea
	pixels := make([]float32, gridT*gridH*gridW*patchDim)
	patch := tensor.FirstOffset
	for temporalGroup := range gridT {
		for blockH := range gridH / spec.MergeSize {
			for blockW := range gridW / spec.MergeSize {
				for mergeH := range spec.MergeSize {
					for mergeW := range spec.MergeSize {
						position := tensor.FirstOffset
						for channel := range media.RGBChannels {
							baseY := (blockH*spec.MergeSize + mergeH) * spec.PatchSize
							baseX := (blockW*spec.MergeSize + mergeW) * spec.PatchSize
							for temporal := range tensor.PairedExtent {
								plane := planes[temporalGroup*tensor.PairedExtent+temporal][channel]
								for py := range spec.PatchSize {
									row := (baseY+py)*resizedW + baseX
									for px := range spec.PatchSize {
										pixels[patch*patchDim+position] = plane[row+px]
										position++
									}
								}
							}
						}
						patch++
					}
				}
			}
		}
	}
	return Qwen3VLImage{PixelValues: pixels, GridT: gridT, GridH: gridH, GridW: gridW}, nil
}

func (r *Qwen3VLRunner) EncodeImage(ctx context.Context, source image.Image, options Qwen3VLPreprocessOptions) (Qwen3VLOutput, error) {
	return executePreparedProjector(
		ctx, r != nil && r.file != nil, source, r.Spec(), options,
		PreprocessQwen3VLImage, r.encodeGraph,
	)
}

func (r *Qwen3VLRunner) EncodeFrames(ctx context.Context, frames []image.Image, options Qwen3VLPreprocessOptions) (Qwen3VLOutput, error) {
	return executePreparedProjector(
		ctx, r != nil && r.file != nil, frames, r.Spec(), options,
		PreprocessQwen3VLFrames, r.encodeGraph,
	)
}

func mergedGrid(height, width, merge int) ([]int, []int) {
	rows := make([]int, height*width)
	columns := make([]int, height*width)
	index := tensor.FirstOffset
	for blockY := range height / merge {
		for blockX := range width / merge {
			for y := range merge {
				for x := range merge {
					rows[index] = blockY*merge + y
					columns[index] = blockX*merge + x
					index++
				}
			}
		}
	}
	return rows, columns
}

func metadataString(file *gguf.File, key string) (string, error) {
	value, ok := file.MetadataValue(key)
	if !ok || value.Type != gguf.ValueTypeString {
		return "", fmt.Errorf("projector: metadata %q must be string", key)
	}
	result, ok := value.Data.(string)
	if !ok {
		return "", fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	return result, nil
}

func metadataUint32(file *gguf.File, key string) (uint32, error) {
	value, ok := file.MetadataValue(key)
	if !ok || value.Type != gguf.ValueTypeUint32 {
		return uint32(tensor.FirstOffset), fmt.Errorf("projector: metadata %q must be uint32", key)
	}
	result, ok := value.Data.(uint32)
	if !ok {
		return uint32(tensor.FirstOffset), fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	return result, nil
}

func metadataFloat32(file *gguf.File, key string) (float32, error) {
	value, ok := file.MetadataValue(key)
	if !ok || value.Type != gguf.ValueTypeFloat32 {
		return float32(tensor.FirstOffset), fmt.Errorf("projector: metadata %q must be float32", key)
	}
	result, ok := value.Data.(float32)
	if !ok || !checked.Finite32(result) {
		return float32(tensor.FirstOffset), fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	return result, nil
}

func metadataBool(file *gguf.File, key string) (bool, error) {
	value, ok := file.MetadataValue(key)
	if !ok || value.Type != gguf.ValueTypeBool {
		return false, fmt.Errorf("projector: metadata %q must be bool", key)
	}
	result, ok := value.Data.(bool)
	if !ok {
		return false, fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	return result, nil
}

func metadataFloat32Array(file *gguf.File, key string, length int) ([]float32, error) {
	value, ok := file.MetadataValue(key)
	if !ok || value.Type != gguf.ValueTypeArray || value.ArrayType != gguf.ValueTypeFloat32 {
		return nil, fmt.Errorf("projector: metadata %q must be float32 array", key)
	}
	result, ok := value.Data.([]float32)
	if !ok || len(result) != length {
		return nil, fmt.Errorf("projector: metadata %q must contain %d values", key, length)
	}
	return result, nil
}
