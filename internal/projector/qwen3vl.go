package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"

	"overgo/internal/gguf"
	"overgo/internal/tensor/reference"
)

const qwen3VLProjectorType = "qwen3vl_merger"

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
		MinPixels:      256 * 256,
		MaxPixels:      4096 * 4096,
		MaxAspectRatio: defaultVisionMaxAspectRatio,
	}
}

func DefaultQwen3VLVideoPreprocessOptions() Qwen3VLPreprocessOptions {
	return Qwen3VLPreprocessOptions{
		MinPixels:      56 * 56,
		MaxPixels:      3584 * 3584,
		MaxAspectRatio: defaultVisionMaxAspectRatio,
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
	var deepstackLayers []bool
	if deepstack, ok := file.MetadataValue("clip.vision.is_deepstack_layers"); ok {
		layers, storageOK := deepstack.Data.([]bool)
		if deepstack.Type != gguf.ValueTypeArray || deepstack.ArrayType != gguf.ValueTypeBool || !storageOK {
			return Qwen3VLSpec{}, errors.New("projector: deepstack metadata must be a bool array")
		}
		deepstackLayers = slices.Clone(layers)
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
		count := 0
		for _, suffix := range []string{"norm.weight", "norm.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"} {
			if _, ok := file.Tensor(prefix + suffix); ok {
				count++
			}
		}
		if count != 0 && count != 6 {
			return Qwen3VLSpec{}, fmt.Errorf("projector: deepstack layer %d has %d of 6 tensors", layer, count)
		}
		tensorDeepstack[layer] = count == 6
	}
	if len(deepstackLayers) == 0 {
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
	if !ok || merger.Dimensions != 2 || merger.Shape[1] > uint64(^uint(0)>>1) {
		return Qwen3VLSpec{}, errors.New("projector: merger input tensor is unavailable or invalid")
	}
	spec.MergerIntermediate = int(merger.Shape[1])
	if err := spec.validate(); err != nil {
		return Qwen3VLSpec{}, err
	}
	return spec, nil
}

func (s Qwen3VLSpec) validate() error {
	if err := s.visionBackboneSpec.validateRotary(); err != nil {
		return err
	}
	if s.MergerIntermediate <= 0 || s.OutputHidden <= 0 || (s.Hidden/s.Heads)%visionRoPEComponentCount != 0 || s.MergeSize <= 0 {
		return fmt.Errorf("projector: invalid Qwen3VL metadata: %+v", s)
	}
	if len(s.DeepstackLayers) != 0 && len(s.DeepstackLayers) != s.Layers {
		return fmt.Errorf("projector: deepstack flags = %d, want %d", len(s.DeepstackLayers), s.Layers)
	}
	return nil
}

func validateQwen3VLCatalog(file *gguf.File, spec Qwen3VLSpec) ([]string, error) {
	required := map[string][]uint64{
		visionPatchWeightTensor1:   {uint64(spec.PatchSize), uint64(spec.PatchSize), rgbChannelCount, uint64(spec.Hidden)},
		visionPostNormWeightTensor: {uint64(spec.Hidden)},
		visionPostNormBiasTensor:   {uint64(spec.Hidden)},
	}
	addTwoLayerProjectionCatalog(file, required, spec.Hidden*spec.MergeSize*spec.MergeSize, spec.MergerIntermediate, spec.OutputHidden, tensorRequired)
	positionSide := spec.ImageSize / spec.PatchSize
	addSpatialVisionEmbeddingCatalog(file, required, spec.visionBackboneSpec, positionSide*positionSide, tensorRequired)
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, nil, true, tensorRequired)
	for layer := 0; layer < spec.Layers; layer++ {
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
	if len(frames) == 0 {
		return Qwen3VLImage{}, errors.New("projector: video has no frames")
	}
	if options == (Qwen3VLPreprocessOptions{}) {
		options = DefaultQwen3VLVideoPreprocessOptions()
	}
	padded := slices.Clone(frames)
	if len(padded)%2 != 0 {
		padded = append(padded, padded[len(padded)-1])
	}
	return preprocessQwen3VLFrames(padded, spec, options)
}

func preprocessQwen3VLFrames(frames []image.Image, spec Qwen3VLSpec, options Qwen3VLPreprocessOptions) (Qwen3VLImage, error) {
	if err := spec.validate(); err != nil {
		return Qwen3VLImage{}, err
	}
	if len(frames) == 0 || len(frames)%2 != 0 || frames[0] == nil {
		return Qwen3VLImage{}, errors.New("projector: temporal frames must form non-empty pairs")
	}
	bounds := frames[0].Bounds()
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
		resized := resizeImageBicubic(frame, resizedW, resizedH)
		planes[frameIndex] = make([][]float32, 3)
		for channel := range planes[frameIndex] {
			planes[frameIndex][channel] = make([]float32, resizedH*resizedW)
		}
		for y := 0; y < resizedH; y++ {
			for x := 0; x < resizedW; x++ {
				r, g, b, _ := resized.At(x, y).RGBA()
				values := [3]uint32{r, g, b}
				for channel := range values {
					value := normalizedImageChannel(values[channel])
					planes[frameIndex][channel][y*resizedW+x] = (value - spec.ImageMean[channel]) / spec.ImageStd[channel]
				}
			}
		}
	}
	gridH, gridW := resizedH/spec.PatchSize, resizedW/spec.PatchSize
	gridT := len(frames) / 2
	patchArea := spec.PatchSize * spec.PatchSize
	patchDim := 2 * 3 * patchArea
	pixels := make([]float32, gridT*gridH*gridW*patchDim)
	patch := 0
	for temporalGroup := 0; temporalGroup < gridT; temporalGroup++ {
		for blockH := 0; blockH < gridH/spec.MergeSize; blockH++ {
			for blockW := 0; blockW < gridW/spec.MergeSize; blockW++ {
				for mergeH := 0; mergeH < spec.MergeSize; mergeH++ {
					for mergeW := 0; mergeW < spec.MergeSize; mergeW++ {
						position := 0
						for channel := 0; channel < 3; channel++ {
							baseY := (blockH*spec.MergeSize + mergeH) * spec.PatchSize
							baseX := (blockW*spec.MergeSize + mergeW) * spec.PatchSize
							for temporal := 0; temporal < 2; temporal++ {
								plane := planes[temporalGroup*2+temporal][channel]
								for py := 0; py < spec.PatchSize; py++ {
									row := (baseY+py)*resizedW + baseX
									for px := 0; px < spec.PatchSize; px++ {
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
	if r == nil || r.file == nil {
		return Qwen3VLOutput{}, errRunnerClosed
	}
	input, err := PreprocessQwen3VLImage(source, r.spec, options)
	if err != nil {
		return Qwen3VLOutput{}, err
	}
	return r.encodeGraph(ctx, input)
}

func (r *Qwen3VLRunner) EncodeFrames(ctx context.Context, frames []image.Image, options Qwen3VLPreprocessOptions) (Qwen3VLOutput, error) {
	if r == nil || r.file == nil {
		return Qwen3VLOutput{}, errRunnerClosed
	}
	input, err := PreprocessQwen3VLFrames(frames, r.spec, options)
	if err != nil {
		return Qwen3VLOutput{}, err
	}
	return r.encodeGraph(ctx, input)
}

func mergedGrid(height, width, merge int) ([]int, []int) {
	rows := make([]int, height*width)
	columns := make([]int, height*width)
	index := 0
	for blockY := 0; blockY < height/merge; blockY++ {
		for blockX := 0; blockX < width/merge; blockX++ {
			for y := 0; y < merge; y++ {
				for x := 0; x < merge; x++ {
					rows[index] = blockY*merge + y
					columns[index] = blockX*merge + x
					index++
				}
			}
		}
	}
	return rows, columns
}

func smartResizeAligned(height, width, factor, minPixels, maxPixels, maxAspectRatio int) (int, int, error) {
	if height <= 0 || width <= 0 || factor <= 0 || minPixels <= 0 || maxPixels < minPixels || maxAspectRatio <= 0 {
		return 0, 0, errors.New("projector: invalid resize contract")
	}
	high, low := max(height, width), min(height, width)
	if float64(high)/float64(low) > float64(maxAspectRatio) {
		return 0, 0, fmt.Errorf("projector: image aspect ratio exceeds %d", maxAspectRatio)
	}
	factorFloat := float64(factor)
	resizedH := max(factor, int(math.RoundToEven(float64(height)/factorFloat))*factor)
	resizedW := max(factor, int(math.RoundToEven(float64(width)/factorFloat))*factor)
	switch {
	case resizedH*resizedW > maxPixels:
		beta := math.Sqrt(float64(height*width) / float64(maxPixels))
		resizedH = max(factor, int(math.Floor(float64(height)/beta/factorFloat))*factor)
		resizedW = max(factor, int(math.Floor(float64(width)/beta/factorFloat))*factor)
	case resizedH*resizedW < minPixels:
		beta := math.Sqrt(float64(minPixels) / float64(height*width))
		resizedH = int(math.Ceil(float64(height)*beta/factorFloat)) * factor
		resizedW = int(math.Ceil(float64(width)*beta/factorFloat)) * factor
	}
	return resizedH, resizedW, nil
}

func resizeImageBicubic(source image.Image, width, height int) *image.RGBA {
	bounds := source.Bounds()
	inputWidth, inputHeight := bounds.Dx(), bounds.Dy()
	xMin, xCount, xWeights := antialiasWeights(inputWidth, width)
	intermediate := make([][3]uint8, inputHeight*width)
	for y := 0; y < inputHeight; y++ {
		for outX := 0; outX < width; outX++ {
			values := [3]float64{}
			for offset := 0; offset < xCount[outX]; offset++ {
				r, g, b, _ := source.At(bounds.Min.X+xMin[outX]+offset, bounds.Min.Y+y).RGBA()
				weight := xWeights[outX][offset]
				values[0] += weight * float64(r>>rgba16To8Shift)
				values[1] += weight * float64(g>>rgba16To8Shift)
				values[2] += weight * float64(b>>rgba16To8Shift)
			}
			for channel := range values {
				intermediate[y*width+outX][channel] = clampUint8(values[channel])
			}
		}
	}
	yMin, yCount, yWeights := antialiasWeights(inputHeight, height)
	output := image.NewRGBA(image.Rect(0, 0, width, height))
	for outY := 0; outY < height; outY++ {
		for x := 0; x < width; x++ {
			values := [3]float64{}
			for offset := 0; offset < yCount[outY]; offset++ {
				pixel := intermediate[(yMin[outY]+offset)*width+x]
				weight := yWeights[outY][offset]
				for channel := range values {
					values[channel] += weight * float64(pixel[channel])
				}
			}
			index := outY*output.Stride + x*4
			output.Pix[index] = clampUint8(values[0])
			output.Pix[index+1] = clampUint8(values[1])
			output.Pix[index+2] = clampUint8(values[2])
			output.Pix[index+3] = opaqueAlpha
		}
	}
	return output
}

func antialiasWeights(input, output int) ([]int, []int, [][]float64) {
	scale := float64(input) / float64(output)
	support, inverseScale := 2.0, 1.0
	if scale >= 1 {
		support, inverseScale = 2*scale, 1/scale
	}
	minimum := make([]int, output)
	count := make([]int, output)
	weights := make([][]float64, output)
	for index := 0; index < output; index++ {
		center := scale * (float64(index) + 0.5)
		low := max(0, int(center-support+0.5))
		high := min(input, int(center+support+0.5))
		values := make([]float64, high-low)
		total := 0.0
		for offset := range values {
			values[offset] = cubicInterpolationWeight((float64(offset+low) - center + 0.5) * inverseScale)
			total += values[offset]
		}
		if total != 0 {
			for offset := range values {
				values[offset] /= total
			}
		}
		minimum[index], count[index], weights[index] = low, len(values), values
	}
	return minimum, count, weights
}

func clampUint8(value float64) uint8 {
	value = math.Floor(value + 0.5)
	if value <= 0 {
		return 0
	}
	if value >= maxUint8Channel {
		return maxUint8Channel
	}
	return uint8(value)
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
		return 0, fmt.Errorf("projector: metadata %q must be uint32", key)
	}
	result, ok := value.Data.(uint32)
	if !ok {
		return 0, fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	return result, nil
}

func metadataFloat32(file *gguf.File, key string) (float32, error) {
	value, ok := file.MetadataValue(key)
	if !ok || value.Type != gguf.ValueTypeFloat32 {
		return 0, fmt.Errorf("projector: metadata %q must be float32", key)
	}
	result, ok := value.Data.(float32)
	if !ok || !finite32(result) {
		return 0, fmt.Errorf("projector: metadata %q has invalid storage", key)
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

func finite32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
