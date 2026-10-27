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

type qwen3VLSpec struct {
	visionBackboneSpec
	MergerIntermediate int
	OutputHidden       int
	MergeSize          int
	DeepstackLayers    []bool
}

type qwen3VLPreprocessOptions = mediaPixelBudget

type Qwen3VLImage struct {
	PixelValues []float32
	GridT       int
	GridH       int
	GridW       int
}

type qwen3VLOutput struct {
	Embeddings          reference.Value
	DeepstackEmbeddings []reference.Value
	GridT               int
	GridH               int
	GridW               int
	MergeSize           int
}

type Qwen3VLRunner struct {
	projectorResources
	mediaPreprocessOwner
	spec qwen3VLSpec
}

func (r *Qwen3VLRunner) Spec() qwen3VLSpec {
	if r == nil {
		return qwen3VLSpec{}
	}
	return r.spec
}

func readQwen3VLSpec(file *gguf.File) (qwen3VLSpec, error) {
	if useGELU, geluErr := metadataBool(file, visionUseGELUKey); geluErr != nil {
		return qwen3VLSpec{}, geluErr
	} else if !useGELU {
		return qwen3VLSpec{}, errors.New("projector: Qwen3VL GELU is disabled")
	}
	deepstackLayers, err := readMetadataBoolArray(file, "clip.vision.is_deepstack_layers", false)
	if err != nil {
		return qwen3VLSpec{}, err
	}
	spec := qwen3VLSpec{}
	if err := readRotaryVisionBackbone(file, qwen3VLProjectorType, &spec.OutputHidden, &spec.visionBackboneSpec); err != nil {
		return qwen3VLSpec{}, err
	}
	if err := readMetadataIntFields(file,
		metadataIntField{visionSpatialMergeKey, &spec.MergeSize},
	); err != nil {
		return qwen3VLSpec{}, err
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
			return qwen3VLSpec{}, fmt.Errorf("projector: deepstack layer %d has %d of %d tensors", layer, count, len(deepstackTensorSuffixes))
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
				return qwen3VLSpec{}, fmt.Errorf("projector: deepstack metadata differs from layer %d tensors", layer)
			}
		}
	}
	spec.DeepstackLayers = deepstackLayers
	merger, ok := file.Tensor(projectionFirstWeightTensor)
	spec.MergerIntermediate, ok = matrixRowsInt(merger, ok)
	if !ok {
		return qwen3VLSpec{}, errors.New("projector: merger input tensor is unavailable or invalid")
	}
	if err := spec.validate(); err != nil {
		return qwen3VLSpec{}, err
	}
	return spec, nil
}

func (s qwen3VLSpec) validate() error {
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

// matrixTensors names the tensors the Qwen3-VL graph reads only as a product's weight (qwen3vl_cuda.go): each block's
// attention and FFN projections, the deepstack mergers' layers and the projection's.
func (s qwen3VLSpec) matrixTensors() []string {
	var names []string
	for layer := range s.Layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		names = append(names, prefix+"attn_qkv.weight", prefix+"attn_out.weight", prefix+"ffn_up.weight", prefix+"ffn_down.weight")
		if len(s.DeepstackLayers) > layer && s.DeepstackLayers[layer] {
			deepstack := fmt.Sprintf("v.deepstack.%d.", layer)
			names = append(names, deepstack+"fc1.weight", deepstack+"fc2.weight")
		}
	}
	return append(names, projectionFirstWeightTensor, projectionSecondWeightTensor)
}

func validateQwen3VLCatalog(file *gguf.File, spec qwen3VLSpec) ([]string, error) {
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

func PreprocessQwen3VLFrames(frames []image.Image, spec qwen3VLSpec, options qwen3VLPreprocessOptions) (Qwen3VLImage, error) {
	if len(frames) == tensor.FirstOffset {
		return Qwen3VLImage{}, errors.New("projector: video has no frames")
	}
	padded := slices.Clone(frames)
	if len(padded)%tensor.PairedExtent != tensor.FirstOffset {
		padded = append(padded, padded[len(padded)-tensor.SingletonExtent])
	}
	return preprocessQwen3VLFrames(padded, spec, options)
}

func preprocessQwen3VLFrames(frames []image.Image, spec qwen3VLSpec, options qwen3VLPreprocessOptions) (Qwen3VLImage, error) {
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

func (r *Qwen3VLRunner) EncodeImage(ctx context.Context, source image.Image, options qwen3VLPreprocessOptions) (qwen3VLOutput, error) {
	return r.EncodeFrames(ctx, []image.Image{source}, options)
}

func (r *Qwen3VLRunner) EncodeFrames(ctx context.Context, frames []image.Image, options qwen3VLPreprocessOptions) (qwen3VLOutput, error) {
	return executePreparedProjector(
		ctx, r != nil && r.file != nil, frames, r.Spec(), options,
		PreprocessQwen3VLFrames, func(ctx context.Context, input Qwen3VLImage) (qwen3VLOutput, error) {
			return r.encodeTemporal(ctx, input, qwen3VLEncodePatches)
		},
	)
}

// qwen3VLEncodePatches bounds the patches one encode reads: a 1024-token picture's, whose workspace fits the device
// memory a hybrid model's expert cache leaves (a VidGen clip encoded whole asked 3.6 GB of its 2 GiB).
const qwen3VLEncodePatches = 4096

// encodeTemporal encodes a video's temporal groups a few at a time, at most patches a time where a group holds no
// more: the encoder attends within a group and every other operation reads a patch or a merged cell, so the rows are
// those of one encode, in its order.
func (r *Qwen3VLRunner) encodeTemporal(ctx context.Context, input Qwen3VLImage, patches int) (qwen3VLOutput, error) {
	spatial := input.GridH * input.GridW
	groups := max(1, patches/max(1, spatial))
	if input.GridT <= groups {
		return r.encodeGraph(ctx, input)
	}
	groupValues := len(input.PixelValues) / input.GridT
	var whole qwen3VLOutput
	for start := 0; start < input.GridT; start += groups {
		count := min(groups, input.GridT-start)
		part, err := r.encodeGraph(ctx, Qwen3VLImage{
			PixelValues: input.PixelValues[start*groupValues : (start+count)*groupValues],
			GridT:       count, GridH: input.GridH, GridW: input.GridW,
		})
		if err != nil {
			return qwen3VLOutput{}, err
		}
		if start == 0 {
			whole = part
			continue
		}
		whole.Embeddings = appendRows(whole.Embeddings, part.Embeddings)
		for stream := range whole.DeepstackEmbeddings {
			whole.DeepstackEmbeddings[stream] = appendRows(whole.DeepstackEmbeddings[stream], part.DeepstackEmbeddings[stream])
		}
	}
	whole.GridT = input.GridT
	return whole, nil
}

// appendRows appends right's rows to left's, of one encoder's width.
func appendRows(left, right reference.Value) reference.Value {
	data := append(left.Data, right.Data...)
	return reference.Value{Shape: tensor.MustShape(left.Shape.Dims[0], uint64(len(data))/left.Shape.Dims[0]), Data: data}
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
