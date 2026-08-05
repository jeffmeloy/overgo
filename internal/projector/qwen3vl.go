package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"runtime"
	"slices"
	"sync"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/reference"
)

const qwen3VLProjectorType = "qwen3vl_merger"

type Qwen3VLSpec struct {
	ImageSize          int
	PatchSize          int
	Hidden             int
	Intermediate       int
	MergerIntermediate int
	OutputHidden       int
	Layers             int
	Heads              int
	MergeSize          int
	LayerNormEpsilon   float32
	ImageMean          [3]float32
	ImageStd           [3]float32
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
	file *gguf.File
	spec Qwen3VLSpec
	cuda *projectorCUDA
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

func OpenQwen3VL(path string) (*Qwen3VLRunner, error) {
	return OpenQwen3VLWithOptions(path, Qwen3VLOpenOptions{})
}

type Qwen3VLOpenOptions = OpenOptions

func OpenQwen3VLWithOptions(path string, options Qwen3VLOpenOptions) (*Qwen3VLRunner, error) {
	return openCatalogProjector(path, options, "Qwen3-VL", nil,
		ReadQwen3VLSpec, validateQwen3VLCatalog,
		func(file *gguf.File, spec Qwen3VLSpec, cuda *projectorCUDA) *Qwen3VLRunner {
			return &Qwen3VLRunner{file: file, spec: spec, cuda: cuda}
		})
}

func (r *Qwen3VLRunner) Close() error {
	if r == nil {
		return nil
	}
	return closeProjectorResources(&r.file, &r.cuda)
}

func (r *Qwen3VLRunner) Spec() Qwen3VLSpec {
	if r == nil {
		return Qwen3VLSpec{}
	}
	return r.spec
}

func ReadQwen3VLSpec(file *gguf.File) (Qwen3VLSpec, error) {
	if err := validateVisionProjector(file, "clip.projector_type", qwen3VLProjectorType); err != nil {
		return Qwen3VLSpec{}, err
	}
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
	if err := readMetadataIntFields(file,
		metadataIntField{"clip.vision.image_size", &spec.ImageSize},
		metadataIntField{"clip.vision.patch_size", &spec.PatchSize},
		metadataIntField{"clip.vision.embedding_length", &spec.Hidden},
		metadataIntField{"clip.vision.feed_forward_length", &spec.Intermediate},
		metadataIntField{"clip.vision.projection_dim", &spec.OutputHidden},
		metadataIntField{"clip.vision.block_count", &spec.Layers},
		metadataIntField{"clip.vision.attention.head_count", &spec.Heads},
		metadataIntField{"clip.vision.spatial_merge_size", &spec.MergeSize},
	); err != nil {
		return Qwen3VLSpec{}, err
	}
	epsilon, err := metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return Qwen3VLSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", 3)
	if err != nil {
		return Qwen3VLSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", 3)
	if err != nil {
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
	spec.LayerNormEpsilon = epsilon
	spec.DeepstackLayers = deepstackLayers
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	merger, ok := file.Tensor("mm.0.weight")
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
	if s.ImageSize <= 0 || s.PatchSize <= 0 || s.Hidden <= 0 || s.Intermediate <= 0 ||
		s.MergerIntermediate <= 0 || s.OutputHidden <= 0 || s.Layers <= 0 || s.Heads <= 0 || s.MergeSize <= 0 ||
		s.Hidden%s.Heads != 0 || (s.Hidden/s.Heads)%4 != 0 || s.ImageSize%s.PatchSize != 0 || s.MergeSize != 2 ||
		s.LayerNormEpsilon <= 0 {
		return fmt.Errorf("projector: invalid Qwen3VL metadata: %+v", s)
	}
	if len(s.DeepstackLayers) != 0 && len(s.DeepstackLayers) != s.Layers {
		return fmt.Errorf("projector: deepstack flags = %d, want %d", len(s.DeepstackLayers), s.Layers)
	}
	for channel := range s.ImageStd {
		if s.ImageStd[channel] <= 0 || !finite32(s.ImageMean[channel]) || !finite32(s.ImageStd[channel]) {
			return fmt.Errorf("projector: invalid normalization channel %d", channel)
		}
	}
	return nil
}

func validateQwen3VLCatalog(file *gguf.File, spec Qwen3VLSpec) ([]string, error) {
	required := map[string][]uint64{
		"v.patch_embd.weight":    {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.patch_embd.weight.1":  {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.patch_embd.bias":      {uint64(spec.Hidden)},
		"v.position_embd.weight": {uint64(spec.Hidden), uint64((spec.ImageSize / spec.PatchSize) * (spec.ImageSize / spec.PatchSize))},
		"v.post_ln.weight":       {uint64(spec.Hidden)},
		"v.post_ln.bias":         {uint64(spec.Hidden)},
		"mm.0.weight":            {uint64(spec.Hidden * 4), uint64(spec.MergerIntermediate)},
		"mm.0.bias":              {uint64(spec.MergerIntermediate)},
		"mm.2.weight":            {uint64(spec.MergerIntermediate), uint64(spec.OutputHidden)},
		"mm.2.bias":              {uint64(spec.OutputHidden)},
	}
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for name, shape := range map[string][]uint64{
			"attn_qkv.weight": {uint64(spec.Hidden), uint64(3 * spec.Hidden)},
			"attn_qkv.bias":   {uint64(3 * spec.Hidden)},
			"attn_out.weight": {uint64(spec.Hidden), uint64(spec.Hidden)},
			"attn_out.bias":   {uint64(spec.Hidden)},
			"ffn_up.weight":   {uint64(spec.Hidden), uint64(spec.Intermediate)},
			"ffn_up.bias":     {uint64(spec.Intermediate)},
			"ffn_down.weight": {uint64(spec.Intermediate), uint64(spec.Hidden)},
			"ffn_down.bias":   {uint64(spec.Hidden)},
			"ln1.weight":      {uint64(spec.Hidden)},
			"ln1.bias":        {uint64(spec.Hidden)},
			"ln2.weight":      {uint64(spec.Hidden)},
			"ln2.bias":        {uint64(spec.Hidden)},
		} {
			required[prefix+name] = shape
		}
		if len(spec.DeepstackLayers) > layer && spec.DeepstackLayers[layer] {
			deepstackPrefix := fmt.Sprintf("v.deepstack.%d.", layer)
			mergedWidth := uint64(spec.Hidden * spec.MergeSize * spec.MergeSize)
			for name, shape := range map[string][]uint64{
				"norm.weight": {mergedWidth},
				"norm.bias":   {mergedWidth},
				"fc1.weight":  {mergedWidth, mergedWidth},
				"fc1.bias":    {mergedWidth},
				"fc2.weight":  {mergedWidth, uint64(spec.OutputHidden)},
				"fc2.bias":    {uint64(spec.OutputHidden)},
			} {
				required[deepstackPrefix+name] = shape
			}
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
	resizedH, resizedW, err := smartResizeAligned(
		height, width, spec.PatchSize*spec.MergeSize,
		options.MinPixels, options.MaxPixels, options.MaxAspectRatio,
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
					value := float32(values[channel]>>8) / 255
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
		return Qwen3VLOutput{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessQwen3VLImage(source, r.spec, options)
	if err != nil {
		return Qwen3VLOutput{}, err
	}
	return r.encode(ctx, input)
}

func (r *Qwen3VLRunner) EncodeFrames(ctx context.Context, frames []image.Image, options Qwen3VLPreprocessOptions) (Qwen3VLOutput, error) {
	if r == nil || r.file == nil {
		return Qwen3VLOutput{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessQwen3VLFrames(frames, r.spec, options)
	if err != nil {
		return Qwen3VLOutput{}, err
	}
	return r.encode(ctx, input)
}

func (r *Qwen3VLRunner) encode(ctx context.Context, input Qwen3VLImage) (Qwen3VLOutput, error) {
	return r.encodeGraph(ctx, input)
}

func linear(input, weight, bias []float32, rows, inputWidth, outputWidth int) []float32 {
	output := make([]float32, rows*outputWidth)
	parallelRows(rows, func(start, end int) {
		for row := start; row < end; row++ {
			source := input[row*inputWidth : (row+1)*inputWidth]
			destination := output[row*outputWidth : (row+1)*outputWidth]
			for channel := 0; channel < outputWidth; channel++ {
				accumulator := 0.0
				if bias != nil {
					accumulator = float64(bias[channel])
				}
				weights := weight[channel*inputWidth : (channel+1)*inputWidth]
				for index, value := range source {
					accumulator += float64(value) * float64(weights[index])
				}
				destination[channel] = float32(accumulator)
			}
		}
	})
	return output
}

func layerNorm(output, input, weight, bias []float32, rows, width int, epsilon float32) {
	parallelRows(rows, func(start, end int) {
		for row := start; row < end; row++ {
			source := input[row*width : (row+1)*width]
			destination := output[row*width : (row+1)*width]
			mean := 0.0
			for _, value := range source {
				mean += float64(value)
			}
			mean /= float64(width)
			variance := 0.0
			for _, value := range source {
				difference := float64(value) - mean
				variance += difference * difference
			}
			inverse := 1 / math.Sqrt(variance/float64(width)+float64(epsilon))
			for channel, value := range source {
				destination[channel] = float32((float64(value)-mean)*inverse)*weight[channel] + bias[channel]
			}
		}
	})
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
				values[0] += weight * float64(r>>8)
				values[1] += weight * float64(g>>8)
				values[2] += weight * float64(b>>8)
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
			output.Pix[index+3] = 255
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
			values[offset] = cubic((float64(offset+low) - center + 0.5) * inverseScale)
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

func cubic(value float64) float64 {
	const coefficient = -0.75
	value = math.Abs(value)
	if value < 1 {
		return ((coefficient+2)*value-(coefficient+3))*value*value + 1
	}
	if value < 2 {
		return (((value-5)*value+8)*value - 4) * coefficient
	}
	return 0
}

func clampUint8(value float64) uint8 {
	value = math.Floor(value + 0.5)
	if value <= 0 {
		return 0
	}
	if value >= 255 {
		return 255
	}
	return uint8(value)
}

func geluTanh(value float32) float32 {
	x := float64(value)
	return float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x))))
}

func parallelRows(rows int, run func(start, end int)) {
	workers := min(runtime.GOMAXPROCS(0), rows)
	if workers <= 1 {
		run(0, rows)
		return
	}
	var group sync.WaitGroup
	group.Add(workers)
	for worker := 0; worker < workers; worker++ {
		start := worker * rows / workers
		end := (worker + 1) * rows / workers
		go func() {
			defer group.Done()
			run(start, end)
		}()
	}
	group.Wait()
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
