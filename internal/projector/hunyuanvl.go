package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
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
	for name, shape := range required {
		info, ok := file.Tensor(name)
		if !ok {
			return fmt.Errorf("projector: missing tensor %q", name)
		}
		if int(info.Dimensions) != len(shape) {
			return fmt.Errorf("projector: tensor %q rank %d, want %d", name, info.Dimensions, len(shape))
		}
		for dimension, want := range shape {
			if info.Shape[dimension] != want {
				return fmt.Errorf("projector: tensor %q shape %v, want %v", name, info.Shape[:info.Dimensions], shape)
			}
		}
	}
	return nil
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
	if r.cuda != nil {
		return r.encodeCUDA(ctx, input)
	}
	rows := input.GridH * input.GridW
	patchWidth := 3 * r.spec.PatchSize * r.spec.PatchSize
	if rows <= 0 || input.GridH%r.spec.MergeSize != 0 || input.GridW%r.spec.MergeSize != 0 || len(input.PixelValues) != rows*patchWidth {
		return HunyuanVLOutput{}, errors.New("projector: Hunyuan-VL input shape is inconsistent")
	}
	hidden, err := r.patchEmbedding(ctx, input)
	if err != nil {
		return HunyuanVLOutput{}, err
	}
	if err := r.addPositions(ctx, hidden, input.GridH, input.GridW); err != nil {
		return HunyuanVLOutput{}, err
	}
	if r.spec.PreLayerNorm {
		hidden, err = r.affineNormalize(ctx, hidden, rows, "v.pre_ln")
		if err != nil {
			return HunyuanVLOutput{}, err
		}
	}
	for layer := 0; layer < r.spec.Layers; layer++ {
		if err := ctx.Err(); err != nil {
			return HunyuanVLOutput{}, err
		}
		if err := r.runLayer(ctx, hidden, rows, layer); err != nil {
			return HunyuanVLOutput{}, err
		}
	}
	if r.spec.PostLayerNorm {
		hidden, err = r.affineNormalize(ctx, hidden, rows, "v.post_ln")
		if err != nil {
			return HunyuanVLOutput{}, err
		}
	}
	embeddings, err := r.project(ctx, hidden, input.GridH, input.GridW)
	if err != nil {
		return HunyuanVLOutput{}, err
	}
	return HunyuanVLOutput{Embeddings: embeddings, GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize}, nil
}

func (r *HunyuanVLRunner) patchEmbedding(ctx context.Context, input HunyuanVLImage) ([]float32, error) {
	weight, err := r.load(ctx, "v.patch_embd.weight")
	if err != nil {
		return nil, err
	}
	bias, err := r.optionalBias(ctx, "v.patch_embd.bias")
	if err != nil {
		return nil, err
	}
	rows := input.GridH * input.GridW
	patchWidth := 3 * r.spec.PatchSize * r.spec.PatchSize
	return linear(input.PixelValues, weight.Data, bias, rows, patchWidth, r.spec.Hidden), nil
}

func (r *HunyuanVLRunner) addPositions(ctx context.Context, hidden []float32, gridH, gridW int) error {
	position, err := r.load(ctx, "v.position_embd.weight")
	if err != nil {
		return err
	}
	positionCount := len(position.Data) / r.spec.Hidden
	side := int(math.Round(math.Sqrt(float64(positionCount))))
	if side*side != positionCount {
		return errors.New("projector: Hunyuan-VL position table is not square")
	}
	sx := (float64(gridW) + 0.1) / float64(side)
	sy := (float64(gridH) + 0.1) / float64(side)
	parallelRows(gridH, func(start, end int) {
		for y := start; y < end; y++ {
			fy := (float64(y)+0.5)/sy - 0.5
			y0 := max(0, min(side-1, int(math.Floor(fy))))
			y1 := max(0, min(side-1, int(math.Floor(fy))+1))
			wy1 := min(1.0, max(0.0, fy-float64(y0)))
			for x := 0; x < gridW; x++ {
				fx := (float64(x)+0.5)/sx - 0.5
				x0 := max(0, min(side-1, int(math.Floor(fx))))
				x1 := max(0, min(side-1, int(math.Floor(fx))+1))
				wx1 := min(1.0, max(0.0, fx-float64(x0)))
				indexes := [4]int{y0*side + x0, y0*side + x1, y1*side + x0, y1*side + x1}
				weights := [4]float64{(1 - wy1) * (1 - wx1), (1 - wy1) * wx1, wy1 * (1 - wx1), wy1 * wx1}
				row := hidden[(y*gridW+x)*r.spec.Hidden:]
				for channel := 0; channel < r.spec.Hidden; channel++ {
					value := 0.0
					for corner := range indexes {
						value += weights[corner] * float64(position.Data[indexes[corner]*r.spec.Hidden+channel])
					}
					row[channel] += float32(value)
				}
			}
		}
	})
	return nil
}

func (r *HunyuanVLRunner) affineNormalize(ctx context.Context, input []float32, rows int, prefix string) ([]float32, error) {
	weight, bias, err := r.loadPair(ctx, prefix+".weight", prefix+".bias")
	if err != nil {
		return nil, err
	}
	output := make([]float32, len(input))
	layerNorm(output, input, weight.Data, bias.Data, rows, len(input)/rows, r.spec.LayerNormEpsilon)
	return output, nil
}

func (r *HunyuanVLRunner) runLayer(ctx context.Context, hidden []float32, rows, layer int) error {
	prefix := fmt.Sprintf("v.blk.%d.", layer)
	normalized, err := r.affineNormalize(ctx, hidden, rows, prefix+"ln1")
	if err != nil {
		return err
	}
	qkv, err := r.projectQKV(ctx, normalized, rows, prefix, layer)
	if err != nil {
		return err
	}
	attention := visionAttention(qkv, rows, r.spec.Hidden, r.spec.Heads)
	outWeight, err := r.load(ctx, prefix+"attn_out.weight")
	if err != nil {
		return err
	}
	outBias, err := r.optionalBias(ctx, prefix+"attn_out.bias")
	if err != nil {
		return err
	}
	projected := linear(attention, outWeight.Data, outBias, rows, r.spec.Hidden, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += projected[index]
	}
	normalized, err = r.affineNormalize(ctx, hidden, rows, prefix+"ln2")
	if err != nil {
		return err
	}
	upWeight, err := r.load(ctx, prefix+"ffn_up.weight")
	if err != nil {
		return err
	}
	upBias, err := r.optionalBias(ctx, prefix+"ffn_up.bias")
	if err != nil {
		return err
	}
	up := linear(normalized, upWeight.Data, upBias, rows, r.spec.Hidden, r.spec.Intermediate)
	for index, value := range up {
		up[index] = geluTanh(value)
	}
	downWeight, err := r.load(ctx, prefix+"ffn_down.weight")
	if err != nil {
		return err
	}
	downBias, err := r.optionalBias(ctx, prefix+"ffn_down.bias")
	if err != nil {
		return err
	}
	down := linear(up, downWeight.Data, downBias, rows, r.spec.Intermediate, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += down[index]
	}
	return nil
}

func (r *HunyuanVLRunner) projectQKV(ctx context.Context, input []float32, rows int, prefix string, layer int) ([]float32, error) {
	if r.spec.FusedQKV[layer] {
		weight, err := r.load(ctx, prefix+"attn_qkv.weight")
		if err != nil {
			return nil, err
		}
		bias, err := r.optionalBias(ctx, prefix+"attn_qkv.bias")
		if err != nil {
			return nil, err
		}
		return linear(input, weight.Data, bias, rows, r.spec.Hidden, 3*r.spec.Hidden), nil
	}
	qkv := make([]float32, rows*3*r.spec.Hidden)
	for partIndex, part := range []string{"q", "k", "v"} {
		weight, err := r.load(ctx, prefix+"attn_"+part+".weight")
		if err != nil {
			return nil, err
		}
		bias, err := r.optionalBias(ctx, prefix+"attn_"+part+".bias")
		if err != nil {
			return nil, err
		}
		projected := linear(input, weight.Data, bias, rows, r.spec.Hidden, r.spec.Hidden)
		for row := 0; row < rows; row++ {
			copy(qkv[row*3*r.spec.Hidden+partIndex*r.spec.Hidden:], projected[row*r.spec.Hidden:(row+1)*r.spec.Hidden])
		}
	}
	return qkv, nil
}

func visionAttention(qkv []float32, rows, hidden, heads int) []float32 {
	headWidth := hidden / heads
	output := make([]float32, rows*hidden)
	scale := 1 / math.Sqrt(float64(headWidth))
	parallelRows(heads, func(startHead, endHead int) {
		scores := make([]float64, rows)
		for head := startHead; head < endHead; head++ {
			for query := 0; query < rows; query++ {
				qOffset := query*3*hidden + head*headWidth
				maximum := math.Inf(-1)
				for key := 0; key < rows; key++ {
					kOffset := key*3*hidden + hidden + head*headWidth
					dot := 0.0
					for dimension := 0; dimension < headWidth; dimension++ {
						dot += float64(qkv[qOffset+dimension]) * float64(qkv[kOffset+dimension])
					}
					scores[key] = dot * scale
					maximum = max(maximum, scores[key])
				}
				total := 0.0
				for key := range scores {
					scores[key] = math.Exp(scores[key] - maximum)
					total += scores[key]
				}
				for dimension := 0; dimension < headWidth; dimension++ {
					value := 0.0
					for source := 0; source < rows; source++ {
						vOffset := source*3*hidden + 2*hidden + head*headWidth
						value += scores[source] / total * float64(qkv[vOffset+dimension])
					}
					output[query*hidden+head*headWidth+dimension] = float32(value)
				}
			}
		}
	})
	return output
}

func (r *HunyuanVLRunner) project(ctx context.Context, hidden []float32, gridH, gridW int) (reference.Value, error) {
	preNorm, err := r.load(ctx, "mm.pre_norm.weight")
	if err != nil {
		return reference.Value{}, err
	}
	rmsNormInPlace(hidden, preNorm.Data, gridH*gridW, r.spec.Hidden, r.spec.LayerNormEpsilon)
	mergedH, mergedW := gridH/r.spec.MergeSize, gridW/r.spec.MergeSize
	patchWidth := r.spec.Hidden * r.spec.MergeSize * r.spec.MergeSize
	patches := make([]float32, mergedH*mergedW*patchWidth)
	for blockY := 0; blockY < mergedH; blockY++ {
		for blockX := 0; blockX < mergedW; blockX++ {
			destination := (blockY*mergedW + blockX) * patchWidth
			for channel := 0; channel < r.spec.Hidden; channel++ {
				for y := 0; y < r.spec.MergeSize; y++ {
					for x := 0; x < r.spec.MergeSize; x++ {
						source := ((blockY*r.spec.MergeSize+y)*gridW+blockX*r.spec.MergeSize+x)*r.spec.Hidden + channel
						patches[destination] = hidden[source]
						destination++
					}
				}
			}
		}
	}
	conv0Weight, conv0Bias, err := r.loadPair(ctx, "mm.0.weight", "mm.0.bias")
	if err != nil {
		return reference.Value{}, err
	}
	conv0 := linear(patches, conv0Weight.Data, conv0Bias.Data, mergedH*mergedW, patchWidth, r.spec.ConvIntermediate)
	for index, value := range conv0 {
		conv0[index] = geluTanh(value)
	}
	conv2Weight, conv2Bias, err := r.loadPair(ctx, "mm.2.weight", "mm.2.bias")
	if err != nil {
		return reference.Value{}, err
	}
	conv2 := linear(conv0, conv2Weight.Data, conv2Bias.Data, mergedH*mergedW, r.spec.ConvIntermediate, r.spec.ProjectorInput)
	newline, err := r.load(ctx, "v.image_newline")
	if err != nil {
		return reference.Value{}, err
	}
	withNewlines := make([]float32, mergedH*(mergedW+1)*r.spec.ProjectorInput)
	for y := 0; y < mergedH; y++ {
		destination := y * (mergedW + 1) * r.spec.ProjectorInput
		source := y * mergedW * r.spec.ProjectorInput
		copy(withNewlines[destination:], conv2[source:source+mergedW*r.spec.ProjectorInput])
		copy(withNewlines[destination+mergedW*r.spec.ProjectorInput:], newline.Data)
	}
	projectWeight, projectBias, err := r.loadPair(ctx, "mm.model.fc.weight", "mm.model.fc.bias")
	if err != nil {
		return reference.Value{}, err
	}
	contentRows := mergedH * (mergedW + 1)
	projected := linear(withNewlines, projectWeight.Data, projectBias.Data, contentRows, r.spec.ProjectorInput, r.spec.OutputHidden)
	begin, end, err := r.loadPair(ctx, "mm.image_begin", "mm.image_end")
	if err != nil {
		return reference.Value{}, err
	}
	output := make([]float32, (contentRows+2)*r.spec.OutputHidden)
	copy(output, begin.Data)
	copy(output[r.spec.OutputHidden:], projected)
	copy(output[(contentRows+1)*r.spec.OutputHidden:], end.Data)
	postNorm, err := r.load(ctx, "mm.post_norm.weight")
	if err != nil {
		return reference.Value{}, err
	}
	rmsNormInPlace(output, postNorm.Data, contentRows+2, r.spec.OutputHidden, r.spec.LayerNormEpsilon)
	return reference.NewValue(tensor.MustShape(uint64(r.spec.OutputHidden), uint64(contentRows+2)), output)
}

func rmsNormInPlace(values, weight []float32, rows, width int, epsilon float32) {
	parallelRows(rows, func(start, end int) {
		for row := start; row < end; row++ {
			current := values[row*width : (row+1)*width]
			squares := 0.0
			for _, value := range current {
				squares += float64(value) * float64(value)
			}
			inverse := 1 / math.Sqrt(squares/float64(width)+float64(epsilon))
			for channel := range current {
				current[channel] = float32(float64(current[channel])*inverse) * weight[channel]
			}
		}
	})
}

func (r *HunyuanVLRunner) optionalBias(ctx context.Context, name string) ([]float32, error) {
	if !hasTensor(r.file, name) {
		return nil, nil
	}
	value, err := r.load(ctx, name)
	return value.Data, err
}

func (r *HunyuanVLRunner) load(ctx context.Context, name string) (reference.Value, error) {
	info, ok := r.file.Tensor(name)
	if !ok {
		return reference.Value{}, fmt.Errorf("projector: tensor %q is unavailable", name)
	}
	return model.LoadHostTensor(ctx, r.file, info)
}

func (r *HunyuanVLRunner) loadPair(ctx context.Context, first, second string) (reference.Value, reference.Value, error) {
	a, err := r.load(ctx, first)
	if err != nil {
		return reference.Value{}, reference.Value{}, err
	}
	b, err := r.load(ctx, second)
	return a, b, err
}
