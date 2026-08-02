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

const mimoVLProjectorType = "mimovl"

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
	cuda *mimoVLCUDA
}

type MiMoVLOpenOptions struct {
	CUDA          bool
	DeviceOrdinal int
}

func OpenMiMoVL(path string) (*MiMoVLRunner, error) {
	return OpenMiMoVLWithOptions(path, MiMoVLOpenOptions{})
}

func OpenMiMoVLWithOptions(path string, options MiMoVLOpenOptions) (*MiMoVLRunner, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(cause error) (*MiMoVLRunner, error) {
		_ = file.Close()
		return nil, cause
	}
	spec, err := ReadMiMoVLSpec(file)
	if err != nil {
		return fail(err)
	}
	if err := validateMiMoVLCatalog(file, spec); err != nil {
		return fail(err)
	}
	runner := &MiMoVLRunner{file: file, spec: spec}
	if options.CUDA {
		runner.cuda, err = openMiMoVLCUDA(context.Background(), file, spec, options.DeviceOrdinal)
		if err != nil {
			return fail(fmt.Errorf("projector: initialize MiMo-VL CUDA: %w", err))
		}
	}
	return runner, nil
}

func (r *MiMoVLRunner) Close() error {
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

func validateMiMoVLCatalog(file *gguf.File, spec MiMoVLSpec) error {
	qWidth := spec.Heads * spec.HeadDim
	kvWidth := spec.KVHeads * spec.HeadDim
	required := map[string][]uint64{
		"v.patch_embd.weight":   {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.patch_embd.weight.1": {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.post_ln.weight":      {uint64(spec.Hidden)},
		"mm.0.weight":           {uint64(spec.Hidden * 4), uint64(spec.MergerIntermediate)},
		"mm.2.weight":           {uint64(spec.MergerIntermediate), uint64(spec.ProjectionDim)},
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
	}
	if err := validateProjectorTensorShapes(file, required); err != nil {
		return err
	}
	for _, name := range []string{"v.post_ln.bias", "mm.0.bias", "mm.2.bias"} {
		if err := validateMiMoVLOptionalVector(file, name, map[string]int{
			"v.post_ln.bias": spec.Hidden, "mm.0.bias": spec.MergerIntermediate, "mm.2.bias": spec.ProjectionDim,
		}[name]); err != nil {
			return err
		}
	}
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for name, width := range map[string]int{
			"attn_out.bias": spec.Hidden, "ln1.bias": spec.Hidden, "ln2.bias": spec.Hidden,
		} {
			if err := validateMiMoVLOptionalVector(file, prefix+name, width); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateMiMoVLOptionalVector(file *gguf.File, name string, width int) error {
	info, ok := file.Tensor(name)
	if !ok {
		return nil
	}
	if info.Dimensions != 1 || info.Shape[0] != uint64(width) {
		return fmt.Errorf("projector: tensor %q shape %v, want [%d]", name, info.Shape[:info.Dimensions], width)
	}
	return nil
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
	if r.cuda != nil {
		return r.encodeCUDA(ctx, input)
	}
	rows := input.GridH * input.GridW
	if rows <= 0 || input.GridH%r.spec.MergeSize != 0 || input.GridW%r.spec.MergeSize != 0 {
		return MiMoVLOutput{}, errors.New("projector: MiMo-VL input geometry is inconsistent")
	}
	hidden, err := r.patchEmbedding(ctx, input)
	if err != nil {
		return MiMoVLOutput{}, err
	}
	rowPositions, columnPositions := mergedGrid(input.GridH, input.GridW, r.spec.MergeSize)
	positionsH, positionsW := rowPositions, columnPositions
	columnOrder := mimoVLColumnOrder(input.GridH/r.spec.MergeSize, input.GridW/r.spec.MergeSize, r.spec.MergeSize)
	inverseColumnOrder := inversePermutation(columnOrder)
	previousMode := -1
	for layer, mode := range r.spec.WindowModes {
		if err := ctx.Err(); err != nil {
			return MiMoVLOutput{}, err
		}
		if mode == 1 && previousMode != 1 {
			hidden = reorderRows(hidden, columnOrder, r.spec.Hidden)
			positionsH = reorderInts(rowPositions, columnOrder)
			positionsW = reorderInts(columnPositions, columnOrder)
		} else if mode != 1 && previousMode == 1 {
			hidden = reorderRows(hidden, inverseColumnOrder, r.spec.Hidden)
			positionsH, positionsW = rowPositions, columnPositions
		}
		if err := r.runLayer(ctx, hidden, rows, positionsH, positionsW, layer, mode); err != nil {
			return MiMoVLOutput{}, err
		}
		previousMode = mode
	}
	if previousMode == 1 {
		hidden = reorderRows(hidden, inverseColumnOrder, r.spec.Hidden)
	}
	hidden, err = r.layerNormalize(ctx, hidden, rows, "v.post_ln", 1e-6)
	if err != nil {
		return MiMoVLOutput{}, err
	}
	embeddings, err := r.merge(ctx, hidden, rows)
	if err != nil {
		return MiMoVLOutput{}, err
	}
	return MiMoVLOutput{Embeddings: embeddings, GridH: input.GridH, GridW: input.GridW, MergeSize: r.spec.MergeSize}, nil
}

func (r *MiMoVLRunner) patchEmbedding(ctx context.Context, input MiMoVLInput) ([]float32, error) {
	weight0, err := r.load(ctx, "v.patch_embd.weight")
	if err != nil {
		return nil, err
	}
	weight1, err := r.load(ctx, "v.patch_embd.weight.1")
	if err != nil {
		return nil, err
	}
	rows := input.GridH * input.GridW
	patchArea := r.spec.PatchSize * r.spec.PatchSize
	temporalWidth := 3 * patchArea
	if len(input.PixelValues) != rows*temporalWidth*2 {
		return nil, errors.New("projector: MiMo-VL pixel tensor is inconsistent")
	}
	output := make([]float32, rows*r.spec.Hidden)
	parallelRows(rows, func(start, end int) {
		for row := start; row < end; row++ {
			source := input.PixelValues[row*temporalWidth*2:]
			for channel := 0; channel < r.spec.Hidden; channel++ {
				accumulator := 0.0
				base := channel * temporalWidth
				for color := 0; color < 3; color++ {
					for pixel := 0; pixel < patchArea; pixel++ {
						source0 := color*2*patchArea + pixel
						weight := base + color*patchArea + pixel
						accumulator += float64(source[source0])*float64(weight0.Data[weight]) +
							float64(source[source0+patchArea])*float64(weight1.Data[weight])
					}
				}
				output[row*r.spec.Hidden+channel] = float32(accumulator)
			}
		}
	})
	return output, nil
}

func (r *MiMoVLRunner) runLayer(ctx context.Context, hidden []float32, rows int, positionsH, positionsW []int, layer, mode int) error {
	prefix := fmt.Sprintf("v.blk.%d.", layer)
	normalized, err := r.rmsNormalize(ctx, hidden, rows, prefix+"ln1")
	if err != nil {
		return err
	}
	qkvWeight, qkvBias, err := r.loadPair(ctx, prefix+"attn_qkv.weight", prefix+"attn_qkv.bias")
	if err != nil {
		return err
	}
	qkvWidth := (r.spec.Heads + 2*r.spec.KVHeads) * r.spec.HeadDim
	qkv := linear(normalized, qkvWeight.Data, qkvBias.Data, rows, r.spec.Hidden, qkvWidth)
	var sinks []float32
	if mode != -1 {
		value, loadErr := r.load(ctx, prefix+"attn_sinks")
		if loadErr != nil {
			return loadErr
		}
		sinks = value.Data
	}
	attention := mimoVLAttention(qkv, rows, r.spec, positionsH, positionsW, sinks, mode != -1)
	outWeight, err := r.load(ctx, prefix+"attn_out.weight")
	if err != nil {
		return err
	}
	outBias, err := r.optional(ctx, prefix+"attn_out.bias")
	if err != nil {
		return err
	}
	projected := linear(attention, outWeight.Data, outBias, rows, r.spec.Heads*r.spec.HeadDim, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += projected[index]
	}
	normalized, err = r.rmsNormalize(ctx, hidden, rows, prefix+"ln2")
	if err != nil {
		return err
	}
	upWeight, upBias, err := r.loadPair(ctx, prefix+"ffn_up.weight", prefix+"ffn_up.bias")
	if err != nil {
		return err
	}
	gateWeight, gateBias, err := r.loadPair(ctx, prefix+"ffn_gate.weight", prefix+"ffn_gate.bias")
	if err != nil {
		return err
	}
	up := linear(normalized, upWeight.Data, upBias.Data, rows, r.spec.Hidden, r.spec.Intermediate)
	gate := linear(normalized, gateWeight.Data, gateBias.Data, rows, r.spec.Hidden, r.spec.Intermediate)
	for index, value := range gate {
		gate[index] = value / (1 + float32(math.Exp(float64(-value)))) * up[index]
	}
	downWeight, downBias, err := r.loadPair(ctx, prefix+"ffn_down.weight", prefix+"ffn_down.bias")
	if err != nil {
		return err
	}
	down := linear(gate, downWeight.Data, downBias.Data, rows, r.spec.Intermediate, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += down[index]
	}
	return nil
}

func mimoVLAttention(qkv []float32, rows int, spec MiMoVLSpec, positionsH, positionsW []int, sinks []float32, windowed bool) []float32 {
	qWidth := spec.Heads * spec.HeadDim
	kvWidth := spec.KVHeads * spec.HeadDim
	rowWidth := qWidth + 2*kvWidth
	q := make([]float64, rows*qWidth)
	k := make([]float64, rows*kvWidth)
	half, quarter := spec.HeadDim/2, spec.HeadDim/4
	for row := 0; row < rows; row++ {
		for head := 0; head < spec.Heads; head++ {
			base := row*rowWidth + head*spec.HeadDim
			out := row*qWidth + head*spec.HeadDim
			mimoVLRotate(q[out:out+spec.HeadDim], qkv[base:base+spec.HeadDim], positionsH[row], positionsW[row], half, quarter)
		}
		for head := 0; head < spec.KVHeads; head++ {
			base := row*rowWidth + qWidth + head*spec.HeadDim
			out := row*kvWidth + head*spec.HeadDim
			mimoVLRotate(k[out:out+spec.HeadDim], qkv[base:base+spec.HeadDim], positionsH[row], positionsW[row], half, quarter)
		}
	}
	output := make([]float32, rows*qWidth)
	scale := 1 / math.Sqrt(float64(spec.HeadDim))
	group := spec.Heads / spec.KVHeads
	parallelRows(spec.Heads, func(start, end int) {
		scores := make([]float64, rows)
		for head := start; head < end; head++ {
			kvHead := head / group
			for queryRow := 0; queryRow < rows; queryRow++ {
				first, limit := 0, rows
				maximum := math.Inf(-1)
				if windowed {
					first = max(0, queryRow-spec.WindowSize)
					limit = min(rows, queryRow+spec.WindowSize+1)
					maximum = float64(sinks[head])
				}
				qOffset := queryRow*qWidth + head*spec.HeadDim
				for keyRow := first; keyRow < limit; keyRow++ {
					kOffset := keyRow*kvWidth + kvHead*spec.HeadDim
					dot := 0.0
					for channel := 0; channel < spec.HeadDim; channel++ {
						dot += q[qOffset+channel] * k[kOffset+channel]
					}
					scores[keyRow] = dot * scale
					maximum = max(maximum, scores[keyRow])
				}
				total := 0.0
				if windowed {
					total = math.Exp(float64(sinks[head]) - maximum)
				}
				for keyRow := first; keyRow < limit; keyRow++ {
					scores[keyRow] = math.Exp(scores[keyRow] - maximum)
					total += scores[keyRow]
				}
				for channel := 0; channel < spec.HeadDim; channel++ {
					value := 0.0
					for valueRow := first; valueRow < limit; valueRow++ {
						vOffset := valueRow*rowWidth + qWidth + kvWidth + kvHead*spec.HeadDim
						value += scores[valueRow] * float64(qkv[vOffset+channel])
					}
					output[queryRow*qWidth+head*spec.HeadDim+channel] = float32(value / total)
				}
			}
		}
	})
	return output
}

func mimoVLRotate(output []float64, input []float32, positionH, positionW, half, quarter int) {
	for pair := 0; pair < half; pair++ {
		position := positionH
		frequency := pair
		if pair >= quarter {
			position = positionW
			frequency -= quarter
		}
		angle := float64(position) / math.Pow(10000, float64(2*frequency)/float64(half))
		cosine, sine := math.Cos(angle), math.Sin(angle)
		first, second := float64(input[pair]), float64(input[pair+half])
		output[pair] = first*cosine - second*sine
		output[pair+half] = first*sine + second*cosine
	}
}

func (r *MiMoVLRunner) rmsNormalize(ctx context.Context, input []float32, rows int, prefix string) ([]float32, error) {
	weight, err := r.load(ctx, prefix+".weight")
	if err != nil {
		return nil, err
	}
	bias, err := r.optional(ctx, prefix+".bias")
	if err != nil {
		return nil, err
	}
	output := append([]float32(nil), input...)
	rmsNormInPlace(output, weight.Data, rows, r.spec.Hidden, r.spec.LayerNormEpsilon)
	if bias != nil {
		for row := 0; row < rows; row++ {
			for channel := 0; channel < r.spec.Hidden; channel++ {
				output[row*r.spec.Hidden+channel] += bias[channel]
			}
		}
	}
	return output, nil
}

func (r *MiMoVLRunner) layerNormalize(ctx context.Context, input []float32, rows int, prefix string, epsilon float32) ([]float32, error) {
	weight, err := r.load(ctx, prefix+".weight")
	if err != nil {
		return nil, err
	}
	bias, err := r.optional(ctx, prefix+".bias")
	if err != nil {
		return nil, err
	}
	if bias == nil {
		bias = make([]float32, r.spec.Hidden)
	}
	output := make([]float32, len(input))
	layerNorm(output, input, weight.Data, bias, rows, r.spec.Hidden, epsilon)
	return output, nil
}

func (r *MiMoVLRunner) merge(ctx context.Context, hidden []float32, rows int) (reference.Value, error) {
	mergedRows := rows / 4
	fc1Weight, err := r.load(ctx, "mm.0.weight")
	if err != nil {
		return reference.Value{}, err
	}
	fc1Bias, err := r.optional(ctx, "mm.0.bias")
	if err != nil {
		return reference.Value{}, err
	}
	fc1 := linear(hidden, fc1Weight.Data, fc1Bias, mergedRows, r.spec.Hidden*4, r.spec.MergerIntermediate)
	for index, value := range fc1 {
		fc1[index] = geluTanh(value)
	}
	fc2Weight, err := r.load(ctx, "mm.2.weight")
	if err != nil {
		return reference.Value{}, err
	}
	fc2Bias, err := r.optional(ctx, "mm.2.bias")
	if err != nil {
		return reference.Value{}, err
	}
	output := linear(fc1, fc2Weight.Data, fc2Bias, mergedRows, r.spec.MergerIntermediate, r.spec.ProjectionDim)
	return reference.NewValue(tensor.MustShape(uint64(r.spec.ProjectionDim), uint64(mergedRows)), output)
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

func (r *MiMoVLRunner) load(ctx context.Context, name string) (reference.Value, error) {
	info, ok := r.file.Tensor(name)
	if !ok {
		return reference.Value{}, fmt.Errorf("projector: tensor %q is unavailable", name)
	}
	return model.LoadHostTensor(ctx, r.file, info)
}

func (r *MiMoVLRunner) loadPair(ctx context.Context, first, second string) (reference.Value, reference.Value, error) {
	a, err := r.load(ctx, first)
	if err != nil {
		return reference.Value{}, reference.Value{}, err
	}
	b, err := r.load(ctx, second)
	return a, b, err
}

func (r *MiMoVLRunner) optional(ctx context.Context, name string) ([]float32, error) {
	if !hasTensor(r.file, name) {
		return nil, nil
	}
	value, err := r.load(ctx, name)
	return value.Data, err
}
