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

const gemma4UVProjectorType = "gemma4uv"

type Gemma4Spec struct {
	TeacherPatch     int
	PoolKernel       int
	ModelPatch       int
	PatchWidth       int
	Hidden           int
	PositionCount    int
	MaxImageTokens   int
	LayerNormEpsilon float32
	RMSNormEpsilon   float32
}

type Gemma4Image struct {
	PixelValues []float32
	Positions   []int
	GridH       int
	GridW       int
}

type Gemma4Output struct {
	Embeddings reference.Value
	GridH      int
	GridW      int
}

type Gemma4VideoOutput struct {
	Embeddings     reference.Value
	Frames         int
	TokensPerFrame int
}

type Gemma4Runner struct {
	file *gguf.File
	spec Gemma4Spec
	cuda *gemma4CUDA
}

func OpenGemma4(path string) (*Gemma4Runner, error) {
	return OpenGemma4WithOptions(path, Gemma4OpenOptions{})
}

type Gemma4OpenOptions struct {
	CUDA          bool
	DeviceOrdinal int
}

func OpenGemma4WithOptions(path string, options Gemma4OpenOptions) (*Gemma4Runner, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(openErr error) (*Gemma4Runner, error) {
		_ = file.Close()
		return nil, openErr
	}
	spec, err := ReadGemma4Spec(file)
	if err != nil {
		return fail(err)
	}
	if err := validateGemma4Catalog(file, spec); err != nil {
		return fail(err)
	}
	runner := &Gemma4Runner{file: file, spec: spec}
	if options.CUDA {
		runner.cuda, err = openGemma4CUDA(context.Background(), file, options.DeviceOrdinal)
		if err != nil {
			return fail(fmt.Errorf("projector: initialize Gemma 4 CUDA: %w", err))
		}
	}
	return runner, nil
}

func (r *Gemma4Runner) Close() error {
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

func (r *Gemma4Runner) Spec() Gemma4Spec {
	if r == nil {
		return Gemma4Spec{}
	}
	return r.spec
}

func ReadGemma4Spec(file *gguf.File) (Gemma4Spec, error) {
	if file == nil {
		return Gemma4Spec{}, errors.New("projector: GGUF file is nil")
	}
	architecture, err := metadataString(file, "general.architecture")
	if err != nil {
		return Gemma4Spec{}, err
	}
	if architecture != "clip" {
		return Gemma4Spec{}, fmt.Errorf("projector: architecture %q is not clip", architecture)
	}
	projectorType, err := metadataString(file, "clip.vision.projector_type")
	if err != nil {
		return Gemma4Spec{}, err
	}
	if projectorType != gemma4UVProjectorType {
		return Gemma4Spec{}, fmt.Errorf("projector: type %q is not %s", projectorType, gemma4UVProjectorType)
	}
	hasVision, err := metadataBool(file, "clip.has_vision_encoder")
	if err != nil {
		return Gemma4Spec{}, err
	}
	if !hasVision {
		return Gemma4Spec{}, errors.New("projector: vision encoder is disabled")
	}
	teacherPatch, err := metadataUint32(file, "clip.vision.patch_size")
	if err != nil {
		return Gemma4Spec{}, err
	}
	hidden, err := metadataUint32(file, "clip.vision.projection_dim")
	if err != nil {
		return Gemma4Spec{}, err
	}
	rmsEpsilon, err := metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return Gemma4Spec{}, err
	}
	patch, ok := file.Tensor("v.patch_embd.weight")
	if !ok || patch.Dimensions != 2 {
		return Gemma4Spec{}, errors.New("projector: Gemma 4 patch tensor is unavailable or invalid")
	}
	position, ok := file.Tensor("v.position_embd.weight")
	if !ok || position.Dimensions != 3 {
		return Gemma4Spec{}, errors.New("projector: Gemma 4 position tensor is unavailable or invalid")
	}
	maxNativeInt := uint64(^uint(0) >> 1)
	if uint64(teacherPatch) > maxNativeInt || uint64(hidden) > maxNativeInt ||
		patch.Shape[0] > maxNativeInt || position.Shape[1] > maxNativeInt {
		return Gemma4Spec{}, errors.New("projector: Gemma 4 dimensions exceed native limits")
	}
	patchWidth := int(patch.Shape[0])
	modelPatchSquared := patchWidth / 3
	modelPatch := int(math.Sqrt(float64(modelPatchSquared)))
	if modelPatch*modelPatch*3 != patchWidth || int(teacherPatch) == 0 || modelPatch%int(teacherPatch) != 0 {
		return Gemma4Spec{}, fmt.Errorf("projector: invalid Gemma 4 patch width %d", patchWidth)
	}
	spec := Gemma4Spec{
		TeacherPatch: int(teacherPatch), PoolKernel: modelPatch / int(teacherPatch),
		ModelPatch: modelPatch, PatchWidth: patchWidth, Hidden: int(hidden),
		PositionCount: int(position.Shape[1]), MaxImageTokens: 280,
		LayerNormEpsilon: 1e-5, RMSNormEpsilon: rmsEpsilon,
	}
	if err := spec.validate(); err != nil {
		return Gemma4Spec{}, err
	}
	return spec, nil
}

func (s Gemma4Spec) validate() error {
	if s.TeacherPatch <= 0 || s.PoolKernel <= 0 || s.ModelPatch != s.TeacherPatch*s.PoolKernel ||
		s.PatchWidth != s.ModelPatch*s.ModelPatch*3 || s.Hidden <= 0 || s.PositionCount <= 0 ||
		s.MaxImageTokens <= 0 || s.LayerNormEpsilon <= 0 || s.RMSNormEpsilon <= 0 {
		return fmt.Errorf("projector: invalid Gemma 4 metadata: %+v", s)
	}
	return nil
}

func validateGemma4Catalog(file *gguf.File, spec Gemma4Spec) error {
	required := map[string][]uint64{
		"v.patch_embd.weight":        {uint64(spec.PatchWidth), uint64(spec.Hidden)},
		"v.patch_embd.bias":          {uint64(spec.Hidden)},
		"v.patch_norm.1.weight":      {uint64(spec.PatchWidth)},
		"v.patch_norm.1.bias":        {uint64(spec.PatchWidth)},
		"v.patch_norm.2.weight":      {uint64(spec.Hidden)},
		"v.patch_norm.2.bias":        {uint64(spec.Hidden)},
		"v.position_embd.weight":     {uint64(spec.Hidden), uint64(spec.PositionCount), 2},
		"v.patch_norm.3.weight":      {uint64(spec.Hidden)},
		"v.patch_norm.3.bias":        {uint64(spec.Hidden)},
		"mm.input_projection.weight": {uint64(spec.Hidden), uint64(spec.Hidden)},
	}
	return validateProjectorTensorShapes(file, required)
}

func PreprocessGemma4Image(source image.Image, spec Gemma4Spec) (Gemma4Image, error) {
	if source == nil {
		return Gemma4Image{}, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return Gemma4Image{}, err
	}
	bounds := source.Bounds()
	height, width := bounds.Dy(), bounds.Dx()
	resizedH, resizedW, err := gemma4ResizeTarget(height, width, spec)
	if err != nil {
		return Gemma4Image{}, err
	}
	resized := source
	if resizedH != height || resizedW != width {
		resized = resizeImageBicubic(source, resizedW, resizedH)
	}
	resizedBounds := resized.Bounds()
	gridH, gridW := resizedH/spec.ModelPatch, resizedW/spec.ModelPatch
	rows := gridH * gridW
	pixels := make([]float32, rows*spec.PatchWidth)
	positions := make([]int, rows*2)
	for gridY := 0; gridY < gridH; gridY++ {
		for gridX := 0; gridX < gridW; gridX++ {
			row := gridY*gridW + gridX
			positions[row*2], positions[row*2+1] = gridX, gridY
			destination := pixels[row*spec.PatchWidth:]
			for y := 0; y < spec.ModelPatch; y++ {
				for x := 0; x < spec.ModelPatch; x++ {
					r, g, b, _ := resized.At(resizedBounds.Min.X+gridX*spec.ModelPatch+x, resizedBounds.Min.Y+gridY*spec.ModelPatch+y).RGBA()
					base := (y*spec.ModelPatch + x) * 3
					destination[base] = float32(r>>8) / 255
					destination[base+1] = float32(g>>8) / 255
					destination[base+2] = float32(b>>8) / 255
				}
			}
		}
	}
	return Gemma4Image{PixelValues: pixels, Positions: positions, GridH: gridH, GridW: gridW}, nil
}

func gemma4ResizeTarget(height, width int, spec Gemma4Spec) (int, int, error) {
	if height <= 0 || width <= 0 {
		return 0, 0, fmt.Errorf("projector: invalid image geometry %dx%d", width, height)
	}
	align := spec.ModelPatch
	round := func(value float64) int { return int(math.Round(value/float64(align))) * align }
	floor := func(value float64) int { return int(math.Floor(value/float64(align))) * align }
	ceil := func(value float64) int { return int(math.Ceil(value/float64(align))) * align }
	resizedH := max(align, round(float64(height)))
	resizedW := max(align, round(float64(width)))
	maxPixels := spec.MaxImageTokens * align * align
	minPixels := min(40, spec.MaxImageTokens) * align * align
	if resizedH*resizedW > maxPixels {
		beta := math.Sqrt(float64(height*width) / float64(maxPixels))
		resizedH = max(align, floor(float64(height)/beta))
		resizedW = max(align, floor(float64(width)/beta))
	} else if resizedH*resizedW < minPixels {
		beta := math.Sqrt(float64(minPixels) / float64(height*width))
		resizedH = ceil(float64(height) * beta)
		resizedW = ceil(float64(width) * beta)
	}
	return resizedH, resizedW, nil
}

func (r *Gemma4Runner) EncodeImage(ctx context.Context, source image.Image) (Gemma4Output, error) {
	if r == nil || r.file == nil {
		return Gemma4Output{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessGemma4Image(source, r.spec)
	if err != nil {
		return Gemma4Output{}, err
	}
	return r.encode(ctx, input)
}

// EncodeVideoFrames: frame-major Gemma 4 projection with the 70-token frame budget.
func (r *Gemma4Runner) EncodeVideoFrames(ctx context.Context, frames []image.Image) (Gemma4VideoOutput, error) {
	if r == nil || r.file == nil {
		return Gemma4VideoOutput{}, errors.New("projector: runner is closed")
	}
	if len(frames) == 0 {
		return Gemma4VideoOutput{}, errors.New("projector: video has no frames")
	}
	videoSpec := r.spec
	videoSpec.MaxImageTokens = 70
	var combined Gemma4Image
	var gridH, gridW int
	for index, frame := range frames {
		input, err := PreprocessGemma4Image(frame, videoSpec)
		if err != nil {
			return Gemma4VideoOutput{}, fmt.Errorf("projector: preprocess video frame %d: %w", index, err)
		}
		if index == 0 {
			gridH, gridW = input.GridH, input.GridW
		} else if input.GridH != gridH || input.GridW != gridW {
			return Gemma4VideoOutput{}, errors.New("projector: Gemma 4 video frames produce inconsistent grids")
		}
		combined.PixelValues = append(combined.PixelValues, input.PixelValues...)
		combined.Positions = append(combined.Positions, input.Positions...)
	}
	combined.GridH, combined.GridW = gridH*len(frames), gridW
	output, err := r.encode(ctx, combined)
	if err != nil {
		return Gemma4VideoOutput{}, err
	}
	return Gemma4VideoOutput{
		Embeddings: output.Embeddings, Frames: len(frames), TokensPerFrame: gridH * gridW,
	}, nil
}

func (r *Gemma4Runner) encode(ctx context.Context, input Gemma4Image) (Gemma4Output, error) {
	return r.encodeWithTrace(ctx, input, nil)
}

type gemma4Trace func(string, []float32)

func (r *Gemma4Runner) encodeWithTrace(ctx context.Context, input Gemma4Image, trace gemma4Trace) (Gemma4Output, error) {
	if r.cuda != nil {
		return r.encodeCUDAWithTrace(ctx, input, trace)
	}
	rows := input.GridH * input.GridW
	if len(input.PixelValues) != rows*r.spec.PatchWidth || len(input.Positions) != rows*2 {
		return Gemma4Output{}, errors.New("projector: Gemma 4 input shape is inconsistent")
	}
	pixels := make([]float32, len(input.PixelValues))
	patchArea := r.spec.ModelPatch * r.spec.ModelPatch
	for row := 0; row < rows; row++ {
		source := input.PixelValues[row*r.spec.PatchWidth:]
		destination := pixels[row*r.spec.PatchWidth:]
		for pixel := 0; pixel < patchArea; pixel++ {
			for channel := 0; channel < 3; channel++ {
				destination[channel*patchArea+pixel] = source[pixel*3+channel]
			}
		}
	}
	bf16RoundSlice(pixels)
	ln1Weight, ln1Bias, err := r.loadPair(ctx, "v.patch_norm.1.weight", "v.patch_norm.1.bias")
	if err != nil {
		return Gemma4Output{}, err
	}
	ln1 := make([]float32, len(pixels))
	layerNorm(ln1, pixels, ln1Weight.Data, ln1Bias.Data, rows, r.spec.PatchWidth, r.spec.LayerNormEpsilon)
	bf16RoundSlice(ln1)
	traceGemma4(trace, "patch_ln1", ln1)
	patchWeight, patchBias, err := r.loadPair(ctx, "v.patch_embd.weight", "v.patch_embd.bias")
	if err != nil {
		return Gemma4Output{}, err
	}
	hidden := linear(ln1, patchWeight.Data, patchBias.Data, rows, r.spec.PatchWidth, r.spec.Hidden)
	bf16RoundSlice(hidden)
	traceGemma4(trace, "patch_dense", hidden)
	ln2Weight, ln2Bias, err := r.loadPair(ctx, "v.patch_norm.2.weight", "v.patch_norm.2.bias")
	if err != nil {
		return Gemma4Output{}, err
	}
	ln2 := make([]float32, len(hidden))
	layerNorm(ln2, hidden, ln2Weight.Data, ln2Bias.Data, rows, r.spec.Hidden, r.spec.LayerNormEpsilon)
	bf16RoundSlice(ln2)
	traceGemma4(trace, "patch_ln2", ln2)
	position, err := r.load(ctx, "v.position_embd.weight")
	if err != nil {
		return Gemma4Output{}, err
	}
	for row := 0; row < rows; row++ {
		x, y := input.Positions[row*2], input.Positions[row*2+1]
		if x < 0 || y < 0 || x >= r.spec.PositionCount || y >= r.spec.PositionCount {
			return Gemma4Output{}, fmt.Errorf("projector: position %d,%d exceeds table", x, y)
		}
		for channel := 0; channel < r.spec.Hidden; channel++ {
			xValue := position.Data[x*r.spec.Hidden+channel]
			yValue := position.Data[(r.spec.PositionCount+y)*r.spec.Hidden+channel]
			pos := bf16Round(xValue + yValue)
			ln2[row*r.spec.Hidden+channel] = bf16Round(ln2[row*r.spec.Hidden+channel] + pos)
		}
	}
	ln3Weight, ln3Bias, err := r.loadPair(ctx, "v.patch_norm.3.weight", "v.patch_norm.3.bias")
	if err != nil {
		return Gemma4Output{}, err
	}
	posNorm := make([]float32, len(ln2))
	layerNorm(posNorm, ln2, ln3Weight.Data, ln3Bias.Data, rows, r.spec.Hidden, r.spec.LayerNormEpsilon)
	bf16RoundSlice(posNorm)
	traceGemma4(trace, "pos_norm", posNorm)
	preProjection := make([]float32, len(posNorm))
	rmsNormNoWeight(preProjection, posNorm, rows, r.spec.Hidden, r.spec.RMSNormEpsilon)
	bf16RoundSlice(preProjection)
	traceGemma4(trace, "pre_projection_norm", preProjection)
	projection, err := r.load(ctx, "mm.input_projection.weight")
	if err != nil {
		return Gemma4Output{}, err
	}
	embeddings := linear(preProjection, projection.Data, nil, rows, r.spec.Hidden, r.spec.Hidden)
	bf16RoundSlice(embeddings)
	traceGemma4(trace, "embedding_projection", embeddings)
	value, err := reference.NewValue(tensor.MustShape(uint64(r.spec.Hidden), uint64(rows)), embeddings)
	if err != nil {
		return Gemma4Output{}, err
	}
	return Gemma4Output{
		Embeddings: value,
		GridH:      input.GridH, GridW: input.GridW,
	}, nil
}

func traceGemma4(trace gemma4Trace, name string, values []float32) {
	if trace != nil {
		trace(name, values)
	}
}

func (r *Gemma4Runner) load(ctx context.Context, name string) (reference.Value, error) {
	info, ok := r.file.Tensor(name)
	if !ok {
		return reference.Value{}, fmt.Errorf("projector: tensor %q is unavailable", name)
	}
	return model.LoadHostTensor(ctx, r.file, info)
}

func (r *Gemma4Runner) loadPair(ctx context.Context, first, second string) (reference.Value, reference.Value, error) {
	a, err := r.load(ctx, first)
	if err != nil {
		return reference.Value{}, reference.Value{}, err
	}
	b, err := r.load(ctx, second)
	return a, b, err
}

func rmsNormNoWeight(output, input []float32, rows, width int, epsilon float32) {
	parallelRows(rows, func(start, end int) {
		for row := start; row < end; row++ {
			source := input[row*width : (row+1)*width]
			destination := output[row*width : (row+1)*width]
			sumSquares := 0.0
			for _, value := range source {
				sumSquares += float64(value) * float64(value)
			}
			inverse := 1 / math.Sqrt(sumSquares/float64(width)+float64(epsilon))
			for channel, value := range source {
				destination[channel] = float32(float64(value) * inverse)
			}
		}
	})
}

func bf16Round(value float32) float32 {
	raw := math.Float32bits(value)
	if raw&0x7f800000 == 0x7f800000 {
		return value
	}
	raw += 0x7fff + ((raw >> 16) & 1)
	return math.Float32frombits(raw & 0xffff0000)
}

func bf16RoundSlice(values []float32) {
	for index, value := range values {
		values[index] = bf16Round(value)
	}
}
