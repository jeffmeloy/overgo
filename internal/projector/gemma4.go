package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tensorcatalog"
)

const (
	gemma4UVProjectorType = "gemma4uv"
)

type Gemma4Spec struct {
	TeacherPatch     int
	PoolKernel       int
	ModelPatch       int
	PatchWidth       int
	Hidden           int
	PositionCount    int
	MaxImageTokens   int
	MaxVideoTokens   int
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
	projectorResources
	spec Gemma4Spec
}

func (r *Gemma4Runner) Spec() Gemma4Spec {
	if r == nil {
		return Gemma4Spec{}
	}
	return r.spec
}

func ReadGemma4Spec(file *gguf.File) (Gemma4Spec, error) {
	if err := validateVisionProjector(file, "clip.vision.projector_type", gemma4UVProjectorType); err != nil {
		return Gemma4Spec{}, err
	}
	teacherPatch, err := metadataUint32(file, "clip.vision.patch_size")
	if err != nil {
		return Gemma4Spec{}, err
	}
	hidden, err := metadataUint32(file, "clip.vision.projection_dim")
	if err != nil {
		return Gemma4Spec{}, err
	}
	rmsEpsilon, err := metadataFloat32(file, visionNormEpsilonKey)
	if err != nil {
		return Gemma4Spec{}, err
	}
	layerNormEpsilon, err := metadataFloat32(file, visionProjectorNormKey)
	if err != nil {
		return Gemma4Spec{}, err
	}
	declaredImageTokens, err := metadataUint32(file, visionMaxSoftTokensKey)
	if err != nil {
		return Gemma4Spec{}, err
	}
	videoTokens, err := metadataUint32(file, visionVideoSoftTokensKey)
	if err != nil {
		return Gemma4Spec{}, err
	}
	patch, ok := file.Tensor(visionPatchWeightTensor)
	if !ok {
		return Gemma4Spec{}, errors.New("projector: Gemma 4 patch tensor is unavailable")
	}
	if err := tensorcatalog.ValidateInfo(patch, tensorcatalog.Requirement{Rank: tensor.PairedExtent, NonEmpty: true}); err != nil {
		return Gemma4Spec{}, fmt.Errorf("projector: Gemma 4 patch tensor: %w", err)
	}
	position, ok := file.Tensor(visionPositionWeightTensor)
	if !ok {
		return Gemma4Spec{}, errors.New("projector: Gemma 4 position tensor is unavailable")
	}
	if err := tensorcatalog.ValidateInfo(position, tensorcatalog.Requirement{Rank: tensor.TripleExtent, NonEmpty: true}); err != nil {
		return Gemma4Spec{}, fmt.Errorf("projector: Gemma 4 position tensor: %w", err)
	}
	teacherPatchInt, teacherOK := checked.Int(uint64(teacherPatch))
	hiddenInt, hiddenOK := checked.Int(uint64(hidden))
	patchWidth, patchOK := checked.Int(patch.Shape[0])
	positionCount, positionOK := checked.Int(position.Shape[1])
	videoTokensInt, videoOK := checked.Int(uint64(videoTokens))
	if !teacherOK || !hiddenOK || !patchOK || !positionOK || !videoOK {
		return Gemma4Spec{}, errors.New("projector: Gemma 4 dimensions exceed native limits")
	}
	modelPatchArea, channelAligned := checked.DivExact64(patch.Shape[0], media.RGBChannels)
	modelPatch, square := tensor.SquareSideInt(modelPatchArea)
	poolKernel, teacherAligned := checked.DivExact64(uint64(modelPatch), uint64(teacherPatchInt))
	if !channelAligned || !square || !teacherAligned {
		return Gemma4Spec{}, fmt.Errorf("projector: invalid Gemma 4 patch width %d", patchWidth)
	}
	maxImageTokens, validTokenGrid := tensor.EqualPartition(
		position.Shape[1], poolKernel*poolKernel,
	)
	maxImageTokensInt, imageOK := checked.Int(maxImageTokens)
	if !validTokenGrid || !imageOK || uint64(declaredImageTokens) != maxImageTokens {
		return Gemma4Spec{}, errors.New("projector: Gemma 4 position grid is incompatible")
	}
	spec := Gemma4Spec{
		TeacherPatch: teacherPatchInt, PoolKernel: int(poolKernel),
		ModelPatch: modelPatch, PatchWidth: patchWidth, Hidden: hiddenInt,
		PositionCount: positionCount, MaxImageTokens: maxImageTokensInt, MaxVideoTokens: videoTokensInt,
		LayerNormEpsilon: layerNormEpsilon, RMSNormEpsilon: rmsEpsilon,
	}
	if err := spec.validate(); err != nil {
		return Gemma4Spec{}, err
	}
	return spec, nil
}

func (s Gemma4Spec) validate() error {
	if s.TeacherPatch <= 0 || s.PoolKernel <= 0 || s.ModelPatch != s.TeacherPatch*s.PoolKernel ||
		s.PatchWidth != s.ModelPatch*s.ModelPatch*media.RGBChannels || s.Hidden <= 0 || s.PositionCount <= 0 ||
		s.MaxImageTokens <= 0 || s.MaxVideoTokens <= 0 ||
		s.LayerNormEpsilon <= 0 || s.RMSNormEpsilon <= 0 {
		return fmt.Errorf("projector: invalid Gemma 4 metadata: %+v", s)
	}
	return nil
}

func validateGemma4Catalog(file *gguf.File, spec Gemma4Spec) ([]string, error) {
	required := map[string][]uint64{
		visionPatchWeightTensor:    {uint64(spec.PatchWidth), uint64(spec.Hidden)},
		visionPatchBiasTensor:      {uint64(spec.Hidden)},
		"v.patch_norm.1.weight":    {uint64(spec.PatchWidth)},
		"v.patch_norm.1.bias":      {uint64(spec.PatchWidth)},
		"v.patch_norm.2.weight":    {uint64(spec.Hidden)},
		"v.patch_norm.2.bias":      {uint64(spec.Hidden)},
		visionPositionWeightTensor: {uint64(spec.Hidden), uint64(spec.PositionCount), tensor.PairedExtent},
		"v.patch_norm.3.weight":    {uint64(spec.Hidden)},
		"v.patch_norm.3.bias":      {uint64(spec.Hidden)},
		multimodalInputProjection:  {uint64(spec.Hidden), uint64(spec.Hidden)},
	}
	return validateProjectorTensorCatalog(file, required)
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
		resized = media.ResizeBicubic(source, resizedW, resizedH)
	}
	resizedBounds := resized.Bounds()
	gridH, gridW := resizedH/spec.ModelPatch, resizedW/spec.ModelPatch
	rows := gridH * gridW
	pixels := make([]float32, rows*spec.PatchWidth)
	positions := make([]int, rows*tensor.PairedExtent)
	for gridY := 0; gridY < gridH; gridY++ {
		for gridX := 0; gridX < gridW; gridX++ {
			row := gridY*gridW + gridX
			positions[row*tensor.PairedExtent], positions[row*tensor.PairedExtent+1] = gridX, gridY
			destination := pixels[row*spec.PatchWidth:]
			for y := 0; y < spec.ModelPatch; y++ {
				for x := 0; x < spec.ModelPatch; x++ {
					r, g, b, _ := resized.At(resizedBounds.Min.X+gridX*spec.ModelPatch+x, resizedBounds.Min.Y+gridY*spec.ModelPatch+y).RGBA()
					base := (y*spec.ModelPatch + x) * media.RGBChannels
					destination[base] = media.NormalizedRGBAChannel(r)
					destination[base+1] = media.NormalizedRGBAChannel(g)
					destination[base+2] = media.NormalizedRGBAChannel(b)
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
	resizedH := max(align, round(float64(height)))
	resizedW := max(align, round(float64(width)))
	maxPixels := spec.MaxImageTokens * align * align
	if resizedH*resizedW > maxPixels {
		beta := math.Sqrt(float64(height*width) / float64(maxPixels))
		resizedH = max(align, floor(float64(height)/beta))
		resizedW = max(align, floor(float64(width)/beta))
	}
	return resizedH, resizedW, nil
}

func (r *Gemma4Runner) EncodeImage(ctx context.Context, source image.Image) (Gemma4Output, error) {
	if r == nil || r.file == nil {
		return Gemma4Output{}, errRunnerClosed
	}
	input, err := PreprocessGemma4Image(source, r.spec)
	if err != nil {
		return Gemma4Output{}, err
	}
	return r.encodeWithTrace(ctx, input, nil)
}

// EncodeVideoFrames: frame-major projection; bounded tokens per frame.
func (r *Gemma4Runner) EncodeVideoFrames(ctx context.Context, frames []image.Image) (Gemma4VideoOutput, error) {
	if r == nil || r.file == nil {
		return Gemma4VideoOutput{}, errRunnerClosed
	}
	if len(frames) == 0 {
		return Gemma4VideoOutput{}, errors.New("projector: video has no frames")
	}
	videoSpec := r.spec
	videoSpec.MaxImageTokens = r.spec.MaxVideoTokens
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
	output, err := r.encodeWithTrace(ctx, combined, nil)
	if err != nil {
		return Gemma4VideoOutput{}, err
	}
	return Gemma4VideoOutput{
		Embeddings: output.Embeddings, Frames: len(frames), TokensPerFrame: gridH * gridW,
	}, nil
}

type gemma4Trace func(string, []float32)

func (r *Gemma4Runner) encodeWithTrace(ctx context.Context, input Gemma4Image, trace gemma4Trace) (Gemma4Output, error) {
	if r.cuda != nil {
		return r.encodeCUDAWithTrace(ctx, input, trace)
	}
	rows, err := validateGridStorage(input.GridH, input.GridW,
		rowStorage{elements: len(input.PixelValues), width: r.spec.PatchWidth},
		rowStorage{elements: len(input.Positions), width: tensor.PairedExtent},
	)
	if err != nil {
		return Gemma4Output{}, fmt.Errorf("projector: Gemma 4 input: %w", err)
	}
	pixels, err := media.InterleavedToPlanarRows(input.PixelValues, rows, media.RGBChannels)
	if err != nil {
		return Gemma4Output{}, err
	}
	dtype.RoundBF16Slice(pixels)
	ln1Weight, ln1Bias, err := r.loadPair(ctx, "v.patch_norm.1.weight", "v.patch_norm.1.bias")
	if err != nil {
		return Gemma4Output{}, err
	}
	ln1 := make([]float32, len(pixels))
	hostmath.LayerNormF32AffineInto(ln1, pixels, ln1Weight.Data, ln1Bias.Data, rows, r.spec.PatchWidth, r.spec.LayerNormEpsilon)
	dtype.RoundBF16Slice(ln1)
	traceGemma4(trace, "patch_ln1", ln1)
	patchWeight, patchBias, err := r.loadPair(ctx, visionPatchWeightTensor, visionPatchBiasTensor)
	if err != nil {
		return Gemma4Output{}, err
	}
	hidden := hostmath.LinearF64BiasFirstNew(ln1, patchWeight.Data, patchBias.Data, rows, r.spec.PatchWidth, r.spec.Hidden)
	dtype.RoundBF16Slice(hidden)
	traceGemma4(trace, "patch_dense", hidden)
	ln2Weight, ln2Bias, err := r.loadPair(ctx, "v.patch_norm.2.weight", "v.patch_norm.2.bias")
	if err != nil {
		return Gemma4Output{}, err
	}
	ln2 := make([]float32, len(hidden))
	hostmath.LayerNormF32AffineInto(ln2, hidden, ln2Weight.Data, ln2Bias.Data, rows, r.spec.Hidden, r.spec.LayerNormEpsilon)
	dtype.RoundBF16Slice(ln2)
	traceGemma4(trace, "patch_ln2", ln2)
	position, err := r.load(ctx, visionPositionWeightTensor)
	if err != nil {
		return Gemma4Output{}, err
	}
	for row := range rows {
		x := input.Positions[row*tensor.PairedExtent]
		y := input.Positions[row*tensor.PairedExtent+tensor.SingletonExtent]
		if x < tensor.FirstOffset || y < tensor.FirstOffset || x >= r.spec.PositionCount || y >= r.spec.PositionCount {
			return Gemma4Output{}, fmt.Errorf("projector: position %d,%d exceeds table", x, y)
		}
		for channel := range r.spec.Hidden {
			xValue := position.Data[x*r.spec.Hidden+channel]
			yValue := position.Data[(r.spec.PositionCount+y)*r.spec.Hidden+channel]
			pos := dtype.RoundBF16(xValue + yValue)
			ln2[row*r.spec.Hidden+channel] = dtype.RoundBF16(ln2[row*r.spec.Hidden+channel] + pos)
		}
	}
	ln3Weight, ln3Bias, err := r.loadPair(ctx, "v.patch_norm.3.weight", "v.patch_norm.3.bias")
	if err != nil {
		return Gemma4Output{}, err
	}
	posNorm := make([]float32, len(ln2))
	hostmath.LayerNormF32AffineInto(posNorm, ln2, ln3Weight.Data, ln3Bias.Data, rows, r.spec.Hidden, r.spec.LayerNormEpsilon)
	dtype.RoundBF16Slice(posNorm)
	traceGemma4(trace, "pos_norm", posNorm)
	preProjection := make([]float32, len(posNorm))
	hostmath.RMSNormInto(preProjection, posNorm, nil, rows, r.spec.Hidden, float64(r.spec.RMSNormEpsilon))
	dtype.RoundBF16Slice(preProjection)
	traceGemma4(trace, "pre_projection_norm", preProjection)
	projection, err := r.load(ctx, multimodalInputProjection)
	if err != nil {
		return Gemma4Output{}, err
	}
	embeddings := hostmath.LinearF64BiasFirstNew(preProjection, projection.Data, nil, rows, r.spec.Hidden, r.spec.Hidden)
	dtype.RoundBF16Slice(embeddings)
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
