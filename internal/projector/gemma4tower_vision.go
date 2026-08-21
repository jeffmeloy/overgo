package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type Gemma4VisionTowerOutput struct {
	Embeddings reference.Value
	PatchCount int
	SoftTokens int
}

// Gemma4VisionTowerTrace: sampled leading rows; parity evidence only.
type Gemma4VisionTowerTrace struct {
	Stages map[string]reference.Value
}

type Gemma4VisionTowerInput struct {
	PixelValues []float32
	Positions   []int32
	GridH       int
	GridW       int
}

type Gemma4VisionTowerVideoOutput struct {
	Embeddings     reference.Value
	Frames         int
	TokensPerFrame int
}

func PreprocessVisionTowerImage(source image.Image, spec Gemma4VisionTowerSpec) (Gemma4VisionTowerInput, error) {
	if source == nil {
		return Gemma4VisionTowerInput{}, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return Gemma4VisionTowerInput{}, err
	}
	bounds := source.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	resizedH, resizedW, err := visionTowerResizeTarget(height, width, spec)
	if err != nil {
		return Gemma4VisionTowerInput{}, err
	}
	resized := source
	if resizedH != height || resizedW != width {
		resized = media.ResizeBicubic(source, resizedW, resizedH)
	}
	resizedBounds := resized.Bounds()
	gridW, gridH := resizedW/spec.PatchSize, resizedH/spec.PatchSize
	rows, rowsOK := checked.MulInt(gridW, gridH)
	poolArea, poolAreaOK := checked.MulInt(spec.PoolKernel, spec.PoolKernel)
	softTokens, poolOK := checked.DivExactInt(rows, poolArea)
	if !rowsOK || !poolAreaOK || !poolOK || softTokens > spec.MaxImageTokens {
		return Gemma4VisionTowerInput{}, fmt.Errorf("projector: Gemma 4 vision image=%dx%d exceeds token contract", width, height)
	}
	patchArea, patchAreaOK := checked.MulInt(spec.PatchSize, spec.PatchSize)
	patchWidth, patchWidthOK := checked.MulInt(patchArea, media.RGBChannels)
	pixelElements, pixelsOK := checked.MulInt(rows, patchWidth)
	positionElements, positionsOK := checked.MulInt(rows, tensor.PairedExtent)
	if !patchAreaOK || !patchWidthOK || !pixelsOK || !positionsOK {
		return Gemma4VisionTowerInput{}, errors.New("projector: Gemma 4 vision input storage exceeds native limits")
	}
	result := Gemma4VisionTowerInput{
		PixelValues: make([]float32, pixelElements),
		Positions:   make([]int32, positionElements),
		GridH:       gridH,
		GridW:       gridW,
	}
	for patchY := range gridH {
		for patchX := range gridW {
			row := patchY*gridW + patchX
			for y := range spec.PatchSize {
				for x := range spec.PatchSize {
					r, g, b, _ := resized.At(resizedBounds.Min.X+patchX*spec.PatchSize+x, resizedBounds.Min.Y+patchY*spec.PatchSize+y).RGBA()
					offset := row*patchWidth + (y*spec.PatchSize+x)*media.RGBChannels
					result.PixelValues[offset] = media.NormalizedRGBAChannel(r)
					result.PixelValues[offset+tensor.SingletonExtent] = media.NormalizedRGBAChannel(g)
					result.PixelValues[offset+tensor.PairedExtent] = media.NormalizedRGBAChannel(b)
				}
			}
			result.Positions[tensor.PairedExtent*row] = int32(patchX)
			result.Positions[tensor.PairedExtent*row+tensor.SingletonExtent] = int32(patchY)
		}
	}
	return result, nil
}

func visionTowerResizeTarget(height, width int, spec Gemma4VisionTowerSpec) (int, int, error) {
	if !checked.PositiveInts(height, width, spec.PatchSize, spec.PoolKernel, spec.MaxImageTokens) {
		return tensor.FirstOffset, tensor.FirstOffset, fmt.Errorf("projector: invalid vision tower image geometry %dx%d", width, height)
	}
	alignment := spec.PatchSize * spec.PoolKernel
	maxPatches := spec.MaxImageTokens * spec.PoolKernel * spec.PoolKernel
	targetPixels := float64(maxPatches * spec.PatchSize * spec.PatchSize)
	factor := math.Sqrt(targetPixels / float64(height*width))
	resizedH := int(math.Floor(factor*float64(height)/float64(alignment))) * alignment
	resizedW := int(math.Floor(factor*float64(width)/float64(alignment))) * alignment
	if resizedH == tensor.FirstOffset && resizedW == tensor.FirstOffset {
		return tensor.FirstOffset, tensor.FirstOffset, fmt.Errorf("projector: vision tower image %dx%d rounds to zero", width, height)
	}
	maxSide := spec.MaxImageTokens * alignment
	if resizedH == tensor.FirstOffset {
		resizedH = alignment
		resizedW = min(width/height*alignment, maxSide)
	} else if resizedW == tensor.FirstOffset {
		resizedW = alignment
		resizedH = min(height/width*alignment, maxSide)
	}
	if resizedH*resizedW > maxPatches*spec.PatchSize*spec.PatchSize {
		return tensor.FirstOffset, tensor.FirstOffset, errors.New("projector: vision tower resize exceeds patch budget")
	}
	return resizedH, resizedW, nil
}

func (r *Gemma4TowerRunner) EncodeVisionImage(
	ctx context.Context,
	source image.Image,
) (Gemma4VisionTowerOutput, error) {
	if r == nil || r.file == nil {
		return Gemma4VisionTowerOutput{}, errRunnerClosed
	}
	input, err := PreprocessVisionTowerImage(source, r.spec.Vision)
	if err != nil {
		return Gemma4VisionTowerOutput{}, err
	}
	return r.EncodeVisionPatches(ctx, input.PixelValues, input.Positions)
}

func (r *Gemma4TowerRunner) EncodeVisionFrames(
	ctx context.Context,
	frames []image.Image,
) (Gemma4VisionTowerVideoOutput, error) {
	if r == nil || r.file == nil {
		return Gemma4VisionTowerVideoOutput{}, errRunnerClosed
	}
	if len(frames) == tensor.FirstOffset {
		return Gemma4VisionTowerVideoOutput{}, errors.New("projector: video has no frames")
	}
	videoSpec := r.spec.Vision
	videoSpec.MaxImageTokens = videoSpec.MaxVideoTokens
	var combined []float32
	tokensPerFrame := tensor.FirstOffset
	for index, frame := range frames {
		input, err := PreprocessVisionTowerImage(frame, videoSpec)
		if err != nil {
			return Gemma4VisionTowerVideoOutput{}, fmt.Errorf("projector: preprocess Gemma 4 video frame %d: %w", index, err)
		}
		output, err := r.EncodeVisionPatches(ctx, input.PixelValues, input.Positions)
		if err != nil {
			return Gemma4VisionTowerVideoOutput{}, fmt.Errorf("projector: encode Gemma 4 video frame %d: %w", index, err)
		}
		if index == tensor.FirstOffset {
			tokensPerFrame = output.SoftTokens
		} else if output.SoftTokens != tokensPerFrame {
			return Gemma4VisionTowerVideoOutput{}, errors.New("projector: Gemma 4 video frames produce inconsistent token counts")
		}
		combined = append(combined, output.Embeddings.Data...)
	}
	value, err := reference.NewValue(
		tensor.MustShape(uint64(r.spec.Vision.ProjectionDim), uint64(tokensPerFrame*len(frames))), combined,
	)
	if err != nil {
		return Gemma4VisionTowerVideoOutput{}, err
	}
	return Gemma4VisionTowerVideoOutput{Embeddings: value, Frames: len(frames), TokensPerFrame: tokensPerFrame}, nil
}

func (r *Gemma4TowerRunner) EncodeVisionPatches(
	ctx context.Context,
	pixels []float32,
	positions []int32,
) (Gemma4VisionTowerOutput, error) {
	output, _, err := r.encodeVisionPatches(ctx, pixels, positions, false)
	return output, err
}

func (r *Gemma4TowerRunner) EncodeVisionPatchesTrace(
	ctx context.Context,
	pixels []float32,
	positions []int32,
) (Gemma4VisionTowerOutput, Gemma4VisionTowerTrace, error) {
	return r.encodeVisionPatches(ctx, pixels, positions, true)
}

func (r *Gemma4TowerRunner) encodeVisionPatches(
	ctx context.Context,
	pixels []float32,
	positions []int32,
	trace bool,
) (Gemma4VisionTowerOutput, Gemma4VisionTowerTrace, error) {
	if r == nil || r.file == nil {
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, errRunnerClosed
	}
	spec := r.spec.Vision
	patchArea, areaOK := checked.MulInt(spec.PatchSize, spec.PatchSize)
	patchWidth, widthOK := checked.MulInt(patchArea, media.RGBChannels)
	rows, rowsOK := checked.DivExactInt(len(pixels), patchWidth)
	if !areaOK || !widthOK || !rowsOK || !checked.PositiveInts(rows) {
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, fmt.Errorf(
			"projector: Gemma 4 vision pixels=%d, patch width=%d", len(pixels), patchWidth)
	}
	if err := validateRowStorage(rows, rowStorage{elements: len(positions), width: tensor.PairedExtent}); err != nil {
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, fmt.Errorf(
			"projector: Gemma 4 vision positions=%d: %w", len(positions), err)
	}
	for row := range rows {
		for axis := range tensor.PairedExtent {
			position := positions[tensor.PairedExtent*row+axis]
			if position >= int32(spec.PositionCount) {
				return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, fmt.Errorf(
					"projector: Gemma 4 vision position=%d exceeds table=%d", position, spec.PositionCount)
			}
		}
	}
	softTokens, pool, err := compileSpatialPool(positions, rows, spec.PoolKernel)
	if err != nil {
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, err
	}
	if softTokens > spec.MaxImageTokens {
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, fmt.Errorf(
			"projector: Gemma 4 vision soft tokens=%d exceed %d", softTokens, spec.MaxImageTokens)
	}

	scaled := media.AffineRGB(pixels, spec.InputScale, spec.InputBias)
	builder := tensor.NewBuilder()
	input := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	graph.hostFeeds[input] = reference.Value{Shape: input.Shape, Data: scaled}

	hidden := builder.MulMat(graph.weight(visionPatchWeightTensor), input)
	positionTable := builder.Reshape(
		graph.weight(visionPositionWeightTensor),
		uint64(spec.Hidden), uint64(tensor.PairedExtent*spec.PositionCount),
	)
	positionRows := compileSpatialPositionRows(positions, spec.PositionCount)
	hidden = builder.Add(hidden, builder.Add(
		builder.GetRows(positionTable, positionRows.x), builder.GetRows(positionTable, positionRows.offsetY),
	))

	stageNames := []string{"patch_embed"}
	stages := []*tensor.Tensor{hidden}
	headWidth := uint64(spec.HeadDim)
	heads := uint64(spec.Heads)
	kvHeads := uint64(spec.KVHeads)
	for layer := range spec.Layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := builder.WeightedRMSNorm(hidden, graph.weight(prefix+"attn_norm.weight"), spec.RMSNormEpsilon)
		q := clippedLinearGraph(graph, norm, prefix+"attn_q")
		k := clippedLinearGraph(graph, norm, prefix+"attn_k")
		v := clippedLinearGraph(graph, norm, prefix+"attn_v")
		q = builder.Reshape(q, headWidth, heads, uint64(rows))
		k = builder.Reshape(k, headWidth, kvHeads, uint64(rows))
		v = builder.Reshape(v, headWidth, kvHeads, uint64(rows))
		q = builder.WeightedRMSNorm(q, graph.weight(prefix+"attn_q_norm.weight"), spec.RMSNormEpsilon)
		k = builder.WeightedRMSNorm(k, graph.weight(prefix+"attn_k_norm.weight"), spec.RMSNormEpsilon)
		v = builder.RMSNorm(v, spec.RMSNormEpsilon)
		q = spatialRoPE(builder, q, positionRows.x, positionRows.y, spec.RopeFreqBase)
		k = spatialRoPE(builder, k, positionRows.x, positionRows.y, spec.RopeFreqBase)
		attention := builder.AttentionWithOptions(q, k, v, tensor.AttentionOptions{Scale: tensor.SingletonExtent, Causal: false})
		attention = builder.Reshape(attention, uint64(spec.Heads*spec.HeadDim), uint64(rows))
		attention = clippedLinearGraph(graph, attention, prefix+"attn_output")
		attention = builder.WeightedRMSNorm(
			attention, graph.weight(prefix+"post_attention_norm.weight"), spec.RMSNormEpsilon,
		)
		hidden = builder.Add(hidden, attention)

		norm = builder.WeightedRMSNorm(hidden, graph.weight(prefix+"ffn_norm.weight"), spec.RMSNormEpsilon)
		gate := clippedLinearGraph(graph, norm, prefix+"ffn_gate")
		up := clippedLinearGraph(graph, norm, prefix+"ffn_up")
		activated := builder.Multiply(builder.GELUTanhExact(gate), up)
		down := clippedLinearGraph(graph, activated, prefix+"ffn_down")
		down = builder.WeightedRMSNorm(down, graph.weight(prefix+"post_ffw_norm.weight"), spec.RMSNormEpsilon)
		hidden = builder.Add(hidden, down)
		stageNames = append(stageNames, fmt.Sprintf("enc%d", layer))
		stages = append(stages, hidden)
	}

	poolInput := builder.Input("vision_pool", dtype.F32, tensor.MustShape(uint64(rows), uint64(softTokens)))
	graph.hostFeeds[poolInput] = reference.Value{Shape: poolInput.Shape, Data: pool}
	pooled := builder.Transpose2D(builder.MulMat(poolInput, builder.Transpose2D(hidden)))
	pooled = builder.Scale(pooled, float32(math.Sqrt(float64(spec.Hidden))))
	stageNames = append(stageNames, "pooler")
	stages = append(stages, pooled)
	normalized := builder.RMSNorm(pooled, spec.RMSNormEpsilon)
	embeddings := builder.MulMat(graph.weight(multimodalInputProjection), normalized)

	targets := []*tensor.Tensor{embeddings}
	if trace {
		for _, stage := range stages {
			targets = append(targets, builder.FlatSlice(stage, tensor.FirstOffset,
				stage.Shape.Dims[tensor.FirstOffset], uint64(min(4, int(stage.Shape.Dims[tensor.SingletonExtent])))))
		}
		targets = append(targets, builder.FlatSlice(embeddings, tensor.FirstOffset,
			embeddings.Shape.Dims[tensor.FirstOffset], uint64(min(4, softTokens))))
	}
	results, err := graph.execute(targets...)
	if err != nil {
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, fmt.Errorf(
			"projector: execute Gemma 4 vision tower: %w", err)
	}
	output := Gemma4VisionTowerOutput{
		Embeddings: results[embeddings], PatchCount: rows, SoftTokens: softTokens,
	}
	traced := Gemma4VisionTowerTrace{}
	if trace {
		traced.Stages = make(map[string]reference.Value, len(stageNames)+tensor.SingletonExtent)
		for index, name := range stageNames {
			traced.Stages[name] = results[targets[index+tensor.SingletonExtent]]
		}
		traced.Stages["soft_tokens"] = results[targets[len(targets)-tensor.SingletonExtent]]
	}
	return output, traced, nil
}

func clippedLinearGraph(
	graph *projectorGraphRuntime,
	input *tensor.Tensor,
	prefix string,
) *tensor.Tensor {
	minimum, maximum := projectorClipBounds(graph, prefix, "input")
	clamped := graph.builder.Clamp(input, minimum, maximum)
	output := graph.builder.MulMat(graph.weight(prefix+".weight"), clamped)
	minimum, maximum = projectorClipBounds(graph, prefix, "output")
	return graph.builder.Clamp(output, minimum, maximum)
}

func projectorClipBounds(
	graph *projectorGraphRuntime,
	prefix, side string,
) (minimum, maximum float32) {
	minimum, err := loadProjectorScalar(graph.ctx, graph.file, prefix+"."+side+"_min")
	if err != nil {
		graph.err = err
		return
	}
	maximum, err = loadProjectorScalar(graph.ctx, graph.file, prefix+"."+side+"_max")
	if err != nil {
		graph.err = err
		return
	}
	return minimum, maximum
}

func loadProjectorScalar(ctx context.Context, file *gguf.File, name string) (scalar float32, err error) {
	value, err := loadProjectorHostTensor(ctx, file, name)
	if err != nil {
		return scalar, err
	}
	if len(value.Data) != tensor.SingletonExtent || !checked.Finite32(value.Data[tensor.FirstOffset]) {
		return scalar, fmt.Errorf("projector: scalar tensor %q is invalid", name)
	}
	return value.Data[tensor.FirstOffset], nil
}

type spatialPositionRows struct {
	x       []uint32
	y       []uint32
	offsetY []uint32
}

func compileSpatialPositionRows(positions []int32, yOffset int) spatialPositionRows {
	rows := len(positions) / tensor.PairedExtent
	result := spatialPositionRows{x: make([]uint32, rows), y: make([]uint32, rows), offsetY: make([]uint32, rows)}
	for row := range rows {
		x := max(tensor.FirstOffset, int(positions[tensor.PairedExtent*row]))
		y := max(tensor.FirstOffset, int(positions[tensor.PairedExtent*row+tensor.SingletonExtent]))
		result.x[row], result.y[row], result.offsetY[row] = uint32(x), uint32(y), uint32(yOffset+y)
	}
	return result
}

func spatialRoPE(
	builder *tensor.Builder,
	input *tensor.Tensor,
	xPositions, yPositions []uint32,
	frequencyBase float32,
) *tensor.Tensor {
	headWidth := input.Shape.Dims[tensor.FirstOffset]
	axisWidth := headWidth / tensor.PairedExtent
	x := builder.GroupSlice(input, tensor.FirstOffset, axisWidth, tensor.SingletonExtent, axisWidth)
	y := builder.GroupSlice(input, axisWidth, axisWidth, tensor.SingletonExtent, axisWidth)
	x = builder.Reshape(x, axisWidth, input.Shape.Dims[tensor.SingletonExtent], input.Shape.Dims[tensor.PairedExtent])
	y = builder.Reshape(y, axisWidth, input.Shape.Dims[tensor.SingletonExtent], input.Shape.Dims[tensor.PairedExtent])
	x = builder.RoPEWithOptions(x, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: xPositions, RotaryDimensions: uint32(axisWidth), FrequencyBase: frequencyBase, FrequencyScale: tensor.SingletonExtent})
	y = builder.RoPEWithOptions(y, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: yPositions, RotaryDimensions: uint32(axisWidth), FrequencyBase: frequencyBase, FrequencyScale: tensor.SingletonExtent})
	return builder.Concat(x, y, tensor.FirstOffset)
}

func compileSpatialPool(positions []int32, rows, kernel int) (int, []float32, error) {
	if !checked.PositiveInts(kernel) || validateRowStorage(rows, rowStorage{elements: len(positions), width: tensor.PairedExtent}) != nil {
		return tensor.FirstOffset, nil, errors.New("projector: spatial pool input is invalid")
	}
	kernelArea, areaOK := checked.MulInt(kernel, kernel)
	softTokens, exact := checked.DivExactInt(rows, kernelArea)
	if !areaOK || !exact {
		return tensor.FirstOffset, nil, fmt.Errorf("projector: spatial pool rows=%d not divisible by area=%d", rows, kernelArea)
	}
	maxX := tensor.FirstOffset
	for row := range rows {
		maxX = max(maxX, int(positions[tensor.PairedExtent*row])+tensor.SingletonExtent)
	}
	blockWidth := maxX / kernel
	if !checked.PositiveInts(blockWidth) {
		return tensor.FirstOffset, nil, errors.New("projector: spatial pool width is zero")
	}
	elements, ok := checked.MulInt(rows, softTokens)
	if !ok {
		return tensor.FirstOffset, nil, errors.New("projector: spatial pool storage exceeds native limits")
	}
	weights := make([]float32, elements)
	for row := range rows {
		x := max(tensor.FirstOffset, int(positions[tensor.PairedExtent*row]))
		y := max(tensor.FirstOffset, int(positions[tensor.PairedExtent*row+tensor.SingletonExtent]))
		block := x/kernel + blockWidth*(y/kernel)
		if block < tensor.FirstOffset || block >= softTokens {
			return tensor.FirstOffset, nil, fmt.Errorf("projector: spatial pool block=%d exceeds %d", block, softTokens)
		}
		weights[block*rows+row] = float32(tensor.SingletonExtent) / float32(kernelArea)
	}
	return softTokens, weights, nil
}
