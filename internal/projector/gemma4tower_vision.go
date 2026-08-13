package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"overgo/internal/gguf"
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
}

func PreprocessGemma4VisionTowerImage(source image.Image, spec Gemma4VisionTowerSpec) (Gemma4VisionTowerInput, error) {
	if source == nil {
		return Gemma4VisionTowerInput{}, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return Gemma4VisionTowerInput{}, err
	}
	bounds := source.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if width <= 0 || height <= 0 || width%spec.PatchSize != 0 || height%spec.PatchSize != 0 {
		return Gemma4VisionTowerInput{}, fmt.Errorf(
			"projector: Gemma 4 vision image=%dx%d is not patch-aligned to %d", width, height, spec.PatchSize)
	}
	gridW, gridH := width/spec.PatchSize, height/spec.PatchSize
	rows := gridW * gridH
	if rows%(spec.PoolKernel*spec.PoolKernel) != 0 || rows/(spec.PoolKernel*spec.PoolKernel) > spec.MaxImageTokens {
		return Gemma4VisionTowerInput{}, fmt.Errorf("projector: Gemma 4 vision image=%dx%d exceeds token contract", width, height)
	}
	patchWidth := spec.PatchSize * spec.PatchSize * 3
	result := Gemma4VisionTowerInput{
		PixelValues: make([]float32, rows*patchWidth),
		Positions:   make([]int32, rows*2),
	}
	for patchY := range gridH {
		for patchX := range gridW {
			row := patchY*gridW + patchX
			for y := range spec.PatchSize {
				for x := range spec.PatchSize {
					r, g, b, _ := source.At(bounds.Min.X+patchX*spec.PatchSize+x, bounds.Min.Y+patchY*spec.PatchSize+y).RGBA()
					offset := row*patchWidth + (y*spec.PatchSize+x)*3
					result.PixelValues[offset] = normalizedImageChannel(r)
					result.PixelValues[offset+1] = normalizedImageChannel(g)
					result.PixelValues[offset+2] = normalizedImageChannel(b)
				}
			}
			result.Positions[2*row] = int32(patchX)
			result.Positions[2*row+1] = int32(patchY)
		}
	}
	return result, nil
}

func (r *Gemma4TowerRunner) EncodeVisionImage(
	ctx context.Context,
	source image.Image,
) (Gemma4VisionTowerOutput, error) {
	if r == nil || r.file == nil {
		return Gemma4VisionTowerOutput{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessGemma4VisionTowerImage(source, r.spec.Vision)
	if err != nil {
		return Gemma4VisionTowerOutput{}, err
	}
	return r.EncodeVisionPatches(ctx, input.PixelValues, input.Positions)
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
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, errors.New("projector: runner is closed")
	}
	spec := r.spec.Vision
	patchWidth := spec.PatchSize * spec.PatchSize * 3
	if len(pixels) == 0 || len(pixels)%patchWidth != 0 {
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, fmt.Errorf(
			"projector: Gemma 4 vision pixels=%d, patch width=%d", len(pixels), patchWidth)
	}
	rows := len(pixels) / patchWidth
	if len(positions) != 2*rows {
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, fmt.Errorf(
			"projector: Gemma 4 vision positions=%d, want %d", len(positions), 2*rows)
	}
	for row := range rows {
		for axis := range 2 {
			position := positions[2*row+axis]
			if position >= int32(spec.PositionCount) {
				return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, fmt.Errorf(
					"projector: Gemma 4 vision position=%d exceeds table=%d", position, spec.PositionCount)
			}
		}
	}
	softTokens, pool, err := gemma4VisionPoolPlan(positions, rows, spec.PoolKernel)
	if err != nil {
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, err
	}
	if softTokens > spec.MaxImageTokens {
		return Gemma4VisionTowerOutput{}, Gemma4VisionTowerTrace{}, fmt.Errorf(
			"projector: Gemma 4 vision soft tokens=%d exceed %d", softTokens, spec.MaxImageTokens)
	}

	scaled := make([]float32, len(pixels))
	for index, value := range pixels {
		scaled[index] = 2*value - 1
	}
	builder := tensor.NewBuilder()
	input := builder.Input("pixel_values", dtype.F32, tensor.MustShape(uint64(patchWidth), uint64(rows)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	graph.hostFeeds[input] = reference.Value{Shape: input.Shape, Data: scaled}

	hidden := builder.MulMat(graph.weight("v.patch_embd.weight"), input)
	positionTable := builder.Reshape(
		graph.weight("v.position_embd.weight"),
		uint64(spec.Hidden), uint64(2*spec.PositionCount),
	)
	xRows, yRows := gemma4VisionPositionRows(positions, spec.PositionCount)
	hidden = builder.Add(hidden, builder.Add(
		builder.GetRows(positionTable, xRows), builder.GetRows(positionTable, yRows),
	))

	stageNames := []string{"patch_embed"}
	stages := []*tensor.Tensor{hidden}
	headWidth := uint64(spec.HeadDim)
	heads := uint64(spec.Heads)
	kvHeads := uint64(spec.KVHeads)
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		norm := builder.WeightedRMSNorm(hidden, graph.weight(prefix+"attn_norm.weight"), spec.RMSNormEpsilon)
		q := r.gemma4TowerClippedLinearGraph(graph, norm, prefix+"attn_q")
		k := r.gemma4TowerClippedLinearGraph(graph, norm, prefix+"attn_k")
		v := r.gemma4TowerClippedLinearGraph(graph, norm, prefix+"attn_v")
		q = builder.Reshape(q, headWidth, heads, uint64(rows))
		k = builder.Reshape(k, headWidth, kvHeads, uint64(rows))
		v = builder.Reshape(v, headWidth, kvHeads, uint64(rows))
		q = builder.WeightedRMSNorm(q, graph.weight(prefix+"attn_q_norm.weight"), spec.RMSNormEpsilon)
		k = builder.WeightedRMSNorm(k, graph.weight(prefix+"attn_k_norm.weight"), spec.RMSNormEpsilon)
		v = builder.RMSNorm(v, spec.RMSNormEpsilon)
		q = gemma4VisionRoPE(builder, q, positions, spec.RopeFreqBase)
		k = gemma4VisionRoPE(builder, k, positions, spec.RopeFreqBase)
		attention := builder.Attention(q, k, v, 1, false)
		attention = builder.Reshape(attention, uint64(spec.Heads*spec.HeadDim), uint64(rows))
		attention = r.gemma4TowerClippedLinearGraph(graph, attention, prefix+"attn_output")
		attention = builder.WeightedRMSNorm(
			attention, graph.weight(prefix+"post_attention_norm.weight"), spec.RMSNormEpsilon,
		)
		hidden = builder.Add(hidden, attention)

		norm = builder.WeightedRMSNorm(hidden, graph.weight(prefix+"ffn_norm.weight"), spec.RMSNormEpsilon)
		gate := r.gemma4TowerClippedLinearGraph(graph, norm, prefix+"ffn_gate")
		up := r.gemma4TowerClippedLinearGraph(graph, norm, prefix+"ffn_up")
		activated := builder.Multiply(builder.GELUTanhExact(gate), up)
		down := r.gemma4TowerClippedLinearGraph(graph, activated, prefix+"ffn_down")
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
	embeddings := builder.MulMat(graph.weight("mm.input_projection.weight"), normalized)

	targets := []*tensor.Tensor{embeddings}
	if trace {
		for _, stage := range stages {
			targets = append(targets, builder.FlatSlice(stage, 0, stage.Shape.Dims[0], uint64(min(4, int(stage.Shape.Dims[1])))))
		}
		targets = append(targets, builder.FlatSlice(embeddings, 0, embeddings.Shape.Dims[0], uint64(min(4, softTokens))))
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
		traced.Stages = make(map[string]reference.Value, len(stageNames)+1)
		for index, name := range stageNames {
			traced.Stages[name] = results[targets[index+1]]
		}
		traced.Stages["soft_tokens"] = results[targets[len(targets)-1]]
	}
	return output, traced, nil
}

func (r *Gemma4TowerRunner) gemma4TowerClippedLinearGraph(
	graph *projectorGraphRuntime,
	input *tensor.Tensor,
	prefix string,
) *tensor.Tensor {
	minimum, maximum := r.gemma4TowerClipBounds(graph, prefix, "input")
	clamped := graph.builder.Clamp(input, minimum, maximum)
	output := graph.builder.MulMat(graph.weight(prefix+".weight"), clamped)
	minimum, maximum = r.gemma4TowerClipBounds(graph, prefix, "output")
	return graph.builder.Clamp(output, minimum, maximum)
}

func (r *Gemma4TowerRunner) gemma4TowerClipBounds(
	graph *projectorGraphRuntime,
	prefix, side string,
) (float32, float32) {
	minimum, err := loadProjectorScalar(graph.ctx, r.file, prefix+"."+side+"_min")
	if err != nil {
		graph.err = err
		return 0, 0
	}
	maximum, err := loadProjectorScalar(graph.ctx, r.file, prefix+"."+side+"_max")
	if err != nil {
		graph.err = err
		return 0, 0
	}
	return minimum, maximum
}

func loadProjectorScalar(ctx context.Context, file *gguf.File, name string) (float32, error) {
	value, err := loadProjectorHostTensor(ctx, file, name)
	if err != nil {
		return 0, err
	}
	if len(value.Data) != 1 || !finite32(value.Data[0]) {
		return 0, fmt.Errorf("projector: scalar tensor %q is invalid", name)
	}
	return value.Data[0], nil
}

func gemma4VisionPositionRows(positions []int32, count int) ([]uint32, []uint32) {
	xRows := make([]uint32, len(positions)/2)
	yRows := make([]uint32, len(positions)/2)
	for row := range xRows {
		x := max(0, int(positions[2*row]))
		y := max(0, int(positions[2*row+1]))
		xRows[row] = uint32(x)
		yRows[row] = uint32(count + y)
	}
	return xRows, yRows
}

func gemma4VisionRoPE(
	builder *tensor.Builder,
	input *tensor.Tensor,
	positions []int32,
	frequencyBase float32,
) *tensor.Tensor {
	headWidth := input.Shape.Dims[0]
	axisWidth := headWidth / 2
	xPositions := make([]uint32, len(positions)/2)
	yPositions := make([]uint32, len(positions)/2)
	for row := range xPositions {
		xPositions[row] = uint32(max(0, int(positions[2*row])))
		yPositions[row] = uint32(max(0, int(positions[2*row+1])))
	}
	x := builder.GroupSlice(input, 0, axisWidth, 1, axisWidth)
	y := builder.GroupSlice(input, axisWidth, axisWidth, 1, axisWidth)
	x = builder.Reshape(x, axisWidth, input.Shape.Dims[1], input.Shape.Dims[2])
	y = builder.Reshape(y, axisWidth, input.Shape.Dims[1], input.Shape.Dims[2])
	x = builder.RoPENeoX(x, xPositions, uint32(axisWidth), frequencyBase)
	y = builder.RoPENeoX(y, yPositions, uint32(axisWidth), frequencyBase)
	return builder.Concat(x, y, 0)
}

func gemma4VisionPoolPlan(positions []int32, rows, kernel int) (int, []float32, error) {
	if rows <= 0 || kernel <= 0 || len(positions) != 2*rows {
		return 0, nil, errors.New("projector: Gemma 4 vision pool input is invalid")
	}
	kernelArea := kernel * kernel
	if rows%kernelArea != 0 {
		return 0, nil, fmt.Errorf("projector: Gemma 4 vision patches=%d not divisible by pool area=%d", rows, kernelArea)
	}
	softTokens := rows / kernelArea
	maxX := 0
	for row := range rows {
		maxX = max(maxX, int(positions[2*row])+1)
	}
	blockWidth := maxX / kernel
	if blockWidth <= 0 {
		return 0, nil, errors.New("projector: Gemma 4 vision pool width is zero")
	}
	weights := make([]float32, rows*softTokens)
	for row := range rows {
		x := max(0, int(positions[2*row]))
		y := max(0, int(positions[2*row+1]))
		block := x/kernel + blockWidth*(y/kernel)
		if block < 0 || block >= softTokens {
			return 0, nil, fmt.Errorf("projector: Gemma 4 vision pool block=%d exceeds %d", block, softTokens)
		}
		weights[block*rows+row] = 1 / float32(kernelArea)
	}
	return softTokens, weights, nil
}
