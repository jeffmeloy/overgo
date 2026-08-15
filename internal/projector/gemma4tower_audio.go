package projector

import (
	"context"
	"fmt"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type Gemma4AudioTowerOutput struct {
	Embeddings reference.Value
	SoftTokens int
}

type Gemma4AudioTowerTrace struct {
	Stages map[string]reference.Value
}

func (r *Gemma4TowerRunner) EncodeAudioFeatures(
	ctx context.Context,
	features []float32,
	frames int,
	profile AudioProjectionProfile,
) (Gemma4AudioTowerOutput, error) {
	output, _, err := r.encodeAudioFeatures(ctx, features, frames, profile, false)
	return output, err
}

func (r *Gemma4TowerRunner) EncodeAudioFeaturesTrace(
	ctx context.Context,
	features []float32,
	frames int,
	profile AudioProjectionProfile,
) (Gemma4AudioTowerOutput, Gemma4AudioTowerTrace, error) {
	return r.encodeAudioFeatures(ctx, features, frames, profile, true)
}

func (r *Gemma4TowerRunner) encodeAudioFeatures(
	ctx context.Context,
	features []float32,
	frames int,
	profile AudioProjectionProfile,
	trace bool,
) (Gemma4AudioTowerOutput, Gemma4AudioTowerTrace, error) {
	if r == nil || r.file == nil {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, errRunnerClosed
	}
	spec := r.spec.Audio
	if frames <= 0 || len(features) != frames*spec.MelBins {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, fmt.Errorf(
			"projector: Gemma 4 audio features=%d, want %d x %d", len(features), frames, spec.MelBins)
	}
	if err := profile.ValidateIdentity(); err != nil {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, fmt.Errorf("projector: Gemma 4 audio profile: %w", err)
	}
	builder := tensor.NewBuilder()
	input := builder.Input("audio_features", dtype.F32, tensor.MustShape(1, uint64(spec.MelBins), uint64(frames)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	graph.hostFeeds[input] = reference.Value{Shape: input.Shape, Data: features}

	hidden, sequence := r.gemma4AudioSubsampleGraph(graph, input, frames)
	stageNames := []string{"subsample"}
	stages := []*tensor.Tensor{hidden}
	position, positionLength := gemma4AudioRelativePositions(spec, profile.AttentionRopeFreqBase)
	positionInput := builder.Input("audio_relative_positions", dtype.F32,
		tensor.MustShape(uint64(spec.Hidden), uint64(positionLength)))
	graph.hostFeeds[positionInput] = reference.Value{Shape: positionInput.Shape, Data: position}
	shift := gemma4AudioRelativeShiftGraph(graph, spec, positionLength)

	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("a.blk.%d.", layer)
		hidden = r.gemma4AudioFeedForwardGraph(graph, hidden, prefix+"ffn1_", spec)
		norm := builder.WeightedRMSNorm(hidden, graph.weight(prefix+"attn_norm.weight"), spec.RMSNormEpsilon)
		attention := r.gemma4AudioAttentionGraph(graph, norm, positionInput, shift, prefix, sequence, spec)
		attention = builder.WeightedRMSNorm(attention, graph.weight(prefix+"post_attention_norm.weight"), spec.RMSNormEpsilon)
		hidden = builder.Add(hidden, attention)
		hidden = r.gemma4AudioLightConvGraph(graph, hidden, prefix, sequence, spec)
		hidden = r.gemma4AudioFeedForwardGraph(graph, hidden, prefix+"ffn2_", spec)
		hidden = builder.WeightedRMSNorm(hidden, graph.weight(prefix+"out_norm.weight"), spec.RMSNormEpsilon)
		stageNames = append(stageNames, fmt.Sprintf("enc%d", layer))
		stages = append(stages, hidden)
	}

	projected := builder.Add(builder.MulMat(graph.weight("a.output_proj.weight"), hidden), graph.weight("a.output_proj.bias"))
	projected = builder.RMSNorm(projected, spec.RMSNormEpsilon)
	embeddings := builder.MulMat(graph.weight("mm.a.input_projection.weight"), projected)
	targets := []*tensor.Tensor{embeddings}
	if trace {
		for _, stage := range stages {
			targets = append(targets, builder.FlatSlice(stage, 0, stage.Shape.Dims[0], uint64(min(4, int(stage.Shape.Dims[1])))))
		}
		targets = append(targets, builder.FlatSlice(embeddings, 0, embeddings.Shape.Dims[0], uint64(min(4, sequence))))
	}
	results, err := graph.execute(targets...)
	if err != nil {
		return Gemma4AudioTowerOutput{}, Gemma4AudioTowerTrace{}, fmt.Errorf("projector: execute Gemma 4 audio tower: %w", err)
	}
	traced := Gemma4AudioTowerTrace{}
	if trace {
		traced.Stages = make(map[string]reference.Value, len(stageNames)+1)
		for index, name := range stageNames {
			traced.Stages[name] = results[targets[index+1]]
		}
		traced.Stages["soft_tokens"] = results[targets[len(targets)-1]]
	}
	return Gemma4AudioTowerOutput{Embeddings: results[embeddings], SoftTokens: sequence}, traced, nil
}

func (r *Gemma4TowerRunner) gemma4AudioSubsampleGraph(
	graph *projectorGraphRuntime,
	input *tensor.Tensor,
	frames int,
) (*tensor.Tensor, int) {
	spec := r.spec.Audio
	hidden := input
	width, height := spec.MelBins, frames
	for layer, channels := range spec.SubChannels {
		prefix := fmt.Sprintf("a.conv.%d.", layer)
		hidden = graph.builder.Conv2D(hidden, graph.weight(prefix+"weight"), nil, 2, 2, 1, 1, 1, 1, false)
		hidden = graph.builder.Multiply(
			graph.builder.LayerNorm(hidden, spec.RMSNormEpsilon), graph.weight(prefix+"norm.weight"),
		)
		hidden = graph.builder.ReLU(hidden)
		width, height = (width+1)/2, (height+1)/2
		if layer == len(spec.SubChannels)-1 {
			hidden = graph.builder.Reshape(hidden, uint64(channels*width), uint64(height))
		}
	}
	return graph.builder.MulMat(graph.weight("a.input_proj.weight"), hidden), height
}

func (r *Gemma4TowerRunner) gemma4AudioFeedForwardGraph(
	graph *projectorGraphRuntime,
	input *tensor.Tensor,
	prefix string,
	spec Gemma4AudioTowerSpec,
) *tensor.Tensor {
	norm := graph.builder.WeightedRMSNorm(input, graph.weight(prefix+"norm.weight"), spec.RMSNormEpsilon)
	up := r.gemma4TowerClippedLinearGraph(graph, norm, prefix+"up")
	down := r.gemma4TowerClippedLinearGraph(graph, graph.builder.SiLU(up), prefix+"down")
	down = graph.builder.WeightedRMSNorm(down, graph.weight(prefix+"post_norm.weight"), spec.RMSNormEpsilon)
	return graph.builder.Add(input, graph.builder.Scale(down, spec.ResidualWeight))
}

func (r *Gemma4TowerRunner) gemma4AudioLightConvGraph(
	graph *projectorGraphRuntime,
	input *tensor.Tensor,
	prefix string,
	sequence int,
	spec Gemma4AudioTowerSpec,
) *tensor.Tensor {
	norm := graph.builder.WeightedRMSNorm(input, graph.weight(prefix+"conv_pre_norm.weight"), spec.RMSNormEpsilon)
	started := r.gemma4TowerClippedLinearGraph(graph, norm, prefix+"conv_start")
	first := graph.builder.Reshape(
		graph.builder.GroupSlice(started, 0, uint64(spec.Hidden), 1, uint64(spec.Hidden)),
		uint64(spec.Hidden), uint64(sequence),
	)
	gate := graph.builder.Reshape(
		graph.builder.GroupSlice(started, uint64(spec.Hidden), uint64(spec.Hidden), 1, uint64(spec.Hidden)),
		uint64(spec.Hidden), uint64(sequence),
	)
	glu := graph.builder.Multiply(first, graph.builder.Sigmoid(gate))
	padding := spec.ConvKernel - 1
	padded := graph.builder.Concat(gemma4AudioZeros(graph, spec.Hidden, padding, "conv_padding", started.ID), glu, 1)
	bias := gemma4AudioZeros(graph, spec.Hidden, 1, "conv_bias", started.ID)
	bias = graph.builder.Reshape(bias, uint64(spec.Hidden))
	convolved := graph.builder.Conv1DSame(padded, graph.weight(prefix+"conv_dw.weight"), bias, true)
	convolved = graph.builder.FlatSlice(convolved, uint64((spec.ConvKernel/2)*spec.Hidden), uint64(spec.Hidden), uint64(sequence))
	convolved = graph.builder.WeightedRMSNorm(convolved, graph.weight(prefix+"conv_norm.weight"), spec.RMSNormEpsilon)
	convolved = graph.builder.SiLU(convolved)
	return graph.builder.Add(input, r.gemma4TowerClippedLinearGraph(graph, convolved, prefix+"conv_end"))
}

func gemma4AudioRelativePositions(spec Gemma4AudioTowerSpec, frequencyBase float32) ([]float32, int) {
	contextSize := spec.ChunkSize + spec.ContextLeft - 1 + spec.ContextRight
	length := contextSize/2 + 1
	half := spec.Hidden / 2
	logIncrement := math.Log(float64(frequencyBase)) / max(float64(half-1), 1)
	result := make([]float32, length*spec.Hidden)
	for position := range length {
		id := float64(contextSize/2 - position)
		for index := range half {
			angle := id * math.Exp(-float64(index)*logIncrement)
			result[position*spec.Hidden+index] = float32(math.Sin(angle))
			result[position*spec.Hidden+half+index] = float32(math.Cos(angle))
		}
	}
	return result, length
}

func gemma4AudioRelativeShiftGraph(
	graph *projectorGraphRuntime,
	spec Gemma4AudioTowerSpec,
	positionLength int,
) *tensor.Tensor {
	chunk := spec.ChunkSize
	contextSize := chunk + spec.ContextLeft - 1 + spec.ContextRight
	inputElements := chunk * positionLength
	outputElements := chunk * contextSize
	values := make([]float32, inputElements*outputElements)
	for query := range chunk {
		for offset := range contextSize {
			flat := query*contextSize + offset
			row, column := flat/(contextSize+1), flat%(contextSize+1)
			if row < chunk && column < positionLength {
				values[flat*inputElements+row*positionLength+column] = 1
			}
		}
	}
	node := graph.builder.Input("audio_relative_shift", dtype.F32,
		tensor.MustShape(uint64(inputElements), uint64(outputElements)))
	graph.hostFeeds[node] = reference.Value{Shape: node.Shape, Data: values}
	return node
}

func (r *Gemma4TowerRunner) gemma4AudioAttentionGraph(
	graph *projectorGraphRuntime,
	input, positions, shift *tensor.Tensor,
	prefix string,
	sequence int,
	spec Gemma4AudioTowerSpec,
) *tensor.Tensor {
	query := r.gemma4TowerClippedLinearGraph(graph, input, prefix+"attn_q")
	key := r.gemma4TowerClippedLinearGraph(graph, input, prefix+"attn_k")
	value := r.gemma4TowerClippedLinearGraph(graph, input, prefix+"attn_v")
	relative := graph.builder.MulMat(graph.weight(prefix+"attn_rel_k.weight"), positions)
	headWidth := spec.Hidden / spec.Heads
	perDim := graph.builder.Scale(graph.builder.Softplus(graph.weight(prefix+"attn_per_dim_scale.weight")),
		float32(math.Pow(float64(headWidth), -0.5)/math.Ln2))
	keyScale := float32(math.Log(1+math.E) / math.Ln2)
	contextSize := spec.ChunkSize + spec.ContextLeft - 1 + spec.ContextRight
	positionLength := contextSize/2 + 1
	blocks := (sequence + spec.ChunkSize - 1) / spec.ChunkSize
	var blockOutputs *tensor.Tensor
	for block := range blocks {
		start := block * spec.ChunkSize
		validQueries := min(spec.ChunkSize, sequence-start)
		var heads *tensor.Tensor
		for head := range spec.Heads {
			qHead := gemma4AudioHead(graph.builder, query, head, headWidth, sequence)
			qHead = graph.builder.Multiply(qHead, perDim)
			kHead := graph.builder.Scale(gemma4AudioHead(graph.builder, key, head, headWidth, sequence), keyScale)
			vHead := gemma4AudioHead(graph.builder, value, head, headWidth, sequence)
			rHead := gemma4AudioHead(graph.builder, relative, head, headWidth, positionLength)
			qBlock := graph.builder.FlatSlice(qHead, uint64(start*headWidth), uint64(headWidth), uint64(validQueries))
			if validQueries < spec.ChunkSize {
				qBlock = gemma4AudioPadRight(graph, qBlock, headWidth, spec.ChunkSize-validQueries, "query")
			}
			kContext := gemma4AudioContext(graph, kHead, headWidth, sequence, start-spec.ContextLeft+1, contextSize, "key")
			vContext := gemma4AudioContext(graph, vHead, headWidth, sequence, start-spec.ContextLeft+1, contextSize, "value")
			content := graph.builder.MulMat(kContext, qBlock)
			relativeLogits := graph.builder.MulMat(rHead, qBlock)
			relativeLogits = graph.builder.MulMat(shift,
				graph.builder.Reshape(relativeLogits, uint64(spec.ChunkSize*positionLength), 1))
			relativeLogits = graph.builder.Reshape(relativeLogits, uint64(contextSize), uint64(spec.ChunkSize))
			logits := graph.builder.Add(content, relativeLogits)
			logits = graph.builder.Scale(graph.builder.Tanh(graph.builder.Scale(logits, 1/spec.LogitSoftcap)), spec.LogitSoftcap)
			mask := gemma4AudioMask(graph, spec, sequence, start, block, head)
			probability := graph.builder.Softmax(graph.builder.Add(logits, mask))
			output := graph.builder.MulMat(graph.builder.Transpose2D(vContext), probability)
			if validQueries < spec.ChunkSize {
				output = graph.builder.FlatSlice(output, 0, uint64(headWidth), uint64(validQueries))
			}
			if heads == nil {
				heads = output
			} else {
				heads = graph.builder.Concat(heads, output, 0)
			}
		}
		if blockOutputs == nil {
			blockOutputs = heads
		} else {
			blockOutputs = graph.builder.Concat(blockOutputs, heads, 1)
		}
	}
	return r.gemma4TowerClippedLinearGraph(graph, blockOutputs, prefix+"attn_output")
}

func gemma4AudioHead(builder *tensor.Builder, input *tensor.Tensor, head, width, sequence int) *tensor.Tensor {
	return builder.Reshape(
		builder.GroupSlice(input, uint64(head*width), uint64(width), 1, uint64(width)),
		uint64(width), uint64(sequence),
	)
}

func gemma4AudioPadRight(
	graph *projectorGraphRuntime,
	input *tensor.Tensor,
	width, columns int,
	label string,
) *tensor.Tensor {
	if columns == 0 {
		return input
	}
	zeros := gemma4AudioZeros(graph, width, columns, label+"_padding", input.ID)
	return graph.builder.Concat(input, zeros, 1)
}

func gemma4AudioZeros(graph *projectorGraphRuntime, width, columns int, label string, id uint64) *tensor.Tensor {
	zeros := graph.builder.Input(fmt.Sprintf("audio_%s.%d", label, id), dtype.F32,
		tensor.MustShape(uint64(width), uint64(columns)))
	graph.hostFeeds[zeros] = reference.Value{Shape: zeros.Shape, Data: make([]float32, width*columns)}
	return zeros
}

func gemma4AudioContext(
	graph *projectorGraphRuntime,
	input *tensor.Tensor,
	width, sequence, start, length int,
	label string,
) *tensor.Tensor {
	prefix := max(0, -start)
	validStart := max(0, start)
	validEnd := min(sequence, start+length)
	valid := max(0, validEnd-validStart)
	suffix := length - prefix - valid
	var result *tensor.Tensor
	appendPart := func(part *tensor.Tensor) {
		if result == nil {
			result = part
		} else {
			result = graph.builder.Concat(result, part, 1)
		}
	}
	if prefix > 0 {
		zeros := gemma4AudioZeros(graph, width, prefix, label+"_prefix", input.ID)
		appendPart(zeros)
	}
	if valid > 0 {
		appendPart(graph.builder.FlatSlice(input, uint64(validStart*width), uint64(width), uint64(valid)))
	}
	if suffix > 0 {
		zeros := gemma4AudioZeros(graph, width, suffix, label+"_suffix", input.ID)
		appendPart(zeros)
	}
	return result
}

func gemma4AudioMask(
	graph *projectorGraphRuntime,
	spec Gemma4AudioTowerSpec,
	sequence, start, block, head int,
) *tensor.Tensor {
	contextSize := spec.ChunkSize + spec.ContextLeft - 1 + spec.ContextRight
	values := make([]float32, contextSize*spec.ChunkSize)
	for query := range spec.ChunkSize {
		globalQuery := start + query
		for offset := range contextSize {
			globalKey := start + offset - spec.ContextLeft + 1
			if globalQuery >= sequence || globalKey < 0 || globalKey >= sequence ||
				globalKey < globalQuery-spec.ContextLeft+1 || globalKey > globalQuery+spec.ContextRight {
				values[query*contextSize+offset] = -1e9
			}
		}
	}
	node := graph.builder.Input(fmt.Sprintf("audio_mask.%d.%d", block, head), dtype.F32,
		tensor.MustShape(uint64(contextSize), uint64(spec.ChunkSize)))
	graph.hostFeeds[node] = reference.Value{Shape: node.Shape, Data: values}
	return node
}
