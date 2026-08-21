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
	"overgo/internal/tensorcatalog"
)

const (
	deepSeekOCR2ProjectorType = "deepseekocr2"
)

type DeepSeekOCR2Spec struct {
	DeepSeekOCRSpec
	KVHeads       int
	RopeFrequency float32
}

type DeepSeekOCR2Runner struct {
	projectorResources
	spec         DeepSeekOCR2Spec
	samPosition  reference.Value
	separator    reference.Value
	dynamicTiles bool
}

func openDeepSeekOCR2(ctx context.Context, file *gguf.File, options OpenOptions) (*DeepSeekOCR2Runner, error) {
	spec, err := ReadDeepSeekOCR2Spec(file)
	if err != nil {
		return nil, err
	}
	runner := &DeepSeekOCR2Runner{projectorResources: projectorResources{file: file}, spec: spec, dynamicTiles: !options.DisableDynamicTiles}
	runner.samPosition, err = loadProjectorHostTensor(ctx, file, "v.sam.pos_embd.weight")
	if err != nil {
		return nil, err
	}
	runner.separator, err = loadProjectorHostTensor(ctx, file, "v.view_seperator")
	if err != nil {
		return nil, err
	}
	if err := runner.validateGraphs(); err != nil {
		return nil, err
	}
	if options.CUDA {
		runner.cuda, err = openProjectorCUDA(ctx, file, spec.TensorNames, nil, options.DeviceOrdinal)
		if err != nil {
			return nil, fmt.Errorf("projector: initialize DeepSeek-OCR-2 CUDA: %w", err)
		}
	}
	return runner, nil
}

func (r *DeepSeekOCR2Runner) Spec() DeepSeekOCR2Spec {
	if r == nil {
		return DeepSeekOCR2Spec{}
	}
	return r.spec
}

func ReadDeepSeekOCR2Spec(file *gguf.File) (DeepSeekOCR2Spec, error) {
	base, err := readDeepSeekOCRBaseSpec(file, deepSeekOCR2ProjectorType)
	if err != nil {
		return DeepSeekOCR2Spec{}, err
	}
	kvHeads, err := metadataUint32(file, "clip.vision.attention.head_count_kv")
	if err != nil {
		return DeepSeekOCR2Spec{}, err
	}
	ropeFrequency, err := metadataFloat32(file, visionRopeFrequencyKey)
	if err != nil {
		return DeepSeekOCR2Spec{}, err
	}
	spec := DeepSeekOCR2Spec{DeepSeekOCRSpec: base, KVHeads: int(kvHeads), RopeFrequency: ropeFrequency}
	if spec.KVHeads <= 0 || spec.Heads%spec.KVHeads != 0 || !checked.PositiveFinite32(spec.RopeFrequency) {
		return DeepSeekOCR2Spec{}, fmt.Errorf("projector: invalid DeepSeek-OCR-2 KV head count %d", spec.KVHeads)
	}
	spec.TensorNames = deepSeekOCR2TensorNames(file, spec)
	if err := validateDeepSeekOCR2Catalog(file, spec); err != nil {
		return DeepSeekOCR2Spec{}, err
	}
	return spec, nil
}

func deepSeekOCR2TensorNames(file *gguf.File, spec DeepSeekOCR2Spec) []string {
	names := append(deepSeekOCRSAMTensorNames(spec.DeepSeekOCRSpec),
		"v.resample_query_768.weight", "v.resample_query_1024.weight",
		multimodalProjectionWeight, multimodalProjectionBias, "v.view_seperator")
	for _, name := range []string{visionPreNormWeightTensor, visionPreNormBiasTensor, visionPostNormWeightTensor, visionPostNormBiasTensor} {
		if hasTensor(file, name) {
			names = append(names, name)
		}
	}
	for layer := range spec.Layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for _, suffix := range []string{
			"ln1.weight", "ln2.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
			"attn_out.weight", "ffn_up.weight", "ffn_gate.weight", "ffn_down.weight",
		} {
			names = append(names, prefix+suffix)
		}
		for _, suffix := range []string{
			"ln1.bias", "ln2.bias", "attn_q.bias", "attn_k.bias", "attn_v.bias",
			"attn_out.bias", "ffn_up.bias", "ffn_gate.bias", "ffn_down.bias",
		} {
			if hasTensor(file, prefix+suffix) {
				names = append(names, prefix+suffix)
			}
		}
	}
	return names
}

func validateDeepSeekOCR2Catalog(file *gguf.File, spec DeepSeekOCR2Spec) error {
	for _, name := range spec.TensorNames {
		info, ok := file.Tensor(name)
		if !ok {
			return fmt.Errorf("projector: missing tensor %q", name)
		}
		if err := tensorcatalog.ValidateInfo(info, tensorcatalog.Requirement{
			Ranks: []uint32{tensor.SingletonExtent, tensor.PairedExtent, tensor.TripleExtent, tensor.MaxDimensions},
		}); err != nil {
			return fmt.Errorf("projector: tensor %q rank %d is invalid", name, info.Dimensions)
		}
	}
	grid := uint64(spec.ImageSize / spec.PatchSize)
	position, _ := file.Tensor("v.sam.pos_embd.weight")
	positionShape := []uint64{uint64(spec.SAMHidden), grid, grid}
	if position.Dimensions == tensor.MaxDimensions {
		positionShape = append(positionShape, tensor.SingletonExtent)
	}
	requiredShapes := map[string][]uint64{
		"v.sam.pos_embd.weight":   positionShape,
		"v.sam.patch_embd.weight": {uint64(spec.PatchSize), uint64(spec.PatchSize), media.RGBChannels, uint64(spec.SAMHidden)},
		"v.sam.patch_embd.bias":   {uint64(spec.SAMHidden)},
		"v.resample_query_768.weight": {uint64(spec.Hidden), uint64(
			(spec.TileSize / spec.PatchSize / (tensor.PairedExtent * tensor.PairedExtent)) *
				(spec.TileSize / spec.PatchSize / (tensor.PairedExtent * tensor.PairedExtent)),
		)},
		"v.resample_query_1024.weight": {uint64(spec.Hidden), uint64(
			(spec.ImageSize / spec.PatchSize / (tensor.PairedExtent * tensor.PairedExtent)) *
				(spec.ImageSize / spec.PatchSize / (tensor.PairedExtent * tensor.PairedExtent)),
		)},
		multimodalProjectionWeight: {uint64(spec.Hidden), uint64(spec.OutputHidden)},
		multimodalProjectionBias:   {uint64(spec.OutputHidden)},
	}
	if err := validateProjectorTensorShapes(file, requiredShapes); err != nil {
		return err
	}
	separator, _ := file.Tensor("v.view_seperator")
	elements, err := separator.ElementCount()
	if err != nil {
		return err
	}
	if elements != uint64(spec.OutputHidden) {
		return fmt.Errorf("projector: tensor %q has %d elements, want %d", "v.view_seperator", elements, spec.OutputHidden)
	}
	for _, prefix := range []string{"v.pre_ln", "v.post_ln"} {
		_, weight := file.Tensor(prefix + ".weight")
		_, bias := file.Tensor(prefix + ".bias")
		if bias && !weight {
			return fmt.Errorf("projector: tensor %q requires %q", prefix+".bias", prefix+".weight")
		}
	}
	return nil
}

func (r *DeepSeekOCR2Runner) EncodeImage(ctx context.Context, source image.Image) (reference.Value, error) {
	if r == nil || r.file == nil {
		return reference.Value{}, errRunnerClosed
	}
	input, err := preprocessDeepSeekOCRImage(source, r.spec.DeepSeekOCRSpec, r.dynamicTiles)
	if err != nil {
		return reference.Value{}, err
	}
	values := make([]reference.Value, len(input.Tiles))
	for index, tile := range input.Tiles {
		values[index], err = r.encodeTile(ctx, tile, index == len(input.Tiles)-tensor.SingletonExtent)
		if err != nil {
			return reference.Value{}, fmt.Errorf("projector: encode DeepSeek-OCR-2 tile %d: %w", index, err)
		}
	}
	return r.assemble(values, input.GridW, input.GridH)
}

func (r *DeepSeekOCR2Runner) encodeTile(ctx context.Context, source image.Image, overview bool) (reference.Value, error) {
	bounds := source.Bounds()
	size := bounds.Dx()
	_, aligned := checked.DivExactInt(size, r.spec.PatchSize)
	if bounds.Empty() || bounds.Dy() != size || !aligned {
		return reference.Value{}, errors.New("projector: DeepSeek-OCR-2 tile shape is invalid")
	}
	builder := tensor.NewBuilder()
	input := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(media.RGBChannels, uint64(size), uint64(size)))
	shared := DeepSeekOCRRunner{spec: r.spec.DeepSeekOCRSpec}
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	graph.hostFeeds[input] = pixelsValue(input, shared.tilePixels(source))
	output := r.buildGraph(builder, input, size, overview, graph.weight, graph.hostFeeds)
	results, err := graph.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *DeepSeekOCR2Runner) buildGraph(builder *tensor.Builder, input *tensor.Tensor, size int, overview bool, weight func(string) *tensor.Tensor, hostFeeds map[*tensor.Tensor]reference.Value) *tensor.Tensor {
	shared := DeepSeekOCRRunner{spec: r.spec.DeepSeekOCRSpec, samPosition: r.samPosition}
	hidden := shared.buildSAMGraph(builder, input, size, weight, hostFeeds)
	patches := hidden.Shape.Dims[tensor.SingletonExtent] * hidden.Shape.Dims[tensor.PairedExtent]
	hidden = builder.Reshape(hidden, hidden.Shape.Dims[tensor.FirstOffset], patches)
	queryName := "v.resample_query_768.weight"
	if overview {
		queryName = "v.resample_query_1024.weight"
	}
	query := weight(queryName)
	hidden = builder.Concat(hidden, query, tensor.SingletonExtent)
	sequence := tensor.PairedExtent * patches
	addOptionalBias := func(value *tensor.Tensor, name string, width uint64) *tensor.Tensor {
		if hasTensor(r.file, name) {
			return builder.Add(value, builder.Reshape(weight(name), width, tensor.SingletonExtent))
		}
		return value
	}
	norm := func(value *tensor.Tensor, prefix string) *tensor.Tensor {
		value = builder.WeightedRMSNorm(value, weight(prefix+".weight"), r.spec.LayerNormEpsilon)
		return addOptionalBias(value, prefix+".bias", uint64(r.spec.Hidden))
	}
	if hasTensor(r.file, visionPreNormWeightTensor) {
		hidden = norm(hidden, "v.pre_ln")
	}
	positions := make([]uint32, sequence)
	for index := range positions {
		positions[index] = uint32(index)
	}
	headWidth := uint64(r.spec.Hidden / r.spec.Heads)
	kvWidth := headWidth * uint64(r.spec.KVHeads)
	for layer := range r.spec.Layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		residual := hidden
		normalized := norm(hidden, prefix+"ln1")
		q := addOptionalBias(builder.MulMat(weight(prefix+"attn_q.weight"), normalized), prefix+"attn_q.bias", uint64(r.spec.Hidden))
		k := addOptionalBias(builder.MulMat(weight(prefix+"attn_k.weight"), normalized), prefix+"attn_k.bias", kvWidth)
		v := addOptionalBias(builder.MulMat(weight(prefix+"attn_v.weight"), normalized), prefix+"attn_v.bias", kvWidth)
		q = builder.Reshape(q, headWidth, uint64(r.spec.Heads), sequence)
		k = builder.Reshape(k, headWidth, uint64(r.spec.KVHeads), sequence)
		v = builder.Reshape(v, headWidth, uint64(r.spec.KVHeads), sequence)
		rope := tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: uint32(headWidth), FrequencyBase: r.spec.RopeFrequency, FrequencyScale: tensor.SingletonExtent}
		q = builder.RoPEWithOptions(q, rope)
		k = builder.RoPEWithOptions(k, rope)
		imageQ := builder.FlatSlice(q, tensor.FirstOffset, headWidth, uint64(r.spec.Heads), patches)
		imageK := builder.FlatSlice(k, tensor.FirstOffset, headWidth, uint64(r.spec.KVHeads), patches)
		imageV := builder.FlatSlice(v, tensor.FirstOffset, headWidth, uint64(r.spec.KVHeads), patches)
		imageAttention := builder.AttentionWithOptions(imageQ, imageK, imageV, tensor.AttentionOptions{Scale: float32(1 / math.Sqrt(float64(headWidth))), Causal: false})
		queryOffset := headWidth * uint64(r.spec.Heads) * patches
		queryQ := builder.FlatSlice(q, queryOffset, headWidth, uint64(r.spec.Heads), patches)
		queryAttention := builder.AttentionWithOptions(queryQ, k, v, tensor.AttentionOptions{Scale: float32(1 / math.Sqrt(float64(headWidth))), Causal: true, QueryStart: uint32(patches)})
		attention := builder.Concat(imageAttention, queryAttention, tensor.PairedExtent)
		attention = builder.Reshape(attention, uint64(r.spec.Hidden), sequence)
		attention = builder.MulMat(weight(prefix+"attn_out.weight"), attention)
		attention = addOptionalBias(attention, prefix+"attn_out.bias", uint64(r.spec.Hidden))
		hidden = builder.Add(residual, attention)
		residual = hidden
		normalized = norm(hidden, prefix+"ln2")
		gate := addOptionalBias(builder.MulMat(weight(prefix+"ffn_gate.weight"), normalized), prefix+"ffn_gate.bias", uint64(r.spec.FeedForward))
		up := addOptionalBias(builder.MulMat(weight(prefix+"ffn_up.weight"), normalized), prefix+"ffn_up.bias", uint64(r.spec.FeedForward))
		activated := builder.Multiply(builder.SiLU(gate), up)
		down := builder.MulMat(weight(prefix+"ffn_down.weight"), activated)
		down = addOptionalBias(down, prefix+"ffn_down.bias", uint64(r.spec.Hidden))
		hidden = builder.Add(residual, down)
	}
	if hasTensor(r.file, visionPostNormWeightTensor) {
		hidden = norm(hidden, "v.post_ln")
	}
	hidden = builder.FlatSlice(hidden, uint64(r.spec.Hidden)*patches, uint64(r.spec.Hidden), patches)
	projected := builder.MulMat(weight(multimodalProjectionWeight), hidden)
	return builder.Add(projected, builder.Reshape(weight(multimodalProjectionBias), uint64(r.spec.OutputHidden), tensor.SingletonExtent))
}

func (r *DeepSeekOCR2Runner) assemble(values []reference.Value, gridW, gridH int) (reference.Value, error) {
	if len(values) != gridW*gridH+tensor.SingletonExtent {
		return reference.Value{}, errors.New("projector: DeepSeek-OCR-2 tile grid is inconsistent")
	}
	elements := len(r.separator.Data)
	for _, value := range values {
		if !value.IsMatrixWidth(uint64(r.spec.OutputHidden)) {
			return reference.Value{}, errors.New("projector: DeepSeek-OCR-2 tile output is inconsistent")
		}
		elements += len(value.Data)
	}
	output := make([]float32, tensor.FirstOffset, elements)
	for _, value := range values {
		output = append(output, value.Data...)
	}
	output = append(output, r.separator.Data...)
	return reference.Value{Shape: tensor.MustShape(uint64(r.spec.OutputHidden), uint64(len(output)/r.spec.OutputHidden)), Data: output}, nil
}

func (r *DeepSeekOCR2Runner) validateGraphs() error {
	for _, item := range []struct {
		size     int
		overview bool
	}{{r.spec.TileSize, false}, {r.spec.ImageSize, true}} {
		builder := tensor.NewBuilder()
		input := builder.Input(
			visionInputTensor, dtype.F32,
			tensor.MustShape(media.RGBChannels, uint64(item.size), uint64(item.size)),
		)
		hostFeeds := make(map[*tensor.Tensor]reference.Value)
		weight := func(name string) *tensor.Tensor {
			info, _ := r.file.Tensor(name)
			return builder.Input(name, dtype.F32, tensor.MustShape(info.Extents()...))
		}
		_ = r.buildGraph(builder, input, item.size, item.overview, weight, hostFeeds)
		if err := builder.Err(); err != nil {
			return fmt.Errorf("projector: validate DeepSeek-OCR-2 graph: %w", err)
		}
	}
	return nil
}

func (r *DeepSeekOCR2Runner) imagesPrompt(ctx context.Context, tokenizer ImageTokenizer, sources []image.Image, text []string, _ PromptOptions) (MultimodalPrompt, error) {
	plan := delimitedImagePromptPlan("DeepSeek-OCR-2", DeepSeekOCRImagePad, "DeepSeek-OCR-2 placeholder", true, r.spec.OutputHidden, "", "")
	return executeImagePromptPlan(ctx, tokenizer, sources, text, plan, referenceImageEncoder(r.EncodeImage))
}
