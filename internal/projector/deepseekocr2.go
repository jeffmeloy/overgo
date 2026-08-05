package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

const (
	deepSeekOCR2ProjectorType   = "deepseekocr2"
	deepSeekOCR2DefaultTileSize = 768
	deepSeekOCR2DefaultMaxTiles = 6
)

type DeepSeekOCR2Spec struct {
	DeepSeekOCRSpec
	KVHeads int
}

type DeepSeekOCR2Runner struct {
	file        *gguf.File
	spec        DeepSeekOCR2Spec
	samPosition reference.Value
	cuda        *projectorCUDA
}

type DeepSeekOCR2OpenOptions = OpenOptions

func OpenDeepSeekOCR2(path string) (*DeepSeekOCR2Runner, error) {
	return OpenDeepSeekOCR2WithOptions(path, DeepSeekOCR2OpenOptions{})
}

func OpenDeepSeekOCR2WithOptions(path string, options DeepSeekOCR2OpenOptions) (*DeepSeekOCR2Runner, error) {
	return openProjectorResource(path, func(file *gguf.File) (*DeepSeekOCR2Runner, error) {
		spec, err := ReadDeepSeekOCR2Spec(file)
		if err != nil {
			return nil, err
		}
		runner := &DeepSeekOCR2Runner{file: file, spec: spec}
		runner.samPosition, err = loadProjectorHostTensor(context.Background(), file, "v.sam.pos_embd.weight")
		if err != nil {
			return nil, err
		}
		if err := runner.validateGraphs(); err != nil {
			return nil, err
		}
		if options.CUDA {
			runner.cuda, err = openProjectorCUDA(context.Background(), file, spec.TensorNames, nil, options.DeviceOrdinal)
			if err != nil {
				return nil, fmt.Errorf("projector: initialize DeepSeek-OCR-2 CUDA: %w", err)
			}
		}
		return runner, nil
	})
}

func (r *DeepSeekOCR2Runner) Close() error {
	if r == nil {
		return nil
	}
	return closeProjectorResources(&r.file, &r.cuda)
}

func (r *DeepSeekOCR2Runner) Spec() DeepSeekOCR2Spec {
	if r == nil {
		return DeepSeekOCR2Spec{}
	}
	return r.spec
}

func ReadDeepSeekOCR2Spec(file *gguf.File) (DeepSeekOCR2Spec, error) {
	base, err := readDeepSeekOCRBaseSpec(
		file, deepSeekOCR2ProjectorType, deepSeekOCR2DefaultTileSize, deepSeekOCR2DefaultMaxTiles,
	)
	if err != nil {
		return DeepSeekOCR2Spec{}, err
	}
	kvHeads, err := metadataUint32(file, "clip.vision.attention.head_count_kv")
	if err != nil {
		return DeepSeekOCR2Spec{}, err
	}
	spec := DeepSeekOCR2Spec{DeepSeekOCRSpec: base, KVHeads: int(kvHeads)}
	if spec.KVHeads <= 0 || spec.Heads%spec.KVHeads != 0 {
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
		"mm.model.fc.weight", "mm.model.fc.bias", "v.view_seperator")
	for _, name := range []string{"v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias"} {
		if hasTensor(file, name) {
			names = append(names, name)
		}
	}
	for layer := 0; layer < spec.Layers; layer++ {
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
		if info.Dimensions == 0 || info.Dimensions > 4 {
			return fmt.Errorf("projector: tensor %q rank %d is invalid", name, info.Dimensions)
		}
	}
	grid := uint64(spec.ImageSize / spec.PatchSize)
	requiredShapes := map[string][]uint64{
		"v.sam.pos_embd.weight":        {uint64(spec.SAMHidden), grid, grid},
		"v.sam.patch_embd.weight":      {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.SAMHidden)},
		"v.sam.patch_embd.bias":        {uint64(spec.SAMHidden)},
		"v.resample_query_768.weight":  {uint64(spec.Hidden), uint64((spec.TileSize / spec.PatchSize / 4) * (spec.TileSize / spec.PatchSize / 4))},
		"v.resample_query_1024.weight": {uint64(spec.Hidden), uint64((spec.ImageSize / spec.PatchSize / 4) * (spec.ImageSize / spec.PatchSize / 4))},
		"mm.model.fc.weight":           {uint64(spec.Hidden), uint64(spec.OutputHidden)},
		"mm.model.fc.bias":             {uint64(spec.OutputHidden)},
	}
	if err := validateProjectorTensorShapes(file, requiredShapes); err != nil {
		return err
	}
	separator, _ := file.Tensor("v.view_seperator")
	elements := uint64(1)
	for dimension := range separator.Dimensions {
		elements *= separator.Shape[dimension]
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
		return reference.Value{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessDeepSeekOCRImage(source, r.spec.DeepSeekOCRSpec)
	if err != nil {
		return reference.Value{}, err
	}
	values := make([]reference.Value, len(input.Tiles))
	for index, tile := range input.Tiles {
		values[index], err = r.encodeTile(ctx, tile, index == len(input.Tiles)-1)
		if err != nil {
			return reference.Value{}, fmt.Errorf("projector: encode DeepSeek-OCR-2 tile %d: %w", index, err)
		}
	}
	return r.assemble(values, input.GridW, input.GridH)
}

func (r *DeepSeekOCR2Runner) encodeTile(ctx context.Context, source image.Image, overview bool) (reference.Value, error) {
	size := source.Bounds().Dx()
	if size <= 0 || source.Bounds().Dy() != size || size%r.spec.PatchSize != 0 {
		return reference.Value{}, errors.New("projector: DeepSeek-OCR-2 tile shape is invalid")
	}
	builder := tensor.NewBuilder()
	input := builder.Input("pixel_values", dtype.F32, tensor.MustShape(3, uint64(size), uint64(size)))
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
	patches := hidden.Shape.Dims[1] * hidden.Shape.Dims[2]
	hidden = builder.Reshape(hidden, hidden.Shape.Dims[0], patches)
	queryName := "v.resample_query_768.weight"
	if overview {
		queryName = "v.resample_query_1024.weight"
	}
	query := weight(queryName)
	hidden = builder.Concat(hidden, query, 1)
	sequence := 2 * patches
	addOptionalBias := func(value *tensor.Tensor, name string, width uint64) *tensor.Tensor {
		if hasTensor(r.file, name) {
			return builder.Add(value, builder.Reshape(weight(name), width, 1))
		}
		return value
	}
	norm := func(value *tensor.Tensor, prefix string) *tensor.Tensor {
		value = builder.WeightedRMSNorm(value, weight(prefix+".weight"), r.spec.LayerNormEpsilon)
		return addOptionalBias(value, prefix+".bias", uint64(r.spec.Hidden))
	}
	if hasTensor(r.file, "v.pre_ln.weight") {
		hidden = norm(hidden, "v.pre_ln")
	}
	positions := make([]uint32, sequence)
	for index := range positions {
		positions[index] = uint32(index)
	}
	headWidth := uint64(r.spec.Hidden / r.spec.Heads)
	kvWidth := headWidth * uint64(r.spec.KVHeads)
	for layer := 0; layer < r.spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		residual := hidden
		normalized := norm(hidden, prefix+"ln1")
		q := addOptionalBias(builder.MulMat(weight(prefix+"attn_q.weight"), normalized), prefix+"attn_q.bias", uint64(r.spec.Hidden))
		k := addOptionalBias(builder.MulMat(weight(prefix+"attn_k.weight"), normalized), prefix+"attn_k.bias", kvWidth)
		v := addOptionalBias(builder.MulMat(weight(prefix+"attn_v.weight"), normalized), prefix+"attn_v.bias", kvWidth)
		q = builder.Reshape(q, headWidth, uint64(r.spec.Heads), sequence)
		k = builder.Reshape(k, headWidth, uint64(r.spec.KVHeads), sequence)
		v = builder.Reshape(v, headWidth, uint64(r.spec.KVHeads), sequence)
		q = builder.RoPENeoX(q, positions, uint32(headWidth), 1_000_000)
		k = builder.RoPENeoX(k, positions, uint32(headWidth), 1_000_000)
		imageQ := builder.FlatSlice(q, 0, headWidth, uint64(r.spec.Heads), patches)
		imageK := builder.FlatSlice(k, 0, headWidth, uint64(r.spec.KVHeads), patches)
		imageV := builder.FlatSlice(v, 0, headWidth, uint64(r.spec.KVHeads), patches)
		imageAttention := builder.Attention(imageQ, imageK, imageV, float32(1/math.Sqrt(float64(headWidth))), false)
		queryOffset := headWidth * uint64(r.spec.Heads) * patches
		queryQ := builder.FlatSlice(q, queryOffset, headWidth, uint64(r.spec.Heads), patches)
		queryAttention := builder.AttentionWithOffset(queryQ, k, v, float32(1/math.Sqrt(float64(headWidth))), true, uint32(patches))
		attention := builder.Concat(imageAttention, queryAttention, 2)
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
	if hasTensor(r.file, "v.post_ln.weight") {
		hidden = norm(hidden, "v.post_ln")
	}
	hidden = builder.FlatSlice(hidden, uint64(r.spec.Hidden)*patches, uint64(r.spec.Hidden), patches)
	projected := builder.MulMat(weight("mm.model.fc.weight"), hidden)
	return builder.Add(projected, builder.Reshape(weight("mm.model.fc.bias"), uint64(r.spec.OutputHidden), 1))
}

func (r *DeepSeekOCR2Runner) assemble(values []reference.Value, gridW, gridH int) (reference.Value, error) {
	if len(values) != gridW*gridH+1 {
		return reference.Value{}, errors.New("projector: DeepSeek-OCR-2 tile grid is inconsistent")
	}
	separator, err := loadProjectorHostTensor(context.Background(), r.file, "v.view_seperator")
	if err != nil {
		return reference.Value{}, err
	}
	var output []float32
	for _, value := range values {
		if value.Shape.Rank != 2 || value.Shape.Dims[0] != uint64(r.spec.OutputHidden) {
			return reference.Value{}, errors.New("projector: DeepSeek-OCR-2 tile output is inconsistent")
		}
		output = append(output, value.Data...)
	}
	output = append(output, separator.Data...)
	return reference.Value{Shape: tensor.MustShape(uint64(r.spec.OutputHidden), uint64(len(output)/r.spec.OutputHidden)), Data: output}, nil
}

func (r *DeepSeekOCR2Runner) validateGraphs() error {
	for _, item := range []struct {
		size     int
		overview bool
	}{{r.spec.TileSize, false}, {r.spec.ImageSize, true}} {
		builder := tensor.NewBuilder()
		input := builder.Input("pixel_values", dtype.F32, tensor.MustShape(3, uint64(item.size), uint64(item.size)))
		hostFeeds := make(map[*tensor.Tensor]reference.Value)
		weight := func(name string) *tensor.Tensor {
			info, _ := r.file.Tensor(name)
			return builder.Input(name, dtype.F32, tensorInfoShape(info))
		}
		_ = r.buildGraph(builder, input, item.size, item.overview, weight, hostFeeds)
		if err := builder.Err(); err != nil {
			return fmt.Errorf("projector: validate DeepSeek-OCR-2 graph: %w", err)
		}
	}
	return nil
}

func (r *DeepSeekOCR2Runner) BuildImagePrompt(ctx context.Context, tokenizer ImageTokenizer, source image.Image, beforeImage, afterImage string, _ bool) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *DeepSeekOCR2Runner) BuildImagesPrompt(ctx context.Context, tokenizer ImageTokenizer, sources []image.Image, text []string, _ bool) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, false)
}

func (r *DeepSeekOCR2Runner) BuildImagesHistoryPrompt(ctx context.Context, tokenizer ImageTokenizer, sources []image.Image, text []string) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, true)
}

func (r *DeepSeekOCR2Runner) buildImagesPrompt(ctx context.Context, tokenizer ImageTokenizer, sources []image.Image, text []string, history bool) (MultimodalPrompt, error) {
	return executeImagePromptPlan(ctx, tokenizer, sources, text, imagePromptPlan{
		Family: "DeepSeek-OCR-2", Placeholder: DeepSeekOCRImagePad, PlaceholderLabel: "DeepSeek-OCR-2 placeholder",
		History: history, EmbeddingWidth: r.spec.OutputHidden,
		Render: func(text []string, items []imagePromptItem) string {
			return renderDelimitedImagePrompt(text, items, DeepSeekOCRImagePad, "", "\n")
		},
	}, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		value, err := r.EncodeImage(ctx, source)
		return imagePromptItem{Embeddings: value.Data, Count: int(value.Shape.Dims[1])}, err
	})
}
