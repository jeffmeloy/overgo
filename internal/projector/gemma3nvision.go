package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

const (
	gemma3nVisionProjectorType = "gemma3nv"
	gemma3nVisionNormEpsilon   = 1e-6
	Gemma3nImagePad            = "<image_soft_token>"
)

type gemma3nVisionBlockKind uint8

const (
	gemma3nEdgeBlock gemma3nVisionBlockKind = iota + 1
	gemma3nInvertedBlock
	gemma3nAttentionBlock
)

type gemma3nVisionBlock struct {
	Stage, Index int
	Kind         gemma3nVisionBlockKind
	Names        map[string]string
}

type Gemma3nVisionSpec struct {
	ImageSize    int
	PatchSize    int
	VisionHidden int
	OutputHidden int
	ImageMean    [3]float32
	ImageStd     [3]float32
	Blocks       []gemma3nVisionBlock
	StageEnds    []int
	TensorNames  []string
}

type Gemma3nVisionRunner struct {
	file *gguf.File
	spec Gemma3nVisionSpec
	cuda *projectorCUDA
}

type Gemma3nVisionOpenOptions struct {
	CUDA          bool
	DeviceOrdinal int
}

func OpenGemma3nVision(path string) (*Gemma3nVisionRunner, error) {
	return OpenGemma3nVisionWithOptions(path, Gemma3nVisionOpenOptions{})
}

func OpenGemma3nVisionWithOptions(path string, options Gemma3nVisionOpenOptions) (*Gemma3nVisionRunner, error) {
	return openProjectorResource(path, func(file *gguf.File) (*Gemma3nVisionRunner, error) {
		spec, err := ReadGemma3nVisionSpec(file)
		if err != nil {
			return nil, err
		}
		runner := &Gemma3nVisionRunner{file: file, spec: spec}
		if err := runner.validateGraph(); err != nil {
			return nil, err
		}
		if options.CUDA {
			runner.cuda, err = openProjectorCUDA(context.Background(), file, spec.TensorNames, nil, options.DeviceOrdinal)
			if err != nil {
				return nil, fmt.Errorf("projector: initialize Gemma 3n CUDA: %w", err)
			}
		}
		return runner, nil
	})
}

func (r *Gemma3nVisionRunner) Close() error {
	if r == nil {
		return nil
	}
	return closeProjectorResources(&r.file, &r.cuda)
}

func (r *Gemma3nVisionRunner) Spec() Gemma3nVisionSpec {
	if r == nil {
		return Gemma3nVisionSpec{}
	}
	return r.spec
}

func ReadGemma3nVisionSpec(file *gguf.File) (Gemma3nVisionSpec, error) {
	if file == nil {
		return Gemma3nVisionSpec{}, errors.New("projector: GGUF file is nil")
	}
	architecture, err := metadataString(file, "general.architecture")
	if err != nil {
		return Gemma3nVisionSpec{}, err
	}
	projectorType, err := metadataString(file, "clip.projector_type")
	if err != nil {
		return Gemma3nVisionSpec{}, err
	}
	if architecture != "clip" || projectorType != gemma3nVisionProjectorType {
		return Gemma3nVisionSpec{}, fmt.Errorf("projector: architecture/type %q/%q is not clip/%s", architecture, projectorType, gemma3nVisionProjectorType)
	}
	hasVision, err := metadataBool(file, "clip.has_vision_encoder")
	if err != nil || !hasVision {
		if err != nil {
			return Gemma3nVisionSpec{}, err
		}
		return Gemma3nVisionSpec{}, errors.New("projector: vision encoder is disabled")
	}
	values := make([]int, 4)
	for index, key := range []string{
		"clip.vision.image_size", "clip.vision.patch_size", "clip.vision.embedding_length", "clip.vision.projection_dim",
	} {
		value, valueErr := metadataUint32(file, key)
		if valueErr != nil {
			return Gemma3nVisionSpec{}, valueErr
		}
		values[index] = int(value)
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", 3)
	if err != nil {
		return Gemma3nVisionSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", 3)
	if err != nil {
		return Gemma3nVisionSpec{}, err
	}
	spec := Gemma3nVisionSpec{
		ImageSize: values[0], PatchSize: values[1], VisionHidden: values[2], OutputHidden: values[3],
	}
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	if spec.ImageSize <= 0 || spec.PatchSize <= 0 || spec.ImageSize/spec.PatchSize != 16 ||
		spec.VisionHidden <= 0 || spec.OutputHidden <= 0 {
		return Gemma3nVisionSpec{}, fmt.Errorf("projector: invalid Gemma 3n vision metadata: %+v", spec)
	}
	for channel := range spec.ImageStd {
		if spec.ImageStd[channel] <= 0 || !finite32(spec.ImageMean[channel]) || !finite32(spec.ImageStd[channel]) {
			return Gemma3nVisionSpec{}, fmt.Errorf("projector: invalid Gemma 3n normalization channel %d", channel)
		}
	}
	required := []string{
		"v.conv_stem.conv.weight", "v.conv_stem.bn.weight",
		"v.msfa.ffn.pw_exp.conv.weight", "v.msfa.ffn.pw_proj.conv.weight", "v.msfa.norm.weight",
		"mm.input_projection.weight", "mm.soft_emb_norm.weight",
	}
	for _, name := range []string{"v.msfa.ffn.pw_exp.bn.weight", "v.msfa.ffn.pw_proj.bn.weight"} {
		if hasTensor(file, name) {
			required = append(required, name)
		}
	}
	if hasTensor(file, "v.conv_stem.conv.bias") {
		required = append(required, "v.conv_stem.conv.bias")
	}
	for stage := 0; stage < 4; stage++ {
		found := 0
		for index := 0; ; index++ {
			prefix := fmt.Sprintf("v.blk.%d.%d.", stage, index)
			edge := hasTensor(file, prefix+"conv_exp.weight")
			inverted := hasTensor(file, prefix+"dw_start.conv.weight") || hasTensor(file, prefix+"pw_exp.conv.weight")
			attention := hasTensor(file, prefix+"attn.query.proj.weight")
			if !edge && !inverted && !attention {
				break
			}
			block := gemma3nVisionBlock{Stage: stage, Index: index, Names: make(map[string]string)}
			suffixes := []string{}
			switch {
			case edge:
				block.Kind = gemma3nEdgeBlock
				suffixes = []string{"conv_exp.weight", "conv_pwl.weight"}
				for _, suffix := range []string{"bn1.weight", "bn2.weight"} {
					if hasTensor(file, prefix+suffix) {
						suffixes = append(suffixes, suffix)
					}
				}
			case attention:
				block.Kind = gemma3nAttentionBlock
				suffixes = []string{"attn.query.proj.weight", "attn.key.proj.weight", "attn.value.proj.weight", "attn.output.proj.weight"}
				for _, suffix := range []string{
					"attn.key.down_conv.weight", "attn.key.norm.weight", "attn.value.down_conv.weight",
					"attn.value.norm.weight", "norm.weight", "layer_scale.gamma",
				} {
					if hasTensor(file, prefix+suffix) {
						suffixes = append(suffixes, suffix)
					}
				}
			default:
				block.Kind = gemma3nInvertedBlock
				for _, suffix := range []string{
					"dw_start.conv.weight", "dw_start.bn.weight", "pw_exp.conv.weight", "pw_exp.bn.weight",
					"dw_mid.conv.weight", "dw_mid.bn.weight", "pw_proj.conv.weight", "pw_proj.bn.weight", "layer_scale.gamma",
				} {
					if hasTensor(file, prefix+suffix) {
						suffixes = append(suffixes, suffix)
					}
				}
			}
			for _, suffix := range suffixes {
				name := prefix + suffix
				if !hasTensor(file, name) {
					return Gemma3nVisionSpec{}, fmt.Errorf("projector: missing tensor %q", name)
				}
				block.Names[suffix] = name
				required = append(required, name)
			}
			spec.Blocks = append(spec.Blocks, block)
			found++
		}
		if found > 0 {
			spec.StageEnds = append(spec.StageEnds, len(spec.Blocks)-1)
		}
	}
	if len(spec.Blocks) == 0 {
		return Gemma3nVisionSpec{}, errors.New("projector: Gemma 3n MobileNetV5 has no blocks")
	}
	slices.Sort(required)
	spec.TensorNames = slices.Compact(required)
	for _, name := range spec.TensorNames {
		info, ok := file.Tensor(name)
		if !ok || info.Dimensions == 0 || info.Dimensions > 4 {
			return Gemma3nVisionSpec{}, fmt.Errorf("projector: tensor %q is unavailable or invalid", name)
		}
	}
	return spec, nil
}

func PreprocessGemma3nVisionImage(source image.Image, spec Gemma3nVisionSpec) ([]float32, error) {
	if source == nil {
		return nil, errors.New("projector: image is nil")
	}
	bounds := source.Bounds()
	if bounds.Dx() <= 0 || bounds.Dy() <= 0 {
		return nil, errors.New("projector: image bounds are empty")
	}
	resized := resizeImageBicubic(source, spec.ImageSize, spec.ImageSize)
	pixels := make([]float32, 3*spec.ImageSize*spec.ImageSize)
	for y := 0; y < spec.ImageSize; y++ {
		for x := 0; x < spec.ImageSize; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			for channel, raw := range [3]uint32{r, g, b} {
				pixels[channel+3*(x+spec.ImageSize*y)] = (float32(raw>>8)/255 - spec.ImageMean[channel]) / spec.ImageStd[channel]
			}
		}
	}
	return pixels, nil
}

func (r *Gemma3nVisionRunner) EncodeImage(ctx context.Context, source image.Image) (reference.Value, error) {
	if r == nil || r.file == nil {
		return reference.Value{}, errors.New("projector: runner is closed")
	}
	pixels, err := PreprocessGemma3nVisionImage(source, r.spec)
	if err != nil {
		return reference.Value{}, err
	}
	builder := tensor.NewBuilder()
	input := builder.Input("pixel_values", dtype.F32, tensor.MustShape(3, uint64(r.spec.ImageSize), uint64(r.spec.ImageSize)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	graph.hostFeeds[input] = pixelsValue(input, pixels)
	output := r.buildGraph(builder, input, graph.weight, graph.hostFeeds)
	results, err := graph.execute(output)
	if err != nil {
		return reference.Value{}, fmt.Errorf("projector: execute Gemma 3n graph: %w", err)
	}
	return results[output], nil
}

func tensorInfoShape(info gguf.TensorInfo) tensor.Shape {
	dimensions := make([]uint64, info.Dimensions)
	copy(dimensions, info.Shape[:info.Dimensions])
	return tensor.MustShape(dimensions...)
}

func (r *Gemma3nVisionRunner) validateGraph() error {
	builder := tensor.NewBuilder()
	input := builder.Input("pixel_values", dtype.F32, tensor.MustShape(3, uint64(r.spec.ImageSize), uint64(r.spec.ImageSize)))
	weight := func(name string) *tensor.Tensor {
		info, ok := r.file.Tensor(name)
		if !ok {
			return nil
		}
		return builder.Input(name, dtype.F32, tensorInfoShape(info))
	}
	_ = r.buildGraph(builder, input, weight, make(map[*tensor.Tensor]reference.Value))
	if err := builder.Err(); err != nil {
		return fmt.Errorf("projector: invalid Gemma 3n catalog: %w", err)
	}
	return nil
}

func (r *Gemma3nVisionRunner) buildGraph(
	builder *tensor.Builder,
	input *tensor.Tensor,
	weight func(string) *tensor.Tensor,
	hostFeeds map[*tensor.Tensor]reference.Value,
) *tensor.Tensor {
	optional := func(name string) *tensor.Tensor {
		if !hasTensor(r.file, name) {
			return nil
		}
		return weight(name)
	}
	stemWeight := weight("v.conv_stem.conv.weight")
	stemBias := optional("v.conv_stem.conv.bias")
	if stemBias != nil {
		stemBias = builder.Reshape(stemBias, stemWeight.Shape.Dims[3])
	}
	cur := gemma3nConvSame(builder, input, stemWeight, stemBias, 2, false)
	cur = gemma3nSpatialNorm(builder, cur, weight("v.conv_stem.bn.weight"))
	cur = qwen3VLGELUTanh(builder, cur, hostFeeds)
	features := make([]*tensor.Tensor, 0, 2)
	for blockIndex, block := range r.spec.Blocks {
		stride := uint32(1)
		if blockIndex == 0 || slices.Contains(r.spec.StageEnds, blockIndex-1) {
			stride = 2
		}
		switch block.Kind {
		case gemma3nEdgeBlock:
			residual := cur
			cur = gemma3nConvSame(builder, cur, weight(block.Names["conv_exp.weight"]), nil, stride, false)
			if name := block.Names["bn1.weight"]; name != "" {
				cur = gemma3nSpatialNorm(builder, cur, weight(name))
			}
			cur = qwen3VLGELUTanh(builder, cur, hostFeeds)
			cur = builder.Conv2D(cur, weight(block.Names["conv_pwl.weight"]), nil, 1, 1, 0, 0, 0, 0, false)
			if name := block.Names["bn2.weight"]; name != "" {
				cur = gemma3nSpatialNorm(builder, cur, weight(name))
			}
			if residual.Shape.Equal(cur.Shape) {
				cur = builder.Add(cur, residual)
			}
		case gemma3nAttentionBlock:
			cur = r.buildAttentionGraph(builder, cur, block, weight)
		default:
			residual := cur
			if name := block.Names["dw_start.conv.weight"]; name != "" {
				cur = gemma3nConvSame(builder, cur, weight(name), nil, 1, true)
				if norm := block.Names["dw_start.bn.weight"]; norm != "" {
					cur = gemma3nSpatialNorm(builder, cur, weight(norm))
				}
			}
			if name := block.Names["pw_exp.conv.weight"]; name != "" {
				cur = builder.Conv2D(cur, weight(name), nil, 1, 1, 0, 0, 0, 0, false)
				if norm := block.Names["pw_exp.bn.weight"]; norm != "" {
					cur = gemma3nSpatialNorm(builder, cur, weight(norm))
				}
				cur = qwen3VLGELUTanh(builder, cur, hostFeeds)
			}
			if name := block.Names["dw_mid.conv.weight"]; name != "" {
				cur = gemma3nConvSame(builder, cur, weight(name), nil, stride, true)
				if norm := block.Names["dw_mid.bn.weight"]; norm != "" {
					cur = gemma3nSpatialNorm(builder, cur, weight(norm))
				}
				cur = qwen3VLGELUTanh(builder, cur, hostFeeds)
			}
			if name := block.Names["pw_proj.conv.weight"]; name != "" {
				cur = builder.Conv2D(cur, weight(name), nil, 1, 1, 0, 0, 0, 0, false)
				if norm := block.Names["pw_proj.bn.weight"]; norm != "" {
					cur = gemma3nSpatialNorm(builder, cur, weight(norm))
				}
			}
			if name := block.Names["layer_scale.gamma"]; name != "" {
				cur = gemma3nChannelScale(builder, cur, weight(name))
			}
			if residual.Shape.Equal(cur.Shape) {
				cur = builder.Add(cur, residual)
			}
		}
		if builder.Err() != nil {
			return nil
		}
		fusion := len(r.spec.StageEnds) >= 4 && (blockIndex == r.spec.StageEnds[2] || blockIndex == r.spec.StageEnds[3])
		fusion = fusion || len(r.spec.StageEnds) < 4 && blockIndex == len(r.spec.Blocks)-1
		if fusion {
			features = append(features, cur)
		}
	}
	if len(features) == 0 {
		return nil
	}
	targetW, targetH := int(features[0].Shape.Dims[1]), int(features[0].Shape.Dims[2])
	resized := make([]*tensor.Tensor, len(features))
	for index, feature := range features {
		resized[index] = gemma3nResizeNearest(builder, feature, targetW, targetH)
	}
	cur = resized[0]
	for _, feature := range resized[1:] {
		cur = builder.Concat(cur, feature, 0)
	}
	cur = builder.Conv2D(cur, weight("v.msfa.ffn.pw_exp.conv.weight"), nil, 1, 1, 0, 0, 0, 0, false)
	if hasTensor(r.file, "v.msfa.ffn.pw_exp.bn.weight") {
		cur = gemma3nSpatialNorm(builder, cur, weight("v.msfa.ffn.pw_exp.bn.weight"))
	}
	cur = qwen3VLGELUTanh(builder, cur, hostFeeds)
	cur = builder.Conv2D(cur, weight("v.msfa.ffn.pw_proj.conv.weight"), nil, 1, 1, 0, 0, 0, 0, false)
	if hasTensor(r.file, "v.msfa.ffn.pw_proj.bn.weight") {
		cur = gemma3nSpatialNorm(builder, cur, weight("v.msfa.ffn.pw_proj.bn.weight"))
	}
	if cur.Shape.Dims[1] > 16 || cur.Shape.Dims[2] > 16 {
		cur = gemma3nAveragePool(builder, cur, 16, 16)
	}
	cur = gemma3nSpatialNorm(builder, cur, weight("v.msfa.norm.weight"))
	channels := cur.Shape.Dims[0]
	cur = builder.Reshape(cur, channels, cur.Shape.Dims[1]*cur.Shape.Dims[2])
	cur = builder.Scale(cur, float32(math.Sqrt(float64(channels))))
	softNorm := builder.Reshape(weight("mm.soft_emb_norm.weight"), channels)
	cur = builder.WeightedRMSNorm(cur, softNorm, gemma3nVisionNormEpsilon)
	cur = builder.MulMat(weight("mm.input_projection.weight"), cur)
	return builder.RMSNorm(cur, gemma3nVisionNormEpsilon)
}

func (r *Gemma3nVisionRunner) buildAttentionGraph(
	builder *tensor.Builder,
	input *tensor.Tensor,
	block gemma3nVisionBlock,
	weight func(string) *tensor.Tensor,
) *tensor.Tensor {
	cur := input
	if name := block.Names["norm.weight"]; name != "" {
		cur = gemma3nSpatialNorm(builder, cur, weight(name))
	}
	q := builder.Conv2D(cur, weight(block.Names["attn.query.proj.weight"]), nil, 1, 1, 0, 0, 0, 0, false)
	buildKV := func(part string) *tensor.Tensor {
		value := cur
		if name := block.Names["attn."+part+".down_conv.weight"]; name != "" {
			value = gemma3nConvSame(builder, value, weight(name), nil, 2, true)
			if norm := block.Names["attn."+part+".norm.weight"]; norm != "" {
				value = gemma3nSpatialNorm(builder, value, weight(norm))
			}
		}
		return builder.Conv2D(value, weight(block.Names["attn."+part+".proj.weight"]), nil, 1, 1, 0, 0, 0, 0, false)
	}
	k, v := buildKV("key"), buildKV("value")
	headWidth := k.Shape.Dims[0]
	if headWidth == 0 || q.Shape.Dims[0]%headWidth != 0 {
		builder.Reshape(q, q.Shape.Dims[0]+1)
		return input
	}
	heads := q.Shape.Dims[0] / headWidth
	q = builder.Reshape(q, headWidth, heads, q.Shape.Dims[1]*q.Shape.Dims[2])
	k = builder.Reshape(k, headWidth, 1, k.Shape.Dims[1]*k.Shape.Dims[2])
	v = builder.Reshape(v, headWidth, 1, v.Shape.Dims[1]*v.Shape.Dims[2])
	attention := builder.Attention(q, k, v, float32(1/math.Sqrt(float64(headWidth))), false)
	if attention == nil {
		return input
	}
	attention = builder.Reshape(attention, headWidth*heads, input.Shape.Dims[1], input.Shape.Dims[2])
	attention = builder.Conv2D(attention, weight(block.Names["attn.output.proj.weight"]), nil, 1, 1, 0, 0, 0, 0, false)
	if attention == nil {
		return input
	}
	if name := block.Names["layer_scale.gamma"]; name != "" {
		attention = gemma3nChannelScale(builder, attention, weight(name))
	}
	if attention.Shape.Equal(input.Shape) {
		attention = builder.Add(attention, input)
	}
	return attention
}

func gemma3nConvSame(builder *tensor.Builder, input, weight, bias *tensor.Tensor, stride uint32, depthwise bool) *tensor.Tensor {
	kw, kh := uint32(weight.Shape.Dims[0]), uint32(weight.Shape.Dims[1])
	w, h := uint32(input.Shape.Dims[1]), uint32(input.Shape.Dims[2])
	outW, outH := (w+stride-1)/stride, (h+stride-1)/stride
	padW := max(int((outW-1)*stride+kw-w), 0)
	padH := max(int((outH-1)*stride+kh-h), 0)
	return builder.Conv2D(input, weight, bias, stride, stride,
		uint32(padW/2), uint32(padW-padW/2), uint32(padH/2), uint32(padH-padH/2), depthwise)
}

func gemma3nSpatialNorm(builder *tensor.Builder, input, weight *tensor.Tensor) *tensor.Tensor {
	channels, width, height := input.Shape.Dims[0], input.Shape.Dims[1], input.Shape.Dims[2]
	flat := builder.Reshape(input, channels, width*height)
	normWeight := builder.Reshape(weight, channels)
	return builder.Reshape(builder.WeightedRMSNorm(flat, normWeight, gemma3nVisionNormEpsilon), channels, width, height)
}

func gemma3nChannelScale(builder *tensor.Builder, input, weight *tensor.Tensor) *tensor.Tensor {
	return builder.Multiply(input, builder.Reshape(weight, input.Shape.Dims[0], 1, 1))
}

func gemma3nResizeNearest(builder *tensor.Builder, input *tensor.Tensor, width, height int) *tensor.Tensor {
	inputW, inputH := int(input.Shape.Dims[1]), int(input.Shape.Dims[2])
	if inputW == width && inputH == height {
		return input
	}
	indices := make([]uint32, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			indices[x+width*y] = uint32((x * inputW / width) + inputW*(y*inputH/height))
		}
	}
	flat := builder.Reshape(input, input.Shape.Dims[0], uint64(inputW*inputH))
	return builder.Reshape(builder.GetRows(flat, indices), input.Shape.Dims[0], uint64(width), uint64(height))
}

func gemma3nAveragePool(builder *tensor.Builder, input *tensor.Tensor, width, height int) *tensor.Tensor {
	inputW, inputH := int(input.Shape.Dims[1]), int(input.Shape.Dims[2])
	if inputW%width != 0 || inputH%height != 0 {
		builder.Reshape(input, 0)
		return nil
	}
	factorW, factorH := inputW/width, inputH/height
	flat := builder.Reshape(input, input.Shape.Dims[0], uint64(inputW*inputH))
	var result *tensor.Tensor
	for dy := 0; dy < factorH; dy++ {
		for dx := 0; dx < factorW; dx++ {
			indices := make([]uint32, width*height)
			for y := 0; y < height; y++ {
				for x := 0; x < width; x++ {
					indices[x+width*y] = uint32(x*factorW + dx + inputW*(y*factorH+dy))
				}
			}
			part := builder.GetRows(flat, indices)
			if result == nil {
				result = part
			} else {
				result = builder.Add(result, part)
			}
		}
	}
	result = builder.Scale(result, 1/float32(factorW*factorH))
	return builder.Reshape(result, input.Shape.Dims[0], uint64(width), uint64(height))
}

func (r *Gemma3nVisionRunner) BuildImagePrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizer, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *Gemma3nVisionRunner) BuildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	_ bool,
) (MultimodalPrompt, error) {
	if len(sources) == 0 || len(text) != len(sources)+1 {
		return MultimodalPrompt{}, errors.New("projector: Gemma 3n image/text sequence is inconsistent")
	}
	formatted := make([]string, len(text))
	copy(formatted, text)
	formatted[0] = "<bos><start_of_turn>user\n" + formatted[0]
	formatted[len(formatted)-1] += "<end_of_turn>\n<start_of_turn>model\n"
	return r.buildImagesPrompt(ctx, tokenizer, sources, formatted, false)
}

func (r *Gemma3nVisionRunner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	return r.buildImagesPrompt(ctx, tokenizer, sources, text, true)
}

func (r *Gemma3nVisionRunner) buildImagesPrompt(
	ctx context.Context,
	tokenizer ImageTokenizer,
	sources []image.Image,
	text []string,
	history bool,
) (MultimodalPrompt, error) {
	return executeImagePromptPlan(ctx, tokenizer, sources, text, imagePromptPlan{
		Family: "Gemma 3n", Placeholder: Gemma3nImagePad, PlaceholderLabel: "Gemma 3n image placeholder",
		History: history, EmbeddingWidth: r.spec.OutputHidden,
		Render: func(text []string, items []imagePromptItem) string {
			return renderDelimitedImagePrompt(text, items, Gemma3nImagePad, "<start_of_image>", "<end_of_image>")
		},
	}, func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		value, err := r.EncodeImage(ctx, source)
		return imagePromptItem{Embeddings: value.Data, Count: int(value.Shape.Dims[1])}, err
	})
}
