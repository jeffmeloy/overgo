package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tensorcatalog"
)

const (
	gemma3nVisionProjectorType = "gemma3nv"
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
	NormEpsilon  float32
	Blocks       []gemma3nVisionBlock
	StageEnds    []int
	TensorNames  []string
}

type Gemma3nVisionRunner struct {
	projectorResources
	spec Gemma3nVisionSpec
}

func openGemma3nVision(ctx context.Context, file *gguf.File, options OpenOptions) (*Gemma3nVisionRunner, error) {
	spec, err := ReadGemma3nVisionSpec(file)
	if err != nil {
		return nil, err
	}
	runner := &Gemma3nVisionRunner{projectorResources: projectorResources{file: file}, spec: spec}
	if err := runner.validateGraph(); err != nil {
		return nil, err
	}
	if options.CUDA {
		runner.cuda, err = openProjectorCUDA(ctx, file, spec.TensorNames, nil, options.DeviceOrdinal)
		if err != nil {
			return nil, fmt.Errorf("projector: initialize Gemma 3n CUDA: %w", err)
		}
	}
	return runner, nil
}

func (r *Gemma3nVisionRunner) Spec() Gemma3nVisionSpec {
	if r == nil {
		return Gemma3nVisionSpec{}
	}
	return r.spec
}

func ReadGemma3nVisionSpec(file *gguf.File) (Gemma3nVisionSpec, error) {
	if err := validateVisionProjector(file, "clip.projector_type", gemma3nVisionProjectorType); err != nil {
		return Gemma3nVisionSpec{}, err
	}
	spec := Gemma3nVisionSpec{}
	if err := readMetadataIntFields(file,
		metadataIntField{"clip.vision.image_size", &spec.ImageSize},
		metadataIntField{"clip.vision.patch_size", &spec.PatchSize},
		metadataIntField{"clip.vision.embedding_length", &spec.VisionHidden},
		metadataIntField{"clip.vision.projection_dim", &spec.OutputHidden},
	); err != nil {
		return Gemma3nVisionSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", media.RGBChannels)
	if err != nil {
		return Gemma3nVisionSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", media.RGBChannels)
	if err != nil {
		return Gemma3nVisionSpec{}, err
	}
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	if err := readProjectionNorm(file, &spec.NormEpsilon); err != nil {
		return Gemma3nVisionSpec{}, err
	}
	if spec.ImageSize <= 0 || spec.PatchSize <= 0 ||
		(spec.ImageSize/spec.PatchSize)*spec.PatchSize != spec.ImageSize ||
		spec.VisionHidden <= 0 || spec.OutputHidden <= 0 || spec.NormEpsilon <= 0 {
		return Gemma3nVisionSpec{}, fmt.Errorf("projector: invalid Gemma 3n vision metadata: %+v", spec)
	}
	for channel := range spec.ImageStd {
		if !checked.PositiveFinite32(spec.ImageStd[channel]) || !checked.Finite32(spec.ImageMean[channel]) {
			return Gemma3nVisionSpec{}, fmt.Errorf("projector: invalid Gemma 3n normalization channel %d", channel)
		}
	}
	required := []string{
		"v.conv_stem.conv.weight", "v.conv_stem.bn.weight",
		"v.msfa.ffn.pw_exp.conv.weight", "v.msfa.ffn.pw_proj.conv.weight", "v.msfa.norm.weight",
		multimodalInputProjection, "mm.soft_emb_norm.weight",
	}
	for _, name := range []string{"v.msfa.ffn.pw_exp.bn.weight", "v.msfa.ffn.pw_proj.bn.weight"} {
		if hasTensor(file, name) {
			required = append(required, name)
		}
	}
	if hasTensor(file, "v.conv_stem.conv.bias") {
		required = append(required, "v.conv_stem.conv.bias")
	}
	for stage := tensor.FirstOffset; ; stage++ {
		found := tensor.FirstOffset
		for index := tensor.FirstOffset; ; index++ {
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
		if found == tensor.FirstOffset {
			break
		}
		spec.StageEnds = append(spec.StageEnds, len(spec.Blocks)-tensor.SingletonExtent)
	}
	if len(spec.Blocks) == tensor.FirstOffset {
		return Gemma3nVisionSpec{}, errors.New("projector: Gemma 3n MobileNetV5 has no blocks")
	}
	slices.Sort(required)
	spec.TensorNames = slices.Compact(required)
	for _, name := range spec.TensorNames {
		info, ok := file.Tensor(name)
		if !ok || tensorcatalog.ValidateInfo(info, tensorcatalog.Requirement{
			Ranks: []uint32{tensor.SingletonExtent, tensor.PairedExtent, tensor.TripleExtent, tensor.MaxDimensions},
		}) != nil {
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
	if bounds.Empty() {
		return nil, errors.New("projector: image bounds are empty")
	}
	resized := media.ResizeBicubic(source, spec.ImageSize, spec.ImageSize)
	pixels := make([]float32, 3*spec.ImageSize*spec.ImageSize)
	for y := 0; y < spec.ImageSize; y++ {
		for x := 0; x < spec.ImageSize; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			for channel, raw := range [media.RGBChannels]uint32{r, g, b} {
				pixels[channel+media.RGBChannels*(x+spec.ImageSize*y)] =
					(media.NormalizedRGBAChannel(raw) - spec.ImageMean[channel]) / spec.ImageStd[channel]
			}
		}
	}
	return pixels, nil
}

func (r *Gemma3nVisionRunner) EncodeImage(ctx context.Context, source image.Image) (reference.Value, error) {
	if r == nil || r.file == nil {
		return reference.Value{}, errRunnerClosed
	}
	pixels, err := PreprocessGemma3nVisionImage(source, r.spec)
	if err != nil {
		return reference.Value{}, err
	}
	builder := tensor.NewBuilder()
	input := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(media.RGBChannels, uint64(r.spec.ImageSize), uint64(r.spec.ImageSize)))
	graph := newProjectorGraphRuntime(ctx, r.file, r.cuda, builder)
	graph.hostFeeds[input] = pixelsValue(input, pixels)
	output := r.buildGraph(builder, input, graph.weight, graph.hostFeeds)
	results, err := graph.execute(output)
	if err != nil {
		return reference.Value{}, fmt.Errorf("projector: execute Gemma 3n graph: %w", err)
	}
	return results[output], nil
}

func (r *Gemma3nVisionRunner) validateGraph() error {
	builder := tensor.NewBuilder()
	input := builder.Input(visionInputTensor, dtype.F32, tensor.MustShape(media.RGBChannels, uint64(r.spec.ImageSize), uint64(r.spec.ImageSize)))
	weight := func(name string) *tensor.Tensor {
		info, ok := r.file.Tensor(name)
		if !ok {
			return nil
		}
		return builder.Input(name, dtype.F32, tensor.MustShape(info.Extents()...))
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
	cur := convolveSame(builder, input, stemWeight, stemBias, tensor.PairedExtent, false)
	cur = spatialWeightedRMSNorm(builder, cur, weight("v.conv_stem.bn.weight"), r.spec.NormEpsilon)
	cur = builder.GELUTanhExact(cur)
	features := make([]*tensor.Tensor, tensor.FirstOffset, tensor.PairedExtent)
	for blockIndex, block := range r.spec.Blocks {
		stride := uint32(tensor.SingletonExtent)
		if blockIndex == tensor.FirstOffset || slices.Contains(r.spec.StageEnds, blockIndex-tensor.SingletonExtent) {
			stride = tensor.PairedExtent
		}
		switch block.Kind {
		case gemma3nEdgeBlock:
			residual := cur
			cur = convolveSame(builder, cur, weight(block.Names["conv_exp.weight"]), nil, stride, false)
			if name := block.Names["bn1.weight"]; name != "" {
				cur = spatialWeightedRMSNorm(builder, cur, weight(name), r.spec.NormEpsilon)
			}
			cur = builder.GELUTanhExact(cur)
			cur = convolveSame(builder, cur, weight(block.Names["conv_pwl.weight"]), nil, tensor.SingletonExtent, false)
			if name := block.Names["bn2.weight"]; name != "" {
				cur = spatialWeightedRMSNorm(builder, cur, weight(name), r.spec.NormEpsilon)
			}
			if residual.Shape.Equal(cur.Shape) {
				cur = builder.Add(cur, residual)
			}
		case gemma3nAttentionBlock:
			cur = r.buildAttentionGraph(builder, cur, block, weight)
		default:
			residual := cur
			if name := block.Names["dw_start.conv.weight"]; name != "" {
				cur = convolveSame(builder, cur, weight(name), nil, tensor.SingletonExtent, true)
				if norm := block.Names["dw_start.bn.weight"]; norm != "" {
					cur = spatialWeightedRMSNorm(builder, cur, weight(norm), r.spec.NormEpsilon)
				}
			}
			if name := block.Names["pw_exp.conv.weight"]; name != "" {
				cur = convolveSame(builder, cur, weight(name), nil, tensor.SingletonExtent, false)
				if norm := block.Names["pw_exp.bn.weight"]; norm != "" {
					cur = spatialWeightedRMSNorm(builder, cur, weight(norm), r.spec.NormEpsilon)
				}
				cur = builder.GELUTanhExact(cur)
			}
			if name := block.Names["dw_mid.conv.weight"]; name != "" {
				cur = convolveSame(builder, cur, weight(name), nil, stride, true)
				if norm := block.Names["dw_mid.bn.weight"]; norm != "" {
					cur = spatialWeightedRMSNorm(builder, cur, weight(norm), r.spec.NormEpsilon)
				}
				cur = builder.GELUTanhExact(cur)
			}
			if name := block.Names["pw_proj.conv.weight"]; name != "" {
				cur = convolveSame(builder, cur, weight(name), nil, tensor.SingletonExtent, false)
				if norm := block.Names["pw_proj.bn.weight"]; norm != "" {
					cur = spatialWeightedRMSNorm(builder, cur, weight(norm), r.spec.NormEpsilon)
				}
			}
			if name := block.Names["layer_scale.gamma"]; name != "" {
				cur = scaleChannels(builder, cur, weight(name))
			}
			if residual.Shape.Equal(cur.Shape) {
				cur = builder.Add(cur, residual)
			}
		}
		if builder.Err() != nil {
			return nil
		}
		if slices.Contains(r.spec.StageEnds, blockIndex) || blockIndex == len(r.spec.Blocks)-tensor.SingletonExtent {
			features = append(features, cur)
		}
	}
	fusionWeight := weight("v.msfa.ffn.pw_exp.conv.weight")
	start, selectedChannels := len(features), uint64(tensor.FirstOffset)
	for start > tensor.FirstOffset && selectedChannels < fusionWeight.Shape.Dims[tensor.PairedExtent] {
		start--
		selectedChannels += features[start].Shape.Dims[tensor.FirstOffset]
	}
	if selectedChannels != fusionWeight.Shape.Dims[tensor.PairedExtent] {
		return nil
	}
	features = features[start:]
	targetW := int(features[tensor.FirstOffset].Shape.Dims[tensor.SingletonExtent])
	targetH := int(features[tensor.FirstOffset].Shape.Dims[tensor.PairedExtent])
	resized := make([]*tensor.Tensor, len(features))
	for index, feature := range features {
		resized[index] = resizeSpatialNearest(builder, feature, targetW, targetH)
	}
	cur = resized[tensor.FirstOffset]
	for _, feature := range resized[tensor.SingletonExtent:] {
		cur = builder.Concat(cur, feature, tensor.FirstOffset)
	}
	cur = convolveSame(builder, cur, fusionWeight, nil, tensor.SingletonExtent, false)
	if hasTensor(r.file, "v.msfa.ffn.pw_exp.bn.weight") {
		cur = spatialWeightedRMSNorm(builder, cur, weight("v.msfa.ffn.pw_exp.bn.weight"), r.spec.NormEpsilon)
	}
	cur = builder.GELUTanhExact(cur)
	cur = convolveSame(builder, cur, weight("v.msfa.ffn.pw_proj.conv.weight"), nil, tensor.SingletonExtent, false)
	if hasTensor(r.file, "v.msfa.ffn.pw_proj.bn.weight") {
		cur = spatialWeightedRMSNorm(builder, cur, weight("v.msfa.ffn.pw_proj.bn.weight"), r.spec.NormEpsilon)
	}
	gridSide := uint64(r.spec.ImageSize / r.spec.PatchSize)
	cur = averagePoolSpatial(builder, cur, int(gridSide), int(gridSide))
	cur = spatialWeightedRMSNorm(builder, cur, weight("v.msfa.norm.weight"), r.spec.NormEpsilon)
	channels := cur.Shape.Dims[tensor.FirstOffset]
	cur = builder.Reshape(cur, channels, cur.Shape.Dims[tensor.SingletonExtent]*cur.Shape.Dims[tensor.PairedExtent])
	cur = builder.Scale(cur, float32(math.Sqrt(float64(channels))))
	softNorm := builder.Reshape(weight("mm.soft_emb_norm.weight"), channels)
	cur = builder.WeightedRMSNorm(cur, softNorm, r.spec.NormEpsilon)
	cur = builder.MulMat(weight(multimodalInputProjection), cur)
	return builder.RMSNorm(cur, r.spec.NormEpsilon)
}

func (r *Gemma3nVisionRunner) buildAttentionGraph(
	builder *tensor.Builder,
	input *tensor.Tensor,
	block gemma3nVisionBlock,
	weight func(string) *tensor.Tensor,
) *tensor.Tensor {
	cur := input
	if name := block.Names["norm.weight"]; name != "" {
		cur = spatialWeightedRMSNorm(builder, cur, weight(name), r.spec.NormEpsilon)
	}
	q := convolveSame(builder, cur, weight(block.Names["attn.query.proj.weight"]), nil, tensor.SingletonExtent, false)
	buildKV := func(part string) *tensor.Tensor {
		value := cur
		if name := block.Names["attn."+part+".down_conv.weight"]; name != "" {
			value = convolveSame(builder, value, weight(name), nil, tensor.PairedExtent, true)
			if norm := block.Names["attn."+part+".norm.weight"]; norm != "" {
				value = spatialWeightedRMSNorm(builder, value, weight(norm), r.spec.NormEpsilon)
			}
		}
		return convolveSame(builder, value, weight(block.Names["attn."+part+".proj.weight"]), nil, tensor.SingletonExtent, false)
	}
	k, v := buildKV("key"), buildKV("value")
	headWidth := k.Shape.Dims[0]
	heads, exact := checked.DivExact64(q.Shape.Dims[0], headWidth)
	if !exact {
		builder.Reshape(q, q.Shape.Dims[tensor.FirstOffset]+tensor.SingletonExtent)
		return input
	}
	q = builder.Reshape(q, headWidth, heads, q.Shape.Dims[1]*q.Shape.Dims[2])
	k = builder.Reshape(k, headWidth, tensor.SingletonExtent, k.Shape.Dims[tensor.SingletonExtent]*k.Shape.Dims[tensor.PairedExtent])
	v = builder.Reshape(v, headWidth, tensor.SingletonExtent, v.Shape.Dims[tensor.SingletonExtent]*v.Shape.Dims[tensor.PairedExtent])
	attention := compileVisionAttention(int(headWidth*heads), int(heads)).graph(builder, q, k, v)
	if attention == nil {
		return input
	}
	attention = builder.Reshape(attention, headWidth*heads, input.Shape.Dims[1], input.Shape.Dims[2])
	attention = convolveSame(builder, attention, weight(block.Names["attn.output.proj.weight"]), nil, tensor.SingletonExtent, false)
	if attention == nil {
		return input
	}
	if name := block.Names["layer_scale.gamma"]; name != "" {
		attention = scaleChannels(builder, attention, weight(name))
	}
	if attention.Shape.Equal(input.Shape) {
		attention = builder.Add(attention, input)
	}
	return attention
}

func (r *Gemma3nVisionRunner) imagePromptProgram() compiledImagePromptProgram {
	compile := func(history bool) imagePromptPlan {
		return delimitedImagePromptPlan(
			"Gemma 3n", Gemma3nImagePad, "Gemma 3n image placeholder", history, r.spec.OutputHidden,
			"<start_of_image>", "<end_of_image>",
		)
	}
	return compiledImagePromptProgram{
		Default: compile(false),
		History: compile(true),
		Encode:  referenceImageEncoder(r.EncodeImage),
		Prepare: func(text []string, options PromptOptions) ([]string, error) {
			if len(text) < tensor.SingletonExtent {
				return nil, errors.New("projector: Gemma 3n image/text sequence is inconsistent")
			}
			if options.History {
				return text, nil
			}
			prepared := slices.Clone(text)
			prepared[tensor.FirstOffset] = "<bos><start_of_turn>user\n" + prepared[tensor.FirstOffset]
			last := len(prepared) - tensor.SingletonExtent
			prepared[last] += "<end_of_turn>\n<start_of_turn>model\n"
			return prepared, nil
		},
	}
}
