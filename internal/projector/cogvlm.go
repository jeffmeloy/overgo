package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"strings"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

const (
	cogVLMProjectorType      = "cogvlm"
	cogVLMAdapterNormEpsilon = 1e-5
)

type CogVLMVisionSpec struct {
	ImageSize           int
	PatchSize           int
	Hidden              int
	Intermediate        int
	OutputHidden        int
	AdapterIntermediate int
	Layers              int
	Heads               int
	LayerNormEpsilon    float32
	ImageMean           [3]float32
	ImageStd            [3]float32
	GatedFFN            []bool
}

type CogVLMVisionRunner struct {
	file *gguf.File
	spec CogVLMVisionSpec
	cuda *projectorCUDA
}

type CogVLMVisionOpenOptions struct {
	CUDA          bool
	DeviceOrdinal int
}

func OpenCogVLMVision(path string) (*CogVLMVisionRunner, error) {
	return OpenCogVLMVisionWithOptions(path, CogVLMVisionOpenOptions{})
}

func OpenCogVLMVisionWithOptions(path string, options CogVLMVisionOpenOptions) (*CogVLMVisionRunner, error) {
	return openProjectorResource(path, func(file *gguf.File) (*CogVLMVisionRunner, error) {
		spec, err := ReadCogVLMVisionSpec(file)
		if err != nil {
			return nil, err
		}
		catalog, err := validateCogVLMVisionCatalog(file, spec)
		if err != nil {
			return nil, err
		}
		runner := &CogVLMVisionRunner{file: file, spec: spec}
		if options.CUDA {
			runner.cuda, err = openProjectorCUDA(context.Background(), file, catalog, nil, options.DeviceOrdinal)
			if err != nil {
				return nil, fmt.Errorf("projector: initialize CogVLM CUDA: %w", err)
			}
		}
		return runner, nil
	})
}

func (r *CogVLMVisionRunner) Close() error {
	if r == nil {
		return nil
	}
	return closeProjectorResources(&r.file, &r.cuda)
}

func (r *CogVLMVisionRunner) Spec() CogVLMVisionSpec {
	if r == nil {
		return CogVLMVisionSpec{}
	}
	return r.spec
}

func ReadCogVLMVisionSpec(file *gguf.File) (CogVLMVisionSpec, error) {
	if file == nil {
		return CogVLMVisionSpec{}, errors.New("projector: GGUF file is nil")
	}
	architecture, err := metadataString(file, "general.architecture")
	if err != nil {
		return CogVLMVisionSpec{}, err
	}
	projectorType, err := metadataString(file, "clip.projector_type")
	if err != nil {
		return CogVLMVisionSpec{}, err
	}
	if architecture != "clip" || projectorType != cogVLMProjectorType {
		return CogVLMVisionSpec{}, fmt.Errorf("projector: architecture/type %q/%q is not clip/%s", architecture, projectorType, cogVLMProjectorType)
	}
	hasVision, err := metadataBool(file, "clip.has_vision_encoder")
	if err != nil || !hasVision {
		if err != nil {
			return CogVLMVisionSpec{}, err
		}
		return CogVLMVisionSpec{}, errors.New("projector: vision encoder is disabled")
	}
	values := make([]int, 7)
	for index, key := range []string{
		"clip.vision.image_size", "clip.vision.patch_size", "clip.vision.embedding_length",
		"clip.vision.feed_forward_length", "clip.vision.projection_dim", "clip.vision.block_count",
		"clip.vision.attention.head_count",
	} {
		value, valueErr := metadataUint32(file, key)
		if valueErr != nil {
			return CogVLMVisionSpec{}, valueErr
		}
		values[index] = int(value)
	}
	epsilon, err := metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return CogVLMVisionSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", 3)
	if err != nil {
		return CogVLMVisionSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", 3)
	if err != nil {
		return CogVLMVisionSpec{}, err
	}
	up, ok := file.Tensor("mm.up.weight")
	if !ok || up.Dimensions != 2 {
		return CogVLMVisionSpec{}, errors.New("projector: CogVLM adapter up tensor is unavailable or invalid")
	}
	spec := CogVLMVisionSpec{
		ImageSize: values[0], PatchSize: values[1], Hidden: values[2], Intermediate: values[3],
		OutputHidden: values[4], AdapterIntermediate: int(up.Shape[1]), Layers: values[5], Heads: values[6],
		LayerNormEpsilon: epsilon, GatedFFN: make([]bool, values[5]),
	}
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	for layer := range spec.GatedFFN {
		spec.GatedFFN[layer] = hasTensor(file, fmt.Sprintf("v.blk.%d.ffn_gate.weight", layer))
	}
	if err := spec.validate(); err != nil {
		return CogVLMVisionSpec{}, err
	}
	return spec, nil
}

func (s CogVLMVisionSpec) validate() error {
	if s.ImageSize <= 0 || s.PatchSize <= 0 || s.Hidden <= 0 || s.Intermediate <= 0 || s.OutputHidden <= 0 ||
		s.AdapterIntermediate <= 0 || s.Layers <= 0 || s.Heads <= 0 || s.Hidden%s.Heads != 0 ||
		s.ImageSize%s.PatchSize != 0 || s.LayerNormEpsilon <= 0 || len(s.GatedFFN) != s.Layers {
		return fmt.Errorf("projector: invalid CogVLM vision metadata: %+v", s)
	}
	for channel := range s.ImageStd {
		if s.ImageStd[channel] <= 0 || !finite32(s.ImageMean[channel]) || !finite32(s.ImageStd[channel]) {
			return fmt.Errorf("projector: invalid CogVLM normalization channel %d", channel)
		}
	}
	return nil
}

func validateCogVLMVisionCatalog(file *gguf.File, spec CogVLMVisionSpec) ([]string, error) {
	grid := spec.ImageSize / spec.PatchSize
	required := map[string][]uint64{
		"v.patch_embd.weight":    {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.class_embd":           {uint64(spec.Hidden), 1},
		"v.position_embd.weight": {uint64(spec.Hidden), uint64(grid*grid + 1)},
		"mm.model.fc.weight":     {uint64(spec.Hidden), uint64(spec.OutputHidden)},
		"mm.post_fc_norm.weight": {uint64(spec.OutputHidden)}, "mm.post_fc_norm.bias": {uint64(spec.OutputHidden)},
		"mm.up.weight":   {uint64(spec.OutputHidden), uint64(spec.AdapterIntermediate)},
		"mm.gate.weight": {uint64(spec.OutputHidden), uint64(spec.AdapterIntermediate)},
		"mm.down.weight": {uint64(spec.AdapterIntermediate), uint64(spec.OutputHidden)},
		"v.boi":          {uint64(spec.OutputHidden), 1, 1}, "v.eoi": {uint64(spec.OutputHidden), 1, 1},
	}
	if hasTensor(file, "v.patch_embd.bias") {
		required["v.patch_embd.bias"] = []uint64{uint64(spec.Hidden)}
	}
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for name, shape := range map[string][]uint64{
			"attn_qkv.weight": {uint64(spec.Hidden), uint64(3 * spec.Hidden)}, "attn_qkv.bias": {uint64(3 * spec.Hidden)},
			"attn_out.weight": {uint64(spec.Hidden), uint64(spec.Hidden)}, "attn_out.bias": {uint64(spec.Hidden)},
			"ffn_up.weight":   {uint64(spec.Hidden), uint64(spec.Intermediate)},
			"ffn_down.weight": {uint64(spec.Intermediate), uint64(spec.Hidden)},
			"ln1.weight":      {uint64(spec.Hidden)}, "ln1.bias": {uint64(spec.Hidden)},
			"ln2.weight": {uint64(spec.Hidden)}, "ln2.bias": {uint64(spec.Hidden)},
		} {
			required[prefix+name] = shape
		}
		if spec.GatedFFN[layer] {
			required[prefix+"ffn_gate.weight"] = []uint64{uint64(spec.Hidden), uint64(spec.Intermediate)}
		}
		for _, name := range []string{"ffn_up", "ffn_gate", "ffn_down"} {
			full := prefix + name + ".bias"
			if hasTensor(file, full) {
				width := spec.Hidden
				if name != "ffn_down" {
					width = spec.Intermediate
				}
				required[full] = []uint64{uint64(width)}
			}
		}
	}
	return validateProjectorTensorCatalog(file, required)
}

func PreprocessCogVLMImage(source image.Image, spec CogVLMVisionSpec) ([]float32, error) {
	if source == nil {
		return nil, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return nil, err
	}
	bounds := source.Bounds()
	if bounds.Dx() <= 0 || bounds.Dy() <= 0 {
		return nil, errors.New("projector: image bounds are empty")
	}
	resized := resizeImageBicubic(source, spec.ImageSize, spec.ImageSize)
	grid, patchArea := spec.ImageSize/spec.PatchSize, spec.PatchSize*spec.PatchSize
	pixels := make([]float32, grid*grid*3*patchArea)
	for patchY := 0; patchY < grid; patchY++ {
		for patchX := 0; patchX < grid; patchX++ {
			row := (patchY*grid + patchX) * 3 * patchArea
			for channel := 0; channel < 3; channel++ {
				position := row + channel*patchArea
				for y := 0; y < spec.PatchSize; y++ {
					for x := 0; x < spec.PatchSize; x++ {
						r, g, b, _ := resized.At(patchX*spec.PatchSize+x, patchY*spec.PatchSize+y).RGBA()
						value := [3]uint32{r, g, b}[channel]
						pixels[position] = (float32(value>>8)/255 - spec.ImageMean[channel]) / spec.ImageStd[channel]
						position++
					}
				}
			}
		}
	}
	return pixels, nil
}

func (r *CogVLMVisionRunner) EncodeImage(ctx context.Context, source image.Image) (reference.Value, error) {
	if r == nil || r.file == nil {
		return reference.Value{}, errors.New("projector: runner is closed")
	}
	pixels, err := PreprocessCogVLMImage(source, r.spec)
	if err != nil {
		return reference.Value{}, err
	}
	return r.encodeGraph(ctx, pixels)
}

func (r *CogVLMVisionRunner) BuildImagePrompt(
	ctx context.Context,
	tokenizerAPI ImageTokenizer,
	source image.Image,
	beforeImage, afterImage string,
	_ bool,
) (MultimodalPrompt, error) {
	return r.BuildImagesPrompt(ctx, tokenizerAPI, []image.Image{source}, []string{beforeImage, afterImage}, false)
}

func (r *CogVLMVisionRunner) BuildImagesPrompt(
	ctx context.Context,
	tokenizerAPI ImageTokenizer,
	sources []image.Image,
	text []string,
	_ bool,
) (MultimodalPrompt, error) {
	if tokenizerAPI == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if len(sources) == 0 || len(text) != len(sources)+1 {
		return MultimodalPrompt{}, errors.New("projector: CogVLM image/text sequence is inconsistent")
	}
	return r.buildImagesPrompt(ctx, tokenizerAPI, sources, "Question: "+strings.Join(text, "")+" Answer:")
}

func (r *CogVLMVisionRunner) BuildImagesHistoryPrompt(
	ctx context.Context,
	tokenizerAPI ImageTokenizer,
	sources []image.Image,
	text []string,
) (MultimodalPrompt, error) {
	if tokenizerAPI == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if len(sources) == 0 || len(text) != len(sources)+1 {
		return MultimodalPrompt{}, errors.New("projector: CogVLM history sequence is inconsistent")
	}
	return r.buildImagesPrompt(ctx, tokenizerAPI, sources, strings.Join(text, ""))
}

func (r *CogVLMVisionRunner) buildImagesPrompt(
	ctx context.Context,
	tokenizerAPI ImageTokenizer,
	sources []image.Image,
	text string,
) (MultimodalPrompt, error) {
	var embeddings []float32
	visualTokens := 0
	for index, source := range sources {
		value, err := r.EncodeImage(ctx, source)
		if err != nil {
			return MultimodalPrompt{}, fmt.Errorf("projector: encode CogVLM image %d: %w", index, err)
		}
		embeddings = append(embeddings, value.Data...)
		visualTokens += int(value.Shape.Dims[1])
	}
	ids, err := tokenizerAPI.TokenizeText(text, true, true)
	if err != nil {
		return MultimodalPrompt{}, fmt.Errorf("projector: tokenize CogVLM prompt: %w", err)
	}
	if len(ids) == 0 {
		return MultimodalPrompt{}, errors.New("projector: CogVLM tokenizer returned no BOS token")
	}
	tokenIDs := make([]tokenizer.TokenID, 0, len(ids)+visualTokens)
	tokenIDs = append(tokenIDs, ids[0])
	tokenIDs = append(tokenIDs, make([]tokenizer.TokenID, visualTokens)...)
	for index := 1; index <= visualTokens; index++ {
		tokenIDs[index] = ids[0]
	}
	tokenIDs = append(tokenIDs, ids[1:]...)
	indices := sequentialTokenIndices(1, visualTokens)
	return MultimodalPrompt{
		TokenIDs: tokenIDs, Embeddings: embeddings, EmbeddingWidth: r.spec.OutputHidden,
		EmbeddingStart: 1, EmbeddingTokenIndices: indices,
		VisualBlocks: []AttentionBlock{{Start: 1, End: uint32(visualTokens + 1)}},
	}, nil
}
