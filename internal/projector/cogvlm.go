package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

const (
	cogVLMProjectorType      = "cogvlm"
	cogVLMAdapterNormEpsilon = 1e-5
)

type CogVLMVisionSpec struct {
	visionBackboneSpec
	OutputHidden        int
	AdapterIntermediate int
	GatedFFN            []bool
}

type CogVLMVisionRunner struct {
	projectorResources
	spec      CogVLMVisionSpec
	attention visionAttentionPlan
}

func (r *CogVLMVisionRunner) Spec() CogVLMVisionSpec {
	if r == nil {
		return CogVLMVisionSpec{}
	}
	return r.spec
}

func ReadCogVLMVisionSpec(file *gguf.File) (CogVLMVisionSpec, error) {
	spec := CogVLMVisionSpec{}
	if err := readVisionBackbone(file, cogVLMProjectorType, &spec.OutputHidden, &spec.visionBackboneSpec); err != nil {
		return CogVLMVisionSpec{}, err
	}
	up, ok := file.Tensor("mm.up.weight")
	if !ok || up.Dimensions != 2 {
		return CogVLMVisionSpec{}, errors.New("projector: CogVLM adapter up tensor is unavailable or invalid")
	}
	spec.AdapterIntermediate = int(up.Shape[1])
	spec.GatedFFN = make([]bool, spec.Layers)
	for layer := range spec.GatedFFN {
		spec.GatedFFN[layer] = hasTensor(file, fmt.Sprintf("v.blk.%d.ffn_gate.weight", layer))
	}
	if err := spec.validate(); err != nil {
		return CogVLMVisionSpec{}, err
	}
	return spec, nil
}

func (s CogVLMVisionSpec) validate() error {
	if err := s.visionBackboneSpec.validate(); err != nil {
		return err
	}
	if s.OutputHidden <= 0 || s.AdapterIntermediate <= 0 || len(s.GatedFFN) != s.Layers {
		return fmt.Errorf("projector: invalid CogVLM vision metadata: %+v", s)
	}
	return nil
}

func validateCogVLMVisionCatalog(file *gguf.File, spec CogVLMVisionSpec) ([]string, error) {
	grid := spec.ImageSize / spec.PatchSize
	required := map[string][]uint64{
		visionClassEmbeddingTensor: {uint64(spec.Hidden), 1},
		multimodalProjectionWeight: {uint64(spec.Hidden), uint64(spec.OutputHidden)},
		"mm.post_fc_norm.weight":   {uint64(spec.OutputHidden)}, "mm.post_fc_norm.bias": {uint64(spec.OutputHidden)},
		"mm.up.weight":   {uint64(spec.OutputHidden), uint64(spec.AdapterIntermediate)},
		"mm.gate.weight": {uint64(spec.OutputHidden), uint64(spec.AdapterIntermediate)},
		"mm.down.weight": {uint64(spec.AdapterIntermediate), uint64(spec.OutputHidden)},
		"v.boi":          {uint64(spec.OutputHidden), 1, 1}, "v.eoi": {uint64(spec.OutputHidden), 1, 1},
	}
	addSpatialVisionEmbeddingCatalog(file, required, spec.visionBackboneSpec, grid*grid+1, tensorOptional)
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
						pixels[position] = (normalizedImageChannel(value) - spec.ImageMean[channel]) / spec.ImageStd[channel]
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
		return reference.Value{}, errRunnerClosed
	}
	pixels, err := PreprocessCogVLMImage(source, r.spec)
	if err != nil {
		return reference.Value{}, err
	}
	return r.encodeGraph(ctx, pixels)
}

func (r *CogVLMVisionRunner) imagesPrompt(
	ctx context.Context,
	tokenizerAPI ImageTokenizer,
	sources []image.Image,
	text []string,
	options PromptOptions,
) (MultimodalPrompt, error) {
	if tokenizerAPI == nil {
		return MultimodalPrompt{}, errors.New("projector: tokenizer is nil")
	}
	if len(sources) == 0 || len(text) != len(sources)+1 {
		return MultimodalPrompt{}, errors.New("projector: CogVLM image/text sequence is inconsistent")
	}
	prompt := strings.Join(text, "")
	if !options.History {
		prompt = "Question: " + prompt + " Answer:"
	}
	return r.buildImagesPrompt(ctx, tokenizerAPI, sources, prompt)
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
