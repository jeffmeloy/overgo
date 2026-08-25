package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const cogVLMProjectorType = "cogvlm"

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
	if err := readProjectionNorm(file, &spec.ProjectionNormEpsilon); err != nil {
		return CogVLMVisionSpec{}, err
	}
	up, ok := file.Tensor("mm.up.weight")
	spec.AdapterIntermediate, ok = matrixRowsInt(up, ok)
	if !ok {
		return CogVLMVisionSpec{}, errors.New("projector: CogVLM adapter up tensor is unavailable or invalid")
	}
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
	if !checked.PositiveInts(s.OutputHidden, s.AdapterIntermediate) ||
		!checked.PositiveFinite32(s.ProjectionNormEpsilon) || len(s.GatedFFN) != s.Layers {
		return fmt.Errorf("projector: invalid CogVLM vision metadata: %+v", s)
	}
	return nil
}

func validateCogVLMVisionCatalog(file *gguf.File, spec CogVLMVisionSpec) ([]string, error) {
	grid := spec.ImageSize / spec.PatchSize
	required := map[string][]uint64{
		visionClassEmbeddingTensor: {uint64(spec.Hidden), uint64(tensor.SingletonExtent)},
		multimodalProjectionWeight: {uint64(spec.Hidden), uint64(spec.OutputHidden)},
		"mm.post_fc_norm.weight":   {uint64(spec.OutputHidden)}, "mm.post_fc_norm.bias": {uint64(spec.OutputHidden)},
		"mm.up.weight":   {uint64(spec.OutputHidden), uint64(spec.AdapterIntermediate)},
		"mm.gate.weight": {uint64(spec.OutputHidden), uint64(spec.AdapterIntermediate)},
		"mm.down.weight": {uint64(spec.AdapterIntermediate), uint64(spec.OutputHidden)},
		"v.boi":          {uint64(spec.OutputHidden), uint64(tensor.SingletonExtent), uint64(tensor.SingletonExtent)},
		"v.eoi":          {uint64(spec.OutputHidden), uint64(tensor.SingletonExtent), uint64(tensor.SingletonExtent)},
	}
	addSpatialVisionEmbeddingCatalog(file, required, spec.visionBackboneSpec, grid*grid+tensor.SingletonExtent, tensorOptional)
	for layer := range spec.Layers {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for name, shape := range map[string][]uint64{
			"attn_qkv.weight": {uint64(spec.Hidden), uint64(tensor.TripleExtent * spec.Hidden)}, "attn_qkv.bias": {uint64(tensor.TripleExtent * spec.Hidden)},
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
	if err := spec.validate(); err != nil {
		return nil, err
	}
	if _, err := imageBounds(source); err != nil {
		return nil, err
	}
	resized := media.ResizeBicubic(source, spec.ImageSize, spec.ImageSize)
	patches, err := patchRasterImage(resized, spec.PatchSize, spec.ImageMean, spec.ImageStd)
	if err != nil {
		return nil, err
	}
	return patches.PixelValues, nil
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

func (r *CogVLMVisionRunner) imagePromptProgram() compiledImagePromptProgram {
	return compiledImagePromptProgram{Custom: func(
		ctx context.Context,
		tokenizerAPI ImageTokenizer,
		sources []image.Image,
		text []string,
		options PromptOptions,
	) (MultimodalPrompt, error) {
		if err := validatePromptSequence(tokenizerAPI, len(sources), text, "CogVLM image/text"); err != nil {
			return MultimodalPrompt{}, err
		}
		prompt := strings.Join(text, "")
		if !options.History {
			prompt = "Question: " + prompt + " Answer:"
		}
		return executePrefixInsertedImagePrompt(
			ctx, tokenizerAPI, sources, prompt, "CogVLM", r.spec.OutputHidden, r.EncodeImage,
		)
	}}
}
