package latentimage

import (
	"strings"

	"overgo/internal/safetensors"
)

// A ComfyUI text encoder file (the MOSS Ultra workflow's Qwen3.5-9B-abliterated_int8_convrot) holds a Qwen3.5 model as
// a Hugging Face checkpoint for generation names it (model.language_model.layers.0..., where MOSS-Image's own text
// encoder, a base model, names language_model.layers.0...), with a vision tower and the head, which conditioning does
// not read; its ConvRot weights the repository reads as the bfloat16 weights they decode to (comfyquant.Decoded).
// comfyLanguageView reads such a checkpoint as MOSS-Image's encoder holds its own: each language-model tensor under its
// name without the model. prefix, and neither the vision tower, the head nor a draft layer.
const (
	comfyModelPrefix    = "model."
	comfyLanguagePrefix = "model.language_model."
	comfyVisual         = "model.visual."
	comfyHead           = "lm_head."
	comfyDraft          = "mtp."
)

// comfyLanguageView is the language model of a checkpoint for generation as a base model's tensors, the source returned
// as it is where it names its tensors as a base model does. The view takes the source's files over (Retensored).
func comfyLanguageView(source *safetensors.Source) *safetensors.Source {
	generation := false
	for name := range source.Tensors {
		if strings.HasPrefix(name, comfyLanguagePrefix) {
			generation = true
			break
		}
	}
	if !generation {
		return source
	}
	tensors := map[string]safetensors.Tensor{}
	for name, value := range source.Tensors {
		if strings.HasPrefix(name, comfyVisual) || strings.HasPrefix(name, comfyHead) || strings.HasPrefix(name, comfyDraft) {
			continue
		}
		renamed := strings.TrimPrefix(name, comfyModelPrefix)
		value.Name = renamed
		tensors[renamed] = value
	}
	return source.Retensored(tensors)
}
