package hfgguf

import (
	"errors"
	"fmt"
	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/model"
	"slices"
	"strings"
)

// LanguageOutput names the execution contract a language catalog must support.
type LanguageOutput string

const (
	// LanguageTokens retains the prediction topology required for token generation.
	LanguageTokens LanguageOutput = "tokens"
	// LanguageHidden excludes prediction-only weights from the hidden-state consumer.
	LanguageHidden LanguageOutput = "hidden-states"
)

// LanguageCatalog holds shape-checked metadata without loading tensor payloads.
type LanguageCatalog struct {
	Output   LanguageOutput
	Spec     model.Spec
	Weights  model.Weights
	mappings []tensorMapping
}

// TensorBinding binds source storage to the converter's ordered operations.
// Shapes use source row-major order on both sides of the conversion.
type TensorBinding struct {
	Source        string            `json:"source"`
	Target        string            `json:"target"`
	SourceShape   []uint64          `json:"source_shape"`
	TargetShape   []uint64          `json:"target_shape"`
	SourceStorage string            `json:"source_storage"`
	TargetStorage string            `json:"target_storage"`
	SourceBytes   uint64            `json:"source_bytes"`
	TargetBytes   uint64            `json:"target_bytes"`
	Transforms    []TensorTransform `json:"transforms,omitempty"`
}

// Bindings returns independent descriptions of the validated materialization mappings.
// Storage conversion precedes the declared payload operations.
func (c LanguageCatalog) Bindings() []TensorBinding {
	bindings := make([]TensorBinding, 0, len(c.mappings))
	for _, mapping := range c.mappings {
		traits, _ := mapping.storage.Traits()
		bindings = append(bindings, TensorBinding{
			Source: mapping.tensor.Name, Target: mapping.name,
			SourceShape: slices.Clone(mapping.tensor.Shape), TargetShape: slices.Clone(mapping.shape),
			SourceStorage: mapping.tensor.DType, TargetStorage: mapping.storage.String(),
			SourceBytes: uint64(mapping.tensor.Size()), TargetBytes: mapping.tensor.Elements() * traits.TypeSize,
			Transforms: slices.Clone(mapping.transforms),
		})
	}
	return bindings
}

// InspectLanguageModel resolves an explicit output contract through existing mappings.
// Base-model conditioning does not establish token-generation eligibility.
func InspectLanguageModel(repository *hfrepo.Repository, output LanguageOutput) (LanguageCatalog, error) {
	if repository == nil || repository.Tensors == nil {
		return LanguageCatalog{}, errors.New("HF/GGUF adapter: nil repository")
	}
	if output != LanguageTokens && output != LanguageHidden {
		return LanguageCatalog{}, fmt.Errorf("HF/GGUF adapter: unknown language output %q", output)
	}
	var items []gguf.Metadata
	var mappings []tensorMapping
	var err error
	switch repository.Identity.ModelType {
	case "llama", "qwen2":
		items, err = denseMetadata(repository, repository.Identity.ModelType)
		if err == nil {
			mappings, err = collectTensorMappings(repository.Tensors, func(name string) (string, bool, error) {
				if output == LanguageHidden && (name == "lm_head.weight" || name == "lm_head.bias") {
					return "", false, nil
				}
				return denseTensorName(name)
			}, nil, nil)
		}
	case "qwen3_5":
		var blocks uint32
		items, blocks, err = qwen35MetadataForOutput(repository, output)
		if err == nil {
			mappings, err = qwen35MappingsForOutput(repository, blocks, output)
		}
	default:
		return LanguageCatalog{}, fmt.Errorf("HF/GGUF adapter: unsupported language model type %q", repository.Identity.ModelType)
	}
	if err != nil {
		return LanguageCatalog{}, err
	}
	tensors, err := mappedTensorCatalog(mappings)
	if err != nil {
		return LanguageCatalog{}, err
	}
	file, err := gguf.Catalog(items, tensors)
	if err != nil {
		return LanguageCatalog{}, err
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return LanguageCatalog{}, err
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return LanguageCatalog{}, err
	}
	if output == LanguageHidden && (weights.Output != nil || spec.NextNPredictLayers != 0) {
		return LanguageCatalog{}, errors.New("HF/GGUF adapter: hidden-state catalog contains prediction topology")
	}
	return LanguageCatalog{Output: output, Spec: spec, Weights: weights, mappings: mappings}, nil
}

func qwen35BaseModel(repository *hfrepo.Repository) bool {
	return slices.Contains(repository.Identity.Architectures, "Qwen3_5Model")
}

func qwen35MappingsForOutput(repository *hfrepo.Repository, blocks uint32, output LanguageOutput) ([]tensorMapping, error) {
	dims, err := qwen35LinearDimensions(repository)
	if err != nil {
		return nil, err
	}
	base := qwen35BaseModel(repository)
	return collectTensorMappings(repository.Tensors, func(name string) (string, bool, error) {
		if output == LanguageHidden && (strings.HasPrefix(name, "mtp.") || name == "lm_head.weight") {
			if base {
				return "", false, fmt.Errorf("HF/GGUF adapter: base model has prediction tensor %q", name)
			}
			return "", false, nil
		}
		if base {
			if !strings.HasPrefix(name, "language_model.") && !strings.HasPrefix(name, "visual.") {
				return "", false, fmt.Errorf("HF/GGUF adapter: base model tensor %q has no declared component", name)
			}
			name = "model." + name
		}
		return qwen35TensorName(name, blocks)
	}, qwen35TensorShape, qwen35TensorTransforms(dims))
}
