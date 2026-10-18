package hfgguf

import (
	"encoding/json"
	"errors"
	"overgo/internal/hfrepo"
	"overgo/internal/model"
	"overgo/internal/safetensors"
	"strings"
	"testing"
)

func TestLanguageCatalogBindings(t *testing.T) {
	repository := baseLanguageFixture(t)
	catalog, err := InspectLanguageModel(repository, LanguageHidden)
	if err != nil {
		t.Fatal(err)
	}
	bindings := catalog.Bindings()
	if len(bindings) != len(catalog.mappings) {
		t.Fatal("missing materialization bindings")
	}
	for index, binding := range bindings {
		mapping := catalog.mappings[index]
		if binding.Source != mapping.tensor.Name || binding.Target != mapping.name || binding.SourceBytes != uint64(mapping.tensor.Size()) {
			t.Fatalf("binding disagrees with materializer: %+v", binding)
		}
		if binding.TargetBytes == 0 || binding.TargetStorage == "" {
			t.Fatal("missing destination storage")
		}
		if len(binding.Transforms) > 0 {
			encoded, err := json.Marshal(binding.Transforms)
			if err != nil {
				t.Fatal(err)
			}
			var decoded []TensorTransform
			if err := json.Unmarshal(encoded, &decoded); err != nil {
				t.Fatal(err)
			}
			if decoded[0] != mapping.transforms[0] {
				t.Fatal("transform cannot round trip")
			}
			binding.Transforms[0].Operation = "mutated"
			if mapping.transforms[0].Operation == "mutated" {
				t.Fatal("description aliases executable mapping")
			}
		}
		binding.SourceShape[0]++
		if binding.SourceShape[0] == mapping.tensor.Shape[0] {
			t.Fatal("description aliases source shape")
		}
	}
}

type refuseLanguagePayload struct{ reads int }

func (r *refuseLanguagePayload) ReadAt([]byte, int64) (int, error) {
	r.reads++
	return 0, errors.New("payload read during inspection")
}
func baseLanguageFixture(t *testing.T) *hfrepo.Repository {
	t.Helper()
	repository := qwen35Fixture(t)
	repository.Identity.Architectures = []string{"Qwen3_5Model"}
	tensors := make(map[string]safetensors.Tensor)
	for name, tensor := range repository.Tensors.Tensors {
		if strings.HasPrefix(name, "mtp.") {
			continue
		}
		name = strings.TrimPrefix(name, "model.")
		tensor.Name = name
		tensors[name] = tensor
	}
	repository.Tensors.Tensors = tensors
	return repository
}

func TestLanguageCatalogOutputContract(t *testing.T) {
	t.Parallel()
	for _, kind := range []string{"base", "wrapper", "qwen2", "llama"} {
		t.Run(kind, func(t *testing.T) {
			var repository *hfrepo.Repository
			switch kind {
			case "base":
				repository = baseLanguageFixture(t)
			case "wrapper":
				repository = qwen35Fixture(t)
			default:
				repository = denseFixture(t, kind, false)
			}
			reader := &refuseLanguagePayload{}
			for name, tensor := range repository.Tensors.Tensors {
				bounded, err := safetensors.NewTensor(name, tensor.DType, tensor.Shape, reader, 0, tensor.Size())
				if err != nil {
					t.Fatal(err)
				}
				repository.Tensors.Tensors[name] = bounded
			}
			hidden, err := InspectLanguageModel(repository, LanguageHidden)
			if err != nil {
				t.Fatal(err)
			}
			if hidden.Output != LanguageHidden || hidden.Weights.Output != nil || hidden.Spec.NextNPredictLayers != 0 {
				t.Fatal("hidden-state catalog contains prediction topology")
			}
			if _, err := model.CompileModelPlanWithProfile(hidden.Spec, hidden.Weights, hidden.Spec.Profile()); err != nil {
				t.Fatal(err)
			}
			if kind == "base" || kind == "wrapper" {
				if hidden.Spec.BlockCount != 4 || len(hidden.Weights.Layers) != 4 {
					t.Fatal("hidden-state catalog lost trunk layers")
				}
			}
			prediction, err := InspectLanguageModel(repository, LanguageTokens)
			if kind == "base" {
				if err == nil {
					t.Fatal("base model accepted as token generator")
				}
				if _, err := ValidateRepository(repository); err == nil {
					t.Fatal("legacy validation advertises base-model prediction")
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				old, err := ValidateRepository(repository)
				if err != nil {
					t.Fatal(err)
				}
				if prediction.Spec.BlockCount != old.BlockCount || prediction.Spec.NextNPredictLayers != old.NextNPredictLayers {
					t.Fatal("prediction topology changed")
				}
			}
			if reader.reads != 0 {
				t.Fatalf("header inspection read payload %d times", reader.reads)
			}
		})
	}
}

func TestLanguageCatalogRefusesInvalidBase(t *testing.T) {
	t.Parallel()
	for _, problem := range []string{"missing-norm", "wrong-embedding-shape", "unknown-language-tensor", "prediction-tensor", "unknown-component", "unknown-output", "nil"} {
		t.Run(problem, func(t *testing.T) {
			repository := baseLanguageFixture(t)
			output := LanguageHidden
			switch problem {
			case "missing-norm":
				delete(repository.Tensors.Tensors, "language_model.norm.weight")
			case "wrong-embedding-shape":
				repository.Tensors.Tensors["language_model.embed_tokens.weight"] = testTensor(t, "language_model.embed_tokens.weight", []uint64{31, 8})
			case "unknown-language-tensor":
				repository.Tensors.Tensors["language_model.mystery.weight"] = testTensor(t, "language_model.mystery.weight", []uint64{8})
			case "prediction-tensor":
				repository.Tensors.Tensors["lm_head.weight"] = testTensor(t, "lm_head.weight", []uint64{32, 8})
			case "unknown-component":
				repository.Tensors.Tensors["audio.weight"] = testTensor(t, "audio.weight", []uint64{8})
			case "unknown-output":
				output = "unknown"
			case "nil":
				repository = nil
			}
			if _, err := InspectLanguageModel(repository, output); err == nil {
				t.Fatal("invalid language catalog accepted")
			}
		})
	}
}
