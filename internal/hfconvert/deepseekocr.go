package hfconvert

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/hfgguf"
	"overgo/internal/hfrepo"
	"overgo/internal/modelartifact"
	"overgo/internal/projector"
)

// deepSeekOCRTokenConfig is what the tokenizer metadata takes from the
// decoder's configuration.
type deepSeekOCRTokenConfig struct {
	Vocabulary uint32          `json:"vocab_size"`
	EOS        json.RawMessage `json:"eos_token_id"`
}

// convertDeepSeekOCR writes a DeepSeek-OCR family checkpoint: the decoder as
// the model and the document encoder as the mmproj, each proved against its
// reader by the adapter before it is written.
func convertDeepSeekOCR(directory string, options Options) (Report, error) {
	repository, err := hfrepo.Open(directory)
	if err != nil {
		return Report{}, err
	}
	defer repository.Close()
	report := Report{}
	if options.OutputPath != "" {
		modelReport, modelErr := writeDeepSeekOCRModel(directory, repository, options)
		if modelErr != nil {
			return report, modelErr
		}
		report = modelReport
	}
	if options.ProjectorPath != "" {
		metadata, tensors, projectorErr := hfgguf.DeepSeekOCRProjectorConversion(repository)
		if projectorErr != nil {
			return report, projectorErr
		}
		// The projector runtime is this package's to reach, not the adapter's:
		// what is about to be written is read by the reader that will consume
		// it before a byte lands.
		if err := requireDeepSeekOCRProjector(metadata, tensors); err != nil {
			return report, err
		}
		slices.SortFunc(tensors, func(left, right gguf.TensorData) int { return strings.Compare(left.Name, right.Name) })
		if err := gguf.WriteFileExclusive(options.ProjectorPath, metadata, tensors, gguf.WriteOptions{}); err != nil {
			return report, err
		}
		report.ProjectorTensors = len(tensors)
	}
	return report, nil
}

// requireDeepSeekOCRProjector reads the mmproj about to be written with the
// projector runtime's own reader: the tensors it names, their shapes and the
// metadata it derives its towers and tiling from.
func requireDeepSeekOCRProjector(metadata []gguf.Metadata, tensors []gguf.TensorData) error {
	infos := make([]gguf.TensorInfo, len(tensors))
	for index, tensor := range tensors {
		info := gguf.TensorInfo{Name: tensor.Name, Type: tensor.Type, Dimensions: uint32(len(tensor.Shape))}
		traits, _ := tensor.Type.Traits()
		elements := uint64(1)
		for axis, extent := range tensor.Shape {
			info.Shape[axis] = extent
			elements *= extent
		}
		info.Size = elements * traits.TypeSize
		infos[index] = info
	}
	catalog, err := gguf.Catalog(metadata, infos)
	if err != nil {
		return err
	}
	if _, err := projector.ReadDeepSeekOCRSpec(catalog); err != nil {
		return fmt.Errorf("HF converter: DeepSeek-OCR projector: %w", err)
	}
	return nil
}

func writeDeepSeekOCRModel(directory string, repository *hfrepo.Repository, options Options) (Report, error) {
	metadata, tensors, err := hfgguf.DeepSeekOCRConversion(repository)
	if err != nil {
		return Report{}, err
	}
	if name := strings.TrimSpace(options.Name); name != "" {
		for index := range metadata {
			if metadata[index].Key == "general.name" {
				metadata[index] = gguf.StringMetadata("general.name", name)
			}
		}
	}
	language, err := hfgguf.DeepSeekOCRLanguageConfig(repository)
	if err != nil {
		return Report{}, err
	}
	encoded, err := json.Marshal(language)
	if err != nil {
		return Report{}, err
	}
	var text deepSeekOCRTokenConfig
	if err := json.Unmarshal(encoded, &text); err != nil {
		return Report{}, fmt.Errorf("HF converter: parse language config: %w", err)
	}
	if text.Vocabulary == 0 {
		return Report{}, errors.New("HF converter: DeepSeek-OCR vocabulary size is missing")
	}
	tokenizerItems, err := tokenizerMetadata(directory, text.Vocabulary, text.EOS)
	if err != nil {
		return Report{}, err
	}
	metadata = append(metadata, tokenizerItems...)
	template, err := chatTemplateMetadata(directory)
	if err != nil {
		return Report{}, err
	}
	metadata = append(metadata, template...)
	slices.SortFunc(tensors, func(left, right gguf.TensorData) int { return strings.Compare(left.Name, right.Name) })
	if err := modelartifact.ValidateGeneratedCatalog(metadata, tensors); err != nil {
		return Report{}, err
	}
	if err := gguf.WriteFileExclusive(options.OutputPath, metadata, tensors, gguf.WriteOptions{}); err != nil {
		return Report{}, err
	}
	written, err := os.Stat(options.OutputPath)
	if err != nil {
		return Report{}, err
	}
	return Report{Tensors: len(tensors), VocabSize: int(text.Vocabulary), OutputBytes: written.Size()}, nil
}
