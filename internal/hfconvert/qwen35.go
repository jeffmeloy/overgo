package hfconvert

// qwen3_5: metadata and tensor mapping delegate to the hfgguf serving adapter
// (one owner per mapping fact); this file adds tokenizer + chat template
// metadata, catalog self-check, and the GGUF write.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/hfgguf"
	"overgo/internal/hfrepo"
	"overgo/internal/jsonfile"
	"overgo/internal/modelartifact"
	"overgo/internal/projector"
)

type qwen35TokenConfig struct {
	Vocabulary uint32          `json:"vocab_size"`
	EOS        json.RawMessage `json:"eos_token_id"`
}

func convertQwen35(directory string, options Options) (Report, error) {
	repository, err := hfrepo.Open(directory)
	if err != nil {
		return Report{}, err
	}
	defer repository.Close()
	report := Report{}
	if options.OutputPath != "" {
		modelReport, modelErr := writeQwen35Model(directory, repository, options)
		if modelErr != nil {
			return report, modelErr
		}
		report = modelReport
	}
	if options.ProjectorPath != "" {
		metadata, tensors, projectorErr := hfgguf.Qwen35ProjectorConversion(repository)
		if projectorErr != nil {
			return report, projectorErr
		}
		if err := gguf.WriteFileExclusive(options.ProjectorPath, metadata, tensors, gguf.WriteOptions{}); err != nil {
			return report, err
		}
		runner, openErr := projector.OpenSession(context.Background(), options.ProjectorPath, projector.OpenOptions{})
		if openErr != nil {
			_ = os.Remove(options.ProjectorPath)
			return report, fmt.Errorf("HF converter: generated Qwen 3.5 projector: %w", openErr)
		}
		if closeErr := runner.Close(); closeErr != nil {
			return report, closeErr
		}
		written, statErr := os.Stat(options.ProjectorPath)
		if statErr != nil {
			return report, statErr
		}
		report.ProjectorTensors = len(tensors)
		report.OutputBytes += written.Size()
	}
	return report, nil
}

func writeQwen35Model(directory string, repository *hfrepo.Repository, options Options) (Report, error) {
	metadata, tensors, err := hfgguf.Qwen35Conversion(repository)
	if err != nil {
		return Report{}, err
	}
	metadata, vocabulary, err := qwen35ModelMetadata(directory, repository, metadata, options.Name)
	if err != nil {
		return Report{}, err
	}
	sort.Slice(tensors, func(i, j int) bool { return tensors[i].Name < tensors[j].Name })
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
	return Report{
		Tensors: len(tensors), VocabSize: int(vocabulary), OutputBytes: written.Size(),
	}, nil
}

// View opens a Hugging Face directory of a text architecture the converter
// writes, a Qwen 3.5 checkpoint or one of a profile (archProfiles), as the
// GGUF Convert would write from it, named name where one is given: the same
// metadata and tensors, their bytes read from the directory's files when
// read, and nothing written (gguf.View). The file's Close closes the
// directory's files.
func View(directory, name string) (*gguf.File, error) {
	directory, err := filepath.Abs(directory)
	if err != nil {
		return nil, err
	}
	var config modelConfig
	if err := jsonfile.Decode(filepath.Join(directory, "config.json"), &config); err != nil {
		return nil, fmt.Errorf("HF converter: config: %w", err)
	}
	if _, profiled := archProfiles[config.ModelType]; profiled {
		return viewProfile(directory, name, config)
	}
	repository, err := hfrepo.Open(directory)
	if err != nil {
		return nil, err
	}
	file, err := viewQwen35(directory, repository, name)
	if err != nil {
		return nil, errors.Join(err, repository.Close())
	}
	return file, nil
}

// Viewed reports whether View serves directory as it is, so its model is
// served from its own files and never converted: a Qwen 3.5 checkpoint, or one
// of a text architecture the converter writes by profile.
func Viewed(directory string) bool {
	var config modelConfig
	if jsonfile.Decode(filepath.Join(directory, "config.json"), &config) != nil {
		return false
	}
	_, profiled := archProfiles[config.ModelType]
	return config.ModelType == "qwen3_5" || profiled
}

// init makes gguf.Open serve a directory as its View, so a model registered
// from its own files opens wherever a converted one does.
func init() {
	gguf.RegisterDirectoryOpener(func(directory string) (*gguf.File, error) { return View(directory, "") })
}

func viewQwen35(directory string, repository *hfrepo.Repository, name string) (*gguf.File, error) {
	metadata, tensors, err := hfgguf.Qwen35View(repository)
	if err != nil {
		return nil, err
	}
	metadata, _, err = qwen35ModelMetadata(directory, repository, metadata, name)
	if err != nil {
		return nil, err
	}
	return viewed(directory, metadata, tensors, repository)
}

// viewed is the GGUF a view of directory serves from metadata and tensors, its tensors ordered by name and held to
// the catalog a conversion is held to, closing closer with it.
func viewed(directory string, metadata []gguf.Metadata, tensors []gguf.ViewTensor, closer io.Closer) (*gguf.File, error) {
	slices.SortFunc(tensors, func(left, right gguf.ViewTensor) int { return strings.Compare(left.Name, right.Name) })
	declared := make([]gguf.TensorData, len(tensors))
	for index, tensor := range tensors {
		declared[index] = gguf.TensorData{Name: tensor.Name, Shape: tensor.Shape, Type: tensor.Type}
	}
	if err := modelartifact.ValidateGeneratedCatalog(metadata, declared); err != nil {
		return nil, err
	}
	return gguf.View(metadata, tensors, gguf.WriteOptions{}, directory, closer)
}

// qwen35ModelMetadata is a converted Qwen 3.5 model's metadata, its language
// model's named name where one is given, then its tokenizer's and its chat
// template's, and its vocabulary's size.
func qwen35ModelMetadata(directory string, repository *hfrepo.Repository, metadata []gguf.Metadata, name string) ([]gguf.Metadata, uint32, error) {
	if name := strings.TrimSpace(name); name != "" {
		for index := range metadata {
			if metadata[index].Key == "general.name" {
				metadata[index] = gguf.StringMetadata("general.name", name)
			}
		}
	}
	var text qwen35TokenConfig
	if err := json.Unmarshal(repository.Config["text_config"], &text); err != nil {
		return nil, 0, fmt.Errorf("HF converter: parse text config: %w", err)
	}
	if text.Vocabulary == 0 {
		return nil, 0, errors.New("HF converter: qwen3_5 vocabulary size is missing")
	}
	tokenizerItems, err := tokenizerMetadata(directory, text.Vocabulary, text.EOS)
	if err != nil {
		return nil, 0, err
	}
	metadata = append(metadata, tokenizerItems...)
	template, err := chatTemplateMetadata(directory)
	if err != nil {
		return nil, 0, err
	}
	return append(metadata, template...), text.Vocabulary, nil
}
