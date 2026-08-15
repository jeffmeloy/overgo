package hfconvert

// qwen3_5: metadata and tensor mapping delegate to the hfgguf serving adapter
// (one owner per mapping fact); this file adds tokenizer + chat template
// metadata, catalog self-check, and the GGUF write.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/hfgguf"
	"overgo/internal/hfrepo"
	"overgo/internal/modelartifact"
	"overgo/internal/projector"
)

type qwen35TokenConfig struct {
	Vocabulary uint32 `json:"vocab_size"`
	EOS        *int   `json:"eos_token_id"`
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
		runner, openErr := projector.OpenAs[projector.ImageProjector](context.Background(), options.ProjectorPath, projector.OpenOptions{})
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
	if name := strings.TrimSpace(options.Name); name != "" {
		for index := range metadata {
			if metadata[index].Key == "general.name" {
				metadata[index] = gguf.StringMetadata("general.name", name)
			}
		}
	}
	var text qwen35TokenConfig
	if err := json.Unmarshal(repository.Config["text_config"], &text); err != nil {
		return Report{}, fmt.Errorf("HF converter: parse text config: %w", err)
	}
	if text.Vocabulary == 0 {
		return Report{}, errors.New("HF converter: qwen3_5 vocabulary size is missing")
	}
	// config.json eos_token_id is the only EOS source for this repo layout
	// (no generation_config.json / special_tokens_map.json companions).
	fallback := noSpecialTokens()
	if text.EOS != nil && *text.EOS >= 0 && *text.EOS < int(text.Vocabulary) {
		fallback.eos = *text.EOS
	}
	tokenizerItems, err := tokenizerMetadata(directory, text.Vocabulary, fallback)
	if err != nil {
		return Report{}, err
	}
	metadata = append(metadata, tokenizerItems...)
	template, err := chatTemplateMetadata(directory)
	if err != nil {
		return Report{}, err
	}
	metadata = append(metadata, template...)
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
		Tensors: len(tensors), VocabSize: int(text.Vocabulary), OutputBytes: written.Size(),
	}, nil
}

// chatTemplateMetadata: embed chat_template.jinja when present (llama.cpp key).
func chatTemplateMetadata(directory string) ([]gguf.Metadata, error) {
	encoded, err := os.ReadFile(filepath.Join(directory, "chat_template.jinja"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("HF converter: read chat template: %w", err)
	}
	return []gguf.Metadata{gguf.StringMetadata("tokenizer.chat_template", string(encoded))}, nil
}
