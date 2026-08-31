package trainingworkflow

import (
	"os"
	"path/filepath"
	"strings"

	"encoding/json"
	"errors"

	"overgo/internal/densecausal"
	artifactexport "overgo/internal/export"
	"overgo/internal/gguf"
	"overgo/internal/hfbpe"
	"overgo/internal/tokenizer"
)

// isGGUFModelInput reports whether the model input names one GGUF file.
// GGUF is a different representation of the same weights, so the workflow
// accepts it wherever a safetensors directory is accepted.
func isGGUFModelInput(input string) bool {
	info, err := os.Stat(input)
	return err == nil && !info.IsDir() && strings.EqualFold(filepath.Ext(input), ".gguf")
}

// loadTrainableModel materializes the trainable F32 inventory from either
// model representation through its own loader.
func loadTrainableModel(input string) (*densecausal.Model, error) {
	if isGGUFModelInput(input) {
		return densecausal.LoadGGUF(input)
	}
	return densecausal.Load(input)
}

// checkpointTokenizerGGUF is the metadata-only tokenizer a GGUF-trained
// checkpoint carries, so resume needs no HF tokenizer files.
const checkpointTokenizerGGUF = "tokenizer.gguf"

// loadTrainableEncoder resolves the text encoder beside the weights: the
// HF tokenizer files in a safetensors directory, the tokenizer GGUF
// metadata embeds, or the metadata-only tokenizer a GGUF-trained
// checkpoint staged for resume.
func loadTrainableEncoder(input string) (func(string) ([]int, error), error) {
	tokenizerPath := input
	if !isGGUFModelInput(input) {
		staged := filepath.Join(input, checkpointTokenizerGGUF)
		if _, err := os.Stat(staged); err != nil {
			loaded, err := hfbpe.Load(input)
			if err != nil {
				return nil, err
			}
			return loaded.Encode, nil
		}
		tokenizerPath = staged
	}
	file, err := gguf.Open(tokenizerPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	vocabulary, err := tokenizer.Load(file)
	if err != nil {
		return nil, err
	}
	return func(text string) ([]int, error) {
		ids, err := vocabulary.Encode(text, tokenizer.EncodeOptions{})
		if err != nil {
			return nil, err
		}
		tokens := make([]int, len(ids))
		for index, id := range ids {
			tokens[index] = int(id)
		}
		return tokens, nil
	}, nil
}

// stageCheckpointMetadata stages the files a checkpoint needs for resume
// beside its weights: a safetensors source copies its own config and
// tokenizer files, while a GGUF source synthesizes the config from the
// loaded model facts and extracts the embedded tokenizer as a
// metadata-only GGUF — the checkpoint stays resumable without the
// multi-gigabyte source.
func stageCheckpointMetadata(source, stage string, model *densecausal.Model) error {
	if !isGGUFModelInput(source) {
		return artifactexport.CopyFiles(source, stage, []string{"config.json", "tokenizer.json"})
	}
	_, untied := model.Shapes["lm_head.weight"]
	configuration := map[string]any{
		"num_attention_heads":     model.Dims.Heads,
		"head_dim":                model.Dims.HeadDim,
		"rope_theta":              model.Dims.RopeTheta,
		"rms_norm_eps":            model.Dims.RMSEps,
		"tie_word_embeddings":     !untied,
		"max_position_embeddings": model.Dims.ContextLength,
	}
	encoded, err := json.MarshalIndent(configuration, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(stage, "config.json"), encoded, 0o644); err != nil {
		return err
	}
	file, err := gguf.Open(source)
	if err != nil {
		return err
	}
	defer file.Close()
	metadata := make([]gguf.Metadata, 0, len(file.Metadata))
	for _, entry := range file.Metadata {
		if entry.Key == "general.architecture" || strings.HasPrefix(entry.Key, "tokenizer.") {
			metadata = append(metadata, entry)
		}
	}
	staged, err := os.Create(filepath.Join(stage, checkpointTokenizerGGUF))
	if err != nil {
		return err
	}
	if err := gguf.Write(staged, metadata, nil, gguf.WriteOptions{}); err != nil {
		return errors.Join(err, staged.Close())
	}
	return staged.Close()
}
