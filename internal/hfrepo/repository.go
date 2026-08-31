package hfrepo

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
)

const maxConfigBytes = 16 << 20

var companionNames = []string{
	"added_tokens.json",
	"chat_template.jinja",
	"generation_config.json",
	"merges.txt",
	"preprocessor_config.json",
	"processor_config.json",
	"special_tokens_map.json",
	"spiece.model",
	"tokenizer.json",
	"tokenizer.model",
	"tokenizer_config.json",
	"video_preprocessor_config.json",
	"vocab.json",
}

// Identity: Hugging Face architecture selectors.
type Identity struct {
	ModelType     string
	TextModelType string
	Architectures []string
	Pipeline      string
}

// InspectIdentity reads available root selectors without opening tensors.
func InspectIdentity(directory string) (Identity, error) {
	_, identity, err := readConfig(filepath.Join(directory, "config.json"))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return Identity{}, err
	}
	file, err := os.Open(filepath.Join(directory, "model_index.json"))
	if errors.Is(err, os.ErrNotExist) {
		return identity, nil
	}
	if err != nil {
		return Identity{}, err
	}
	defer file.Close()
	var index map[string]json.RawMessage
	if err := strictjson.DecodeBounded(file, maxConfigBytes, &index); err != nil {
		return Identity{}, fmt.Errorf("model repository: parse model index: %w", err)
	}
	if err := decodeOptional(index, "_class_name", &identity.Pipeline); err != nil {
		return Identity{}, err
	}
	return identity, nil
}

// Repository: config, companions, and lazy tensor catalog.
type Repository struct {
	Directory  string
	Config     map[string]json.RawMessage
	Identity   Identity
	Companions []string
	Tensors    *safetensors.Source
}

// Open: open one Hugging Face model repository.
func Open(directory string) (*Repository, error) {
	absolute, err := filepath.Abs(directory)
	if err != nil {
		return nil, fmt.Errorf("model repository: resolve directory: %w", err)
	}
	config, identity, err := readConfig(filepath.Join(absolute, "config.json"))
	if err != nil {
		return nil, err
	}
	source, err := safetensors.OpenSource(absolute)
	if err != nil {
		return nil, err
	}
	companions, err := companionFiles(absolute)
	if err != nil {
		_ = source.Close()
		return nil, err
	}
	return &Repository{
		Directory: absolute, Config: config, Identity: identity,
		Companions: companions, Tensors: source,
	}, nil
}

// Close: release repository shards.
func (r *Repository) Close() error {
	if r == nil {
		return nil
	}
	return r.Tensors.Close()
}

func readConfig(path string) (map[string]json.RawMessage, Identity, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, Identity{}, fmt.Errorf("model repository: open config: %w", err)
	}
	defer file.Close()
	var config map[string]json.RawMessage
	if err := strictjson.DecodeBounded(file, maxConfigBytes, &config); err != nil {
		return nil, Identity{}, fmt.Errorf("model repository: parse config: %w", err)
	}
	identity := Identity{}
	if err := decodeOptional(config, "model_type", &identity.ModelType); err != nil {
		return nil, Identity{}, err
	}
	if err := decodeOptional(config, "architectures", &identity.Architectures); err != nil {
		return nil, Identity{}, err
	}
	var text map[string]json.RawMessage
	if err := decodeOptional(config, "text_config", &text); err != nil {
		return nil, Identity{}, err
	}
	if err := decodeOptional(text, "model_type", &identity.TextModelType); err != nil {
		return nil, Identity{}, err
	}
	return config, identity, nil
}

func decodeOptional(values map[string]json.RawMessage, key string, destination any) error {
	if len(values) == 0 {
		return nil
	}
	encoded, ok := values[key]
	if !ok || !strictjson.HasValue(encoded) {
		return nil
	}
	if err := strictjson.DecodeBytes(encoded, destination); err != nil {
		return fmt.Errorf("model repository: config field %q: %w", key, err)
	}
	return nil
}

func companionFiles(directory string) ([]string, error) {
	files := make([]string, 0, len(companionNames))
	for _, name := range companionNames {
		info, err := os.Stat(filepath.Join(directory, name))
		if err == nil && !info.IsDir() {
			files = append(files, name)
		} else if err != nil && !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("model repository: inspect companion %s: %w", name, err)
		}
	}
	slices.Sort(files)
	return files, nil
}
