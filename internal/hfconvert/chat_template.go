package hfconvert

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/strictjson"
)

// chatTemplateMetadata preserves model-declared prompt bytes. Hugging Face
// standalone templates override the corresponding legacy config entries.
func chatTemplateMetadata(directory string) ([]gguf.Metadata, error) {
	var config struct {
		ChatTemplate json.RawMessage `json:"chat_template"`
	}
	if err := readOptionalJSON(filepath.Join(directory, "tokenizer_config.json"), &config); err != nil {
		return nil, err
	}
	templates := make(map[string]string)
	if strictjson.HasValue(config.ChatTemplate) {
		var source string
		if err := json.Unmarshal(config.ChatTemplate, &source); err == nil {
			templates["default"] = source
		} else {
			var named []struct {
				Name     string  `json:"name"`
				Template *string `json:"template"`
			}
			if err := json.Unmarshal(config.ChatTemplate, &named); err != nil {
				return nil, fmt.Errorf("HF converter: chat_template must be a string or named template list: %w", err)
			}
			for _, entry := range named {
				if entry.Name == "" || strings.TrimSpace(entry.Name) != entry.Name || entry.Template == nil {
					return nil, errors.New("HF converter: named chat template requires a name and template string")
				}
				if _, exists := templates[entry.Name]; exists {
					return nil, fmt.Errorf("HF converter: duplicate chat template %q", entry.Name)
				}
				templates[entry.Name] = *entry.Template
			}
		}
	}
	encoded, err := os.ReadFile(filepath.Join(directory, "chat_template.jinja"))
	if err == nil {
		templates["default"] = string(encoded)
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("HF converter: read chat template: %w", err)
	}
	if len(templates) == 0 {
		if strictjson.HasValue(config.ChatTemplate) {
			return nil, errors.New("HF converter: chat_template list is empty")
		}
		return nil, nil
	}
	if _, ok := templates["default"]; !ok {
		return nil, errors.New("HF converter: named chat templates require a default for automatic chat formatting")
	}
	names := make([]string, 0, len(templates))
	for name, source := range templates {
		if strings.TrimSpace(source) == "" {
			return nil, fmt.Errorf("HF converter: chat template %q is empty", name)
		}
		names = append(names, name)
	}
	slices.Sort(names)
	metadata := make([]gguf.Metadata, 0, len(names))
	var variants []string
	for _, name := range names {
		key := "tokenizer.chat_template"
		if name != "default" {
			key += "." + name
			variants = append(variants, name)
		}
		metadata = append(metadata, gguf.StringMetadata(key, templates[name]))
	}
	if len(variants) != 0 {
		metadata = append(metadata, gguf.ArrayMetadata("tokenizer.chat_templates", gguf.ValueTypeString, variants))
	}
	return metadata, nil
}
