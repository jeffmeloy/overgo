package modelintake

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/strictjson"
)

// importPolicy binds upstream declarations without granting execution authority.
type importPolicy struct {
	Source     artifact.ID                `json:"source"`
	Name       string                     `json:"name"`
	Role       artifact.ComponentRole     `json:"role"`
	Parameters map[string]json.RawMessage `json:"parameters,omitempty"`
}

func componentImportPolicies(inventory modelartifact.Inventory, name string) ([]importPolicy, error) {
	paths := map[artifact.ID]string{}
	for _, location := range inventory.Locations {
		if location.Kind == artifact.LocationFile {
			paths[location.Artifact] = location.Value
		}
	}
	var policies []importPolicy
	for _, source := range inventory.Manifest.Components {
		local := source.Name
		if name != "." {
			var ok bool
			local, ok = strings.CutPrefix(local, name+"/")
			if !ok {
				continue
			}
		} else if strings.Contains(local, "/") {
			continue
		}
		switch source.Role {
		case artifact.ComponentConfig, artifact.ComponentTokenizer, artifact.ComponentGenerationConfig, artifact.ComponentChatTemplate, artifact.ComponentPreprocessor:
		default:
			continue
		}
		policy := importPolicy{Source: source.Artifact, Name: local, Role: source.Role}
		switch filepath.Base(local) {
		case "config", "config.json", "scheduler_config.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "processor_config.json", "preprocessor_config.json":
			encoded, err := artifact.ReadContentFile(paths[source.Artifact])
			if err != nil {
				return nil, err
			}
			id, err := artifact.IdentifyBytes(source.Artifact.Kind(), encoded)
			if err != nil {
				return nil, err
			}
			if id != source.Artifact {
				return nil, fmt.Errorf("model intake: policy %q changed after source inventory", source.Name)
			}
			if err := strictjson.DecodeBytes(encoded, &policy.Parameters); err != nil {
				return nil, fmt.Errorf("model intake: invalid policy %q: %w", source.Name, err)
			}
			if policy.Parameters == nil {
				return nil, fmt.Errorf("model intake: policy %q must be an object", source.Name)
			}
		}
		policies = append(policies, policy)
	}
	return policies, nil
}
