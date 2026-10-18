package modelartifact

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/pathidentity"
	"overgo/internal/strictjson"
)

// pipelineComponentFields is the library/class pair in Diffusers model_index.json.
// Reconsider it if the upstream component declaration format changes.
const pipelineComponentFields = 2

// fromHFPipeline inventories declared components without importing their source.
// A config.json component requires safetensors and every shard index is checked by FromFiles.
func fromHFPipeline(directory string) (Inventory, error) {
	index, err := pipelineConfig(filepath.Join(directory, "model_index.json"))
	if err != nil {
		return Inventory{}, err
	}
	var class string
	if err := strictjson.DecodeBytes(index["_class_name"], &class); err != nil || class == "" {
		return Inventory{}, errors.New("model artifact: pipeline requires a nonempty _class_name")
	}
	files := map[string]artifact.ComponentRole{"model_index.json": artifact.ComponentConfig}
	components := 0
	for name, encoded := range index {
		if strings.HasPrefix(name, "_") {
			continue
		}
		var value any
		if err := strictjson.DecodeBytes(encoded, &value); err != nil {
			return Inventory{}, err
		}
		parameters, isArray := value.([]any)
		if !isArray {
			continue
		}
		componentPair := false
		for _, parameter := range parameters {
			if parameter == nil {
				componentPair = true
			}
			if _, ok := parameter.(string); ok {
				componentPair = true
			}
		}
		if !componentPair {
			continue
		}
		if name == "" || name == "." || name == ".." || strings.ContainsAny(name, "/\\") || !filepath.IsLocal(name) {
			return Inventory{}, fmt.Errorf("model artifact: invalid pipeline component %q", name)
		}
		var pair []*string
		if err := strictjson.DecodeBytes(encoded, &pair); err != nil || len(pair) != pipelineComponentFields {
			return Inventory{}, fmt.Errorf("model artifact: pipeline component %q requires a library/class pair", name)
		}
		if pair[0] == nil && pair[1] == nil {
			continue
		}
		if pair[0] == nil || pair[1] == nil || *pair[0] == "" || *pair[1] == "" {
			return Inventory{}, fmt.Errorf("model artifact: incomplete pipeline component %q", name)
		}
		componentDir := filepath.Join(directory, name)
		contained, err := pathidentity.Contains(directory, componentDir)
		if err != nil || !contained {
			return Inventory{}, fmt.Errorf("model artifact: pipeline component %q escapes its root", name)
		}
		entries, err := os.ReadDir(componentDir)
		if err != nil {
			return Inventory{}, err
		}
		var weights []string
		hasConfig, hasPolicy := false, false
		for _, entry := range entries {
			base := entry.Name()
			relative := filepath.ToSlash(filepath.Join(name, base))
			role, _, _ := companionContract(base)
			switch {
			case base == "config.json" || base == "scheduler_config.json" || base == "tokenizer_config.json":
				if _, err := pipelineConfig(filepath.Join(componentDir, base)); err != nil {
					return Inventory{}, err
				}
				hasConfig = hasConfig || base == "config.json"
				hasPolicy = true
				if base != "tokenizer_config.json" {
					role = artifact.ComponentConfig
				}
			case strings.HasSuffix(base, ".safetensors.index.json"):
				role = artifact.ComponentShardIndex
			case strings.HasSuffix(base, ".safetensors"):
				weights = append(weights, relative)
				continue
			case role == artifact.ComponentCompanion:
				continue
			}
			files[relative] = role
		}
		if !hasPolicy || (hasConfig && len(weights) == 0) {
			return Inventory{}, fmt.Errorf("model artifact: component %q requires configuration and model components require safetensors weights", name)
		}
		for _, weight := range weights {
			role := artifact.ComponentWeights
			if len(weights) > 1 {
				role = artifact.ComponentWeightsShard
			}
			files[weight] = role
		}
		module := *pair[0]
		if strings.ContainsAny(module, "/\\") || strings.HasPrefix(module, ".") {
			return Inventory{}, fmt.Errorf("model artifact: invalid component module %q", module)
		}
		source := module + ".py"
		if info, err := os.Stat(filepath.Join(directory, source)); err == nil && !info.IsDir() {
			files[source] = artifact.ComponentCompanion
		} else if err != nil && !errors.Is(err, os.ErrNotExist) {
			return Inventory{}, err
		}
		components++
	}
	if components == 0 {
		return Inventory{}, errors.New("model artifact: pipeline declares no present components")
	}
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	slices.Sort(names)
	specs := make([]FileSpec, 0, len(names))
	for _, name := range names {
		specs = append(specs, FileSpec{Path: filepath.Join(directory, filepath.FromSlash(name)), Name: name, Role: files[name]})
	}
	return FromFiles(directory, specs)
}

func pipelineConfig(path string) (map[string]json.RawMessage, error) {
	data, err := artifact.ReadContentFile(path)
	if err != nil {
		return nil, err
	}
	var config map[string]json.RawMessage
	if err := strictjson.DecodeBytes(data, &config); err != nil {
		return nil, fmt.Errorf("model artifact: pipeline config %s: %w", path, err)
	}
	if config == nil {
		return nil, fmt.Errorf("model artifact: pipeline config %s must be an object", path)
	}
	return config, nil
}
