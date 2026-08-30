package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/gguf"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
)

// routeCatalog is the committed architecture-to-trainer declaration set:
// model facts stay data, and the lane substitutes only recorded facts
// into each declared bounded invocation.
type routeCatalog struct {
	Version  int                         `json:"version"`
	Doctrine string                      `json:"doctrine"`
	Routes   map[string]routeDeclaration `json:"routes"`
}

type routeDeclaration struct {
	Trainer string   `json:"trainer"`
	Argv    []string `json:"argv"`
}

// trainerRoute is one resolved bounded training invocation: the route
// name reports which trainer owns the step, and the argv is the fully
// substituted command the process owner runs.
type trainerRoute struct {
	Name string
	Argv []string
}

const routeCatalogVersion = 1

func loadRouteCatalog(path string) (routeCatalog, error) {
	var catalog routeCatalog
	if err := jsonfile.Decode(path, &catalog); err != nil {
		return routeCatalog{}, err
	}
	if catalog.Version != routeCatalogVersion {
		return routeCatalog{}, fmt.Errorf("train-lane: route catalog %s declares version %d, this lane consumes %d", path, catalog.Version, routeCatalogVersion)
	}
	for key, route := range catalog.Routes {
		if len(route.Argv) == 0 || route.Trainer == "" {
			return routeCatalog{}, fmt.Errorf("train-lane: route %q declares no trainer invocation", key)
		}
	}
	return catalog, nil
}

// routeKey derives the declaration key from the model input's own
// declaration: a GGUF names its architecture, a directory names its
// model type, and a bare checkpoint file names its form.
func routeKey(input string) (string, error) {
	info, err := os.Stat(input)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		modelType, err := directoryModelType(input)
		if err != nil {
			return "", err
		}
		return "dir:" + modelType, nil
	}
	switch strings.ToLower(filepath.Ext(input)) {
	case ".gguf":
		architecture, err := ggufArchitecture(input)
		if err != nil {
			return "", err
		}
		return "gguf:" + architecture, nil
	case ".pt":
		return "file:pt", nil
	}
	return "", fmt.Errorf("train-lane: %s declares no routable representation", input)
}

func ggufArchitecture(path string) (string, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	value, found := file.MetadataValue("general.architecture")
	if !found {
		return "", fmt.Errorf("train-lane: %s declares no general.architecture", path)
	}
	architecture, ok := value.Data.(string)
	if !ok || architecture == "" {
		return "", fmt.Errorf("train-lane: %s general.architecture is not a name", path)
	}
	return architecture, nil
}

func directoryModelType(directory string) (string, error) {
	var config struct {
		ModelType string `json:"model_type"`
	}
	if err := jsonfile.Decode(filepath.Join(directory, "config.json"), &config); err != nil {
		return "", fmt.Errorf("train-lane: %s holds no readable model declaration: %v", directory, err)
	}
	if config.ModelType == "" {
		return "", fmt.Errorf("train-lane: %s declares no model_type", directory)
	}
	return config.ModelType, nil
}

// resolveRoute maps one trainable input to its trainer: a declared route
// when the input's architecture has one, the dense workflow when the
// input is a dense representation, and a named refusal otherwise. The
// dense argv comes from the caller because its bounds are lane flags.
func resolveRoute(catalog routeCatalog, input, storePath, t2vDir string, dense []string) (trainerRoute, error) {
	key, err := routeKey(input)
	if err != nil {
		return trainerRoute{}, err
	}
	declaration, declared := catalog.Routes[key]
	if !declared {
		return trainerRoute{Name: "dense", Argv: dense}, nil
	}
	argv := make([]string, len(declaration.Argv))
	for index, argument := range declaration.Argv {
		substituted := strings.ReplaceAll(argument, "{input}", input)
		substituted = strings.ReplaceAll(substituted, "{store}", storePath)
		if strings.Contains(substituted, "{t2v-dir}") {
			if t2vDir == "" {
				return trainerRoute{}, fmt.Errorf("train-lane: route %q needs the recorded t2v stack directory and none resolves", key)
			}
			substituted = strings.ReplaceAll(substituted, "{t2v-dir}", t2vDir)
		}
		argv[index] = substituted
	}
	return trainerRoute{Name: key, Argv: argv}, nil
}

// resolveT2VDirectory finds the unique recorded t2v video-stack
// directory among the catalogued models: the conditioning stack a raw
// DiT checkpoint trains against. Zero or several candidates resolve to
// empty; a route that needs the stack then refuses by name.
func resolveT2VDirectory(ctx context.Context, storePath string, entries []discovery.CatalogEntry) string {
	reader, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		return ""
	}
	defer reader.Close()
	var candidates []string
	for _, entry := range entries {
		directory, err := artifact.AvailablePath(ctx, reader, entry.Model, artifact.LocationDirectory)
		if err != nil {
			continue
		}
		if modelType, err := directoryModelType(directory); err == nil && modelType == "t2v" {
			candidates = append(candidates, directory)
		}
	}
	if len(candidates) != 1 {
		return ""
	}
	return candidates[0]
}
