package main

import (
	"cmp"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

// The inventory starts from the model directories on disk, where the
// validation plan starts from the registered models, so a directory nothing
// has registered is reported instead of being absent from every listing.

// Directory standings, from least to most complete.
const (
	standingNotAModel    = "not-a-model"
	standingUnregistered = "unregistered"
	standingRegistered   = "registered"
	standingValidated    = "validated"
)

// How a registered model is tied to a directory.
const (
	linkLocation  = "location"
	linkConverted = "converted-from"
)

// weightExtensions mark a file that holds model weights.
var weightExtensions = []string{".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt"}

// modelConfigFile declares a checkpoint's architecture.
const modelConfigFile = "config.json"

// registrationSuffix names the declaration recipe spec writes for a directory.
const registrationSuffix = "-registration.json"

// modelDirectory is what the inventory reads from one directory on disk.
type modelDirectory struct {
	Path          string
	Weights       bool
	ModelType     string
	ConfigDigests []string
}

// InventoryModel is one registered model tied to a directory and the
// validation cells its declared functions derive.
type InventoryModel struct {
	Model    artifact.ID       `json:"model"`
	Location string            `json:"location"`
	Link     string            `json:"link"`
	Standing string            `json:"standing"`
	Cells    []ModelValidation `json:"cells"`
}

// DirectoryInventory is one model directory's standing.
type DirectoryInventory struct {
	Directory string           `json:"directory"`
	Standing  string           `json:"standing"`
	ModelType string           `json:"model_type,omitzero"`
	Supported bool             `json:"recipe_supported"`
	Models    []InventoryModel `json:"models,omitempty"`
	Commands  []string         `json:"commands,omitempty"`
}

// readModelDirectory reads the facts of one directory: whether it holds
// weights, the architecture its config declares, and the digests of the config
// files a conversion records -- the content by which a converted model is
// tied back to the directory it was converted from, wherever its file lives.
func readModelDirectory(path string) (modelDirectory, error) {
	directory := modelDirectory{Path: path}
	err := filepath.WalkDir(path, func(_ string, entry os.DirEntry, walkErr error) error {
		if walkErr == nil && !entry.IsDir() && slices.Contains(weightExtensions, strings.ToLower(filepath.Ext(entry.Name()))) {
			directory.Weights = true
			return filepath.SkipAll
		}
		return walkErr
	})
	if err != nil {
		return modelDirectory{}, err
	}
	// A directory whose config is absent or malformed still has a standing:
	// it declares no architecture and is tied to nothing by content.
	if _, _, sources, err := modelartifact.ReadModelConfigComponents(path); err == nil {
		for _, source := range sources {
			directory.ConfigDigests = append(directory.ConfigDigests, source.SHA256)
		}
	}
	if data, err := os.ReadFile(filepath.Join(path, modelConfigFile)); err == nil {
		var declared struct {
			ModelType string `json:"model_type"`
		}
		if json.Unmarshal(data, &declared) == nil {
			directory.ModelType = declared.ModelType
		}
	}
	return directory, nil
}

// convertedSources maps each model that carries a model-config declaration to
// the digests of the config files it was extracted from.
func convertedSources(ctx context.Context, store overgodb.DocumentReader) (map[artifact.ID][]string, error) {
	sources := map[artifact.ID][]string{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindProfile, MediaType: modelartifact.ModelConfigMediaType, Schema: modelartifact.ModelConfigSchema,
		}},
		Order: overgodb.DocumentNewestFirst,
	}, modelartifact.ParseModelConfigDocument, func(_ overgodb.DocumentView, document modelartifact.ModelConfigDocument) error {
		for _, source := range document.Sources {
			sources[document.Model] = append(sources[document.Model], source.SHA256)
		}
		return nil
	})
	return sources, err
}

// buildInventory ties the registered cells to the directories and states each
// directory's standing. A model is tied by its location beneath the directory,
// or by a recorded config file whose content is the directory's own; never by
// a name. A model is validated when every one of its cells carries accepted
// evidence and none is stale; a directory takes the standing of its best
// model, because a superseded artifact left beside a validated one -- a
// conversion from before a fix -- does not make the model any less validated.
// Every model that is not validated keeps its command.
func buildInventory(directories []modelDirectory, cells []ModelValidation, converted map[artifact.ID][]string) []DirectoryInventory {
	inventory := make([]DirectoryInventory, 0, len(directories))
	for _, directory := range directories {
		entry := DirectoryInventory{
			Directory: filepath.ToSlash(directory.Path), ModelType: directory.ModelType,
			Supported: slices.Contains(model.SupportedArchitectures(), directory.ModelType),
		}
		prefix := strings.ToLower(filepath.ToSlash(directory.Path)) + "/"
		tied := map[artifact.ID]int{}
		for _, cell := range cells {
			link := ""
			switch {
			case strings.HasPrefix(strings.ToLower(filepath.ToSlash(cell.Location)), prefix):
				link = linkLocation
			case slices.ContainsFunc(converted[cell.Model], func(digest string) bool { return slices.Contains(directory.ConfigDigests, digest) }):
				link = linkConverted
			default:
				continue
			}
			index, known := tied[cell.Model]
			if !known {
				index = len(entry.Models)
				tied[cell.Model] = index
				entry.Models = append(entry.Models, InventoryModel{Model: cell.Model, Location: cell.Location, Link: link})
			}
			entry.Models[index].Cells = append(entry.Models[index].Cells, cell)
		}
		entry.Standing, entry.Commands = directoryStanding(directory, entry.Models)
		inventory = append(inventory, entry)
	}
	slices.SortFunc(inventory, func(left, right DirectoryInventory) int { return cmp.Compare(left.Directory, right.Directory) })
	return inventory
}

// directoryStanding states the standing and the commands that would advance
// it: registering an unregistered directory, re-acquiring a cell with no
// accepted evidence. It states each model's own standing in place.
func directoryStanding(directory modelDirectory, models []InventoryModel) (string, []string) {
	if len(models) == 0 {
		if !directory.Weights {
			return standingNotAModel, nil
		}
		root, name := filepath.Split(filepath.ToSlash(directory.Path))
		specification := name + registrationSuffix
		return standingUnregistered, []string{
			"go run ./cmd/recipe spec -root " + root + " -output " + specification + " " + name,
			"go run ./cmd/recipe register -repo STORE -root " + root + " -spec " + specification,
		}
	}
	var commands []string
	standing := standingRegistered
	for index, tied := range models {
		models[index].Standing = standingValidated
		for _, cell := range tied.Cells {
			if cell.Evidence.Valid() && cell.Stale == "" {
				continue
			}
			models[index].Standing = standingRegistered
			command := "go run ./cmd/recipe verify " + tied.Model.String()
			if name, args, err := reacquireCommand(cell); err == nil {
				command = name + " " + strings.Join(args, " ")
			}
			if !slices.Contains(commands, command) {
				commands = append(commands, command)
			}
		}
		if models[index].Standing == standingValidated {
			standing = standingValidated
		}
	}
	return standing, commands
}

// modelDirectories reads every immediate subdirectory of the roots.
func modelDirectories(roots []string) ([]modelDirectory, error) {
	var directories []modelDirectory
	for _, root := range roots {
		entries, err := os.ReadDir(root)
		if err != nil {
			return nil, err
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			directory, err := readModelDirectory(filepath.Join(root, entry.Name()))
			if err != nil {
				return nil, err
			}
			directories = append(directories, directory)
		}
	}
	return directories, nil
}
