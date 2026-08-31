package main

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
	"overgo/internal/latentimage"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/routedlm"
)

func resolveImageSource(path string) (capabilitySource, error) {
	identity, err := hfrepo.InspectIdentity(path)
	if err != nil {
		return capabilitySource{}, err
	}
	source, err := modelrecipe.ResolveGenerationSource(recipe.TaskImageGen, identity)
	if err != nil {
		return capabilitySource{}, err
	}

	var inventory modelartifact.Inventory
	switch source.Inventory {
	case modelrecipe.SourceInventoryHF:
		inventory, err = modelartifact.FromHFPath(path)
	case modelrecipe.SourceInventoryLatentImage:
		inventory, err = latentImageInventory(path)
	case modelrecipe.SourceInventoryRootSafetensors:
		inventory, err = imageGenInventory(path)
	default:
		return capabilitySource{}, fmt.Errorf("image-gen: unsupported inventory strategy %q", source.Inventory)
	}
	if err != nil {
		return capabilitySource{}, err
	}

	var profileID artifact.ID
	var contents []artifact.Content
	switch source.Profile {
	case recipe.DependencyFlowProfile:
		profile, err := routedlm.InspectFlowProfile(path)
		if err != nil {
			return capabilitySource{}, err
		}
		profileID = profile.ID
		content, err := profile.Content()
		if err != nil {
			return capabilitySource{}, err
		}
		contents = []artifact.Content{content}
	case recipe.DependencyProfile:
		profile, err := latentimage.ResolveProfile(path)
		if err != nil {
			return capabilitySource{}, err
		}
		profileID = profile.ID
		content, err := profile.Content()
		if err != nil {
			return capabilitySource{}, err
		}
		contents = []artifact.Content{content}
	case "":
	default:
		return capabilitySource{}, fmt.Errorf("image-gen: unsupported profile strategy %q", source.Profile)
	}

	bindings, manifests, err := latentImageComponentBindings(source, inventory)
	if err != nil {
		return capabilitySource{}, err
	}
	return capabilitySource{inventory: inventory, manifests: manifests, define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
		definition, err := modelrecipe.GenerationDefinitionWithComponents(source.Prepare, modelID, profileID, bindings)
		return definition, contents, err
	}}, nil
}

// latentImageComponentBindings slots the latent-image stages onto the
// diffusers-layout component groups: prepare on the text encoder,
// integrate on the transformer, decode on the VAE. Each group becomes
// its own content-identified model manifest, so a byte-identical
// component shared across models (a Flux-lineage VAE, for example)
// resolves to the same artifact everywhere. Sources with other
// inventory layouts stay on the composite slot.
func latentImageComponentBindings(
	source modelrecipe.GenerationSourceProfile,
	inventory modelartifact.Inventory,
) ([]modelrecipe.ComponentBinding, []artifact.Manifest, error) {
	if source.Inventory != modelrecipe.SourceInventoryLatentImage {
		return nil, nil, nil
	}
	groups := []struct {
		node      recipe.NodeID
		directory string
	}{
		{"prepare", "text_encoder"},
		{"integrate", "transformer"},
		{"decode", "vae"},
	}
	bindings := make([]modelrecipe.ComponentBinding, 0, len(groups))
	manifests := make([]artifact.Manifest, 0, len(groups))
	for _, group := range groups {
		var members []artifact.Component
		for _, component := range inventory.Manifest.Components {
			if strings.HasPrefix(component.Name, group.directory+"/") {
				members = append(members, component)
			}
		}
		if len(members) == 0 {
			return nil, nil, fmt.Errorf("image-gen: component group %q is empty", group.directory)
		}
		manifest, err := artifact.NewManifest(artifact.KindModel, members)
		if err != nil {
			return nil, nil, fmt.Errorf("image-gen: component group %q: %w", group.directory, err)
		}
		bindings = append(bindings, modelrecipe.ComponentBinding{Node: group.node, Model: manifest.ID})
		manifests = append(manifests, manifest)
	}
	return bindings, manifests, nil
}

func latentImageInventory(path string) (modelartifact.Inventory, error) {
	files := []struct {
		path string
		name string
		role artifact.ComponentRole
	}{
		{"model_index.json", "pipeline/config", artifact.ComponentConfig},
		{"scheduler/scheduler_config.json", "scheduler/config", artifact.ComponentConfig},
		{"text_encoder/config.json", "text_encoder/config", artifact.ComponentConfig},
		{"tokenizer/tokenizer.json", "tokenizer", artifact.ComponentTokenizer},
		{"tokenizer/tokenizer_config.json", "tokenizer/config", artifact.ComponentConfig},
		{"transformer/config.json", "transformer/config", artifact.ComponentConfig},
		{"vae/config.json", "vae/config", artifact.ComponentConfig},
	}
	for _, directory := range []string{"text_encoder", "transformer", "vae"} {
		entries, err := os.ReadDir(filepath.Join(path, directory))
		if err != nil {
			return modelartifact.Inventory{}, err
		}
		var weights []string
		indexName := ""
		for _, entry := range entries {
			if !entry.IsDir() && strings.HasSuffix(strings.ToLower(entry.Name()), ".safetensors") {
				weights = append(weights, entry.Name())
			}
			if !entry.IsDir() && strings.HasSuffix(strings.ToLower(entry.Name()), ".safetensors.index.json") {
				indexName = entry.Name()
			}
		}
		slices.Sort(weights)
		if len(weights) == 0 {
			return modelartifact.Inventory{}, fmt.Errorf("latent image inventory: no %s weights", directory)
		}
		if len(weights) > 1 && indexName == "" {
			return modelartifact.Inventory{}, fmt.Errorf("latent image inventory: %s shards have no index", directory)
		}
		if indexName != "" {
			files = append(files, struct {
				path string
				name string
				role artifact.ComponentRole
			}{filepath.Join(directory, indexName), directory + "/shard-index", artifact.ComponentShardIndex})
		}
		for index, name := range weights {
			logical, role := directory+"/weights", artifact.ComponentWeights
			if len(weights) > 1 {
				logical, role = fmt.Sprintf("%s/weights-%05d", directory, index), artifact.ComponentWeightsShard
			}
			files = append(files, struct {
				path string
				name string
				role artifact.ComponentRole
			}{filepath.Join(directory, name), logical, role})
		}
	}
	specs := make([]modelartifact.FileSpec, len(files))
	for index, file := range files {
		specs[index] = modelartifact.FileSpec{Path: filepath.Join(path, file.path), Name: file.name, Role: file.role}
	}
	return modelartifact.FromFiles(path, specs)
}
