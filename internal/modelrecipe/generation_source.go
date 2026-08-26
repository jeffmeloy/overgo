package modelrecipe

import (
	_ "embed"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

//go:embed generation_source_profiles.json
var generationSourceProfilesJSON []byte

var generationSources = mustGenerationSources()

type SourceInventory string

const (
	SourceInventoryHF              SourceInventory = "hf"
	SourceInventoryLatentImage     SourceInventory = "latent-image"
	SourceInventoryRootSafetensors SourceInventory = "root-safetensors"
)

type SourceSelector struct {
	ModelType    string `json:"model_type,omitempty"`
	Architecture string `json:"architecture,omitempty"`
	Pipeline     string `json:"pipeline,omitempty"`
}

// GenerationSourceProfile binds artifact identity to recipe topology.
type GenerationSourceProfile struct {
	Task      recipe.Task           `json:"task"`
	Selector  SourceSelector        `json:"selector"`
	Inventory SourceInventory       `json:"inventory"`
	Profile   recipe.DependencyRole `json:"profile"`
	Prepare   recipe.ModuleID       `json:"prepare"`
}

type generationSourceCatalog struct {
	Version  uint16                    `json:"version"`
	Profiles []GenerationSourceProfile `json:"profiles"`
}

// ResolveGenerationSource selects one declared task profile.
func ResolveGenerationSource(task recipe.Task, identity hfrepo.Identity) (GenerationSourceProfile, error) {
	var found *GenerationSourceProfile
	for index := range generationSources {
		profile := &generationSources[index]
		if profile.Task != task || !profile.Selector.matches(identity) {
			continue
		}
		if found != nil {
			return GenerationSourceProfile{}, fmt.Errorf("model recipe: ambiguous %s source profile", task)
		}
		found = profile
	}
	if found == nil {
		return GenerationSourceProfile{}, fmt.Errorf("model recipe: artifact has no registered %s source profile", task)
	}
	return *found, nil
}

// GenerationDefinition compiles one declared linear generation topology.
func GenerationDefinition(prepare recipe.ModuleID, modelID, profileID artifact.ID) (recipe.Definition, error) {
	capability, ok := generationCapabilities[prepare]
	if !ok {
		return recipe.Definition{}, fmt.Errorf("model recipe: unknown generation module %q", prepare)
	}
	if capability.profile == "" {
		if profileID.Valid() {
			return recipe.Definition{}, errors.New("model recipe: profile-free source has a profile")
		}
		return capability.topology.definition(capability.task, modelID)
	}
	if !profileID.Valid() {
		return recipe.Definition{}, errors.New("model recipe: source profile is absent")
	}
	return capability.topology.definition(capability.task, modelID, recipe.Dependency{Role: capability.profile, Artifact: profileID})
}

func (s SourceSelector) matches(identity hfrepo.Identity) bool {
	return (s.ModelType == "" || s.ModelType == identity.ModelType) &&
		(s.Architecture == "" || slices.Contains(identity.Architectures, s.Architecture)) &&
		(s.Pipeline == "" || s.Pipeline == identity.Pipeline)
}

type generationSourceBinding struct {
	task     recipe.Task
	topology linearCapability
	profile  recipe.DependencyRole
}

var generationCapabilities = map[recipe.ModuleID]generationSourceBinding{
	ModuleRoutedImagePrepare:     {recipe.TaskImageGen, routedImageCapability, recipe.DependencyFlowProfile},
	ModuleLatentImagePrepare:     {recipe.TaskImageGen, latentImageCapability, recipe.DependencyProfile},
	ModuleDiffusionImagePrepare:  {recipe.TaskImageGen, diffusionImageCapability, ""},
	ModuleOscillatorImagePrepare: {recipe.TaskImageGen, oscillatorImageCapability, ""},
	ModuleOscillatorVideoPrepare: {recipe.TaskVideoGen, oscillatorVideoCapability, ""},
	ModuleLatentVideoPrepare:     {recipe.TaskVideoGen, latentVideoCapability, recipe.DependencyProfile},
}

func mustGenerationSources() []GenerationSourceProfile {
	var catalog generationSourceCatalog
	if err := strictjson.DecodeBytes(generationSourceProfilesJSON, &catalog); err != nil {
		panic(fmt.Errorf("model recipe: decode generation source profiles: %w", err))
	}
	if catalog.Version != 1 || len(catalog.Profiles) == 0 {
		panic("model recipe: invalid generation source catalog")
	}
	for _, profile := range catalog.Profiles {
		binding, ok := generationCapabilities[profile.Prepare]
		if !ok || binding.task != profile.Task || profile.Selector == (SourceSelector{}) || !slices.Contains(
			[]SourceInventory{SourceInventoryHF, SourceInventoryLatentImage, SourceInventoryRootSafetensors}, profile.Inventory,
		) || profile.Profile != binding.profile {
			panic("model recipe: invalid generation source profile")
		}
	}
	return catalog.Profiles
}
