package mediacapability

import (
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/hfrepo"
	"overgo/internal/latentimage"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/rae"
	"overgo/internal/recipe"
	"overgo/internal/routedlm"
)

func resolveImageSource(path string) (Source, error) {
	collection, held, err := rae.ResolveCollection(path)
	if err != nil {
		return Source{}, err
	}
	if held {
		return raeImageSource(path, collection)
	}
	identity, err := hfrepo.InspectIdentity(path)
	if err != nil {
		return Source{}, err
	}
	source, err := modelrecipe.ResolveGenerationSource(recipe.TaskImageGen, identity)
	if err != nil {
		return Source{}, err
	}

	var inventory modelartifact.Inventory
	switch source.Inventory {
	case modelrecipe.SourceInventoryHF:
		inventory, err = modelartifact.FromHFPath(path)
	case modelrecipe.SourceInventoryRootSafetensors:
		inventory, err = imageGenInventory(path)
	default:
		return Source{}, fmt.Errorf("image-gen: unsupported inventory strategy %q", source.Inventory)
	}
	if err != nil {
		return Source{}, err
	}

	var profileID artifact.ID
	var contents []artifact.Content
	switch source.Profile {
	case recipe.DependencyFlowProfile:
		profile, err := routedlm.InspectFlowProfile(path)
		if err != nil {
			return Source{}, err
		}
		profileID = profile.ID
		content, err := profile.Content()
		if err != nil {
			return Source{}, err
		}
		contents = []artifact.Content{content}
	case recipe.DependencyProfile:
		profile, err := latentimage.ResolveProfile(path)
		if err != nil {
			return Source{}, err
		}
		profileID = profile.ID
		content, err := profile.Content()
		if err != nil {
			return Source{}, err
		}
		contents = []artifact.Content{content}
	case "":
	default:
		return Source{}, fmt.Errorf("image-gen: unsupported profile strategy %q", source.Profile)
	}

	bindings, manifests, err := latentImageComponentBindings(source, inventory)
	if err != nil {
		return Source{}, err
	}
	return Source{Inventory: inventory, Manifests: manifests, Define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
		definition, err := modelrecipe.GenerationDefinitionWithComponents(source.Prepare, modelID, profileID, bindings)
		return definition, contents, err
	}}, nil
}

// raeImageSource is an RAE collection as the image-gen capability serves
// it: the DiT checkpoint, the decoder and the latent statistics its
// reviewed sampler reads, on the profile-free RAE topology. A collection
// carries no Hugging Face configuration, so it is recognized by the
// sampler's checkpoint rather than by a source profile's selector.
func raeImageSource(path string, collection rae.Collection) (Source, error) {
	inventory, err := modelartifact.FromFiles(path, []modelartifact.FileSpec{
		{Path: collection.Checkpoint, Name: "dit", Role: artifact.ComponentWeights},
		{Path: collection.Decoder, Name: "decoder", Role: artifact.ComponentWeights},
		{Path: collection.Statistics, Name: "statistics", Role: artifact.ComponentWeights},
	})
	return definitionSource(inventory, err, func(modelID artifact.ID) (recipe.Definition, error) {
		return modelrecipe.GenerationDefinition(modelrecipe.ModuleRAEImagePrepare, modelID, artifact.ID{})
	})
}

// raeImage generates an RAE collection's image on the host, the same on
// every platform: the request's class and seed in, the decoded PNG out.
// The decoder normalizes by the image processor of the encoder beside the
// collection, which the inventory does not hold, so an answer is kept under
// that file too.
var raeImage = capabilityruntime.Reading(capabilityruntime.JSONScalar[rae.Request, *rae.Generator, latentimage.EncodedImage](
	"image-gen", rae.ValidateRequest,
	capabilityruntime.IgnoreInput[rae.Request](rae.Load), rae.RegisterRuntime,
), capabilityruntime.Contents(rae.ProcessorFiles))

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
	if source.Prepare != modelrecipe.ModuleLatentImagePrepare {
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
		slices.SortFunc(members, func(a, b artifact.Component) int { return strings.Compare(a.Name, b.Name) })
		ordinals := map[artifact.ComponentRole]uint32{}
		for i := range members {
			members[i].Ordinal = ordinals[members[i].Role]
			ordinals[members[i].Role]++
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
