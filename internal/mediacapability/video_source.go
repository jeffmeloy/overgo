package mediacapability

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
	"overgo/internal/latentvideo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

func resolveDeclaredVideoSource(path string) (Source, error) {
	identity, err := hfrepo.InspectIdentity(path)
	if err != nil {
		return Source{}, err
	}
	source, err := modelrecipe.ResolveGenerationSource(recipe.TaskVideoGen, identity)
	if err != nil {
		return Source{}, err
	}
	inventory, err := imageGenInventory(path)
	if err != nil {
		return Source{}, err
	}
	return Source{Inventory: inventory, Define: func(modelID artifact.ID) (recipe.Definition, []artifact.Content, error) {
		definition, err := modelrecipe.GenerationDefinition(source.Prepare, modelID, artifact.ID{})
		return definition, nil, err
	}}, nil
}

// Autoencoder is an installed autoencoder held open for many round trips,
// which is what an evaluation asks of one: a clip in, the clip that comes
// back, and the extent of the latent between them.
type Autoencoder interface {
	RoundTrip(ctx context.Context, source latentvideo.SourceVideo) (latentvideo.SourceVideo, latentvideo.SourceVideoShape, error)
	Close(context.Context) error
}

// autoencoderInventory reads the one file an autoencoder is, whichever
// generator's directory it sits in: the round trip loads nothing else.
func autoencoderInventory(path string) (modelartifact.Inventory, error) {
	return modelartifact.FromFiles(path, []modelartifact.FileSpec{
		{Path: latentvideo.AutoencoderWeights(path), Name: "vae/weights", Role: artifact.ComponentWeights},
	})
}
