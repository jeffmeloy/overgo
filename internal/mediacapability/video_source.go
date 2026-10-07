package mediacapability

import (
	"context"
	"fmt"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/hfrepo"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/rae"
	"overgo/internal/recipe"
	"overgo/internal/tensor"
	"overgo/internal/workflowruntime"
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

// resolveAutoencoder reads a directory as the autoencoder it holds: a
// representation autoencoder's collection, whose pair is its decoder, the
// latent statistics it decodes under and the encoder beside the collection,
// on the host topology; or a Wan release's one autoencoder file.
func resolveAutoencoder(path string) (Source, error) {
	collection, held, err := rae.ResolveCollection(path)
	if err != nil {
		return Source{}, err
	}
	if !held {
		inventory, err := autoencoderInventory(path)
		return Source{Inventory: inventory}, err
	}
	// The pair spans the collection and its encoder's repository beside
	// it, so the inventory roots at their common parent; the capability
	// resolves only from the collection, which is recorded beside the root.
	encoder := collection.EncoderDirectory()
	inventory, err := modelartifact.FromFiles(filepath.Dir(filepath.Clean(path)), []modelartifact.FileSpec{
		{Path: collection.Decoder, Name: "decoder", Role: artifact.ComponentWeights},
		{Path: collection.Statistics, Name: "statistics", Role: artifact.ComponentWeights},
		{Path: filepath.Join(encoder, "model.safetensors"), Name: "encoder/weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(encoder, "config.json"), Name: "encoder/config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(encoder, "preprocessor_config.json"), Name: "encoder/processor", Role: artifact.ComponentConfig},
	})
	if err == nil {
		var own artifact.Location
		own, err = artifact.CanonicalLocalLocation(inventory.Manifest.ID, artifact.LocationDirectory, path)
		inventory.Locations = append(inventory.Locations, own)
	}
	return definitionSource(inventory, err, modelrecipe.RepresentationReconstructionDefinition)
}

// representationReconstruction passes a picture through a representation
// autoencoder on the host, the same on every platform.
var representationReconstruction = capabilityruntime.JSONScalar[latentvideo.ReconstructionRequest, *rae.Autoencoder, latentimage.EncodedImage](
	"reconstruction", latentvideo.ValidateReconstructionRequest,
	capabilityruntime.IgnoreInput[latentvideo.ReconstructionRequest](rae.LoadAutoencoder),
	func(runtime *workflowruntime.Runtime, modelID artifact.ID, pair *rae.Autoencoder) error {
		return workflowruntime.RegisterContextStage(runtime, modelrecipe.ModuleReconstructRepresentation, modelID,
			func(ctx context.Context, request latentvideo.ReconstructionRequest) (latentimage.EncodedImage, error) {
				return pair.Reconstruct(ctx, request.Image)
			}, latentimage.PNGContent)
	},
)

// reconstructionCapability runs each autoencoder where it runs: a
// representation autoencoder on the host, a Wan release's on the device.
func reconstructionCapability() Capability {
	return Capability{
		Resolve: resolveAutoencoder,
		Execute: capabilityruntime.ExecutorCatalog{
			modelrecipe.ModuleReconstructImage:          deviceReconstruction(),
			modelrecipe.ModuleReconstructRepresentation: representationReconstruction,
		}.Execute,
	}
}

// OpenAutoencoder loads the autoencoder a directory holds and keeps it until
// the caller closes it.
func OpenAutoencoder(path string) (Autoencoder, error) {
	if _, held, err := rae.ResolveCollection(path); err != nil {
		return nil, err
	} else if !held {
		return openDeviceAutoencoder(path)
	}
	pair, err := rae.LoadAutoencoder(path)
	if err != nil {
		return nil, err
	}
	return representationAutoencoder{pair: pair}, nil
}

// representationAutoencoder is a representation autoencoder as an
// evaluation holds one: it takes a single picture of the pair's own size, in
// the signed unit range clips are held in.
type representationAutoencoder struct{ pair *rae.Autoencoder }

// RoundTrip passes the one picture through the pair.
func (a representationAutoencoder) RoundTrip(ctx context.Context, source latentvideo.SourceVideo) (latentvideo.SourceVideo, latentvideo.SourceVideoShape, error) {
	side := a.pair.Side()
	if source.Channels != media.RGBChannels || source.Frames != tensor.SingletonExtent || source.Height != side || source.Width != side {
		return latentvideo.SourceVideo{}, latentvideo.SourceVideoShape{}, fmt.Errorf(
			"media capability: this autoencoder takes one picture of %d by %d, and was given %d frames of %d by %d", side, side, source.Frames, source.Height, source.Width)
	}
	// The pair works in [0, 1]; a clip is held in [-1, 1].
	picture := make([]float32, len(source.Pixels))
	for index, value := range source.Pixels {
		picture[index] = (value + tensor.SingletonExtent) / tensor.PairedExtent
	}
	back, err := a.pair.RoundTrip(ctx, picture)
	if err != nil {
		return latentvideo.SourceVideo{}, latentvideo.SourceVideoShape{}, err
	}
	for index, value := range back {
		back[index] = tensor.PairedExtent*value - tensor.SingletonExtent
	}
	channels, grid := a.pair.Latent()
	output := source
	output.Pixels = back
	return output, latentvideo.SourceVideoShape{Channels: channels, Frames: tensor.SingletonExtent, Height: grid, Width: grid}, nil
}

// Close releases nothing: the pair holds no device.
func (representationAutoencoder) Close(context.Context) error { return nil }
