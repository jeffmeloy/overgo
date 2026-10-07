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

// pairFiles are the files a decoder and the encoder that feeds it are: the
// collection's decoder and latent statistics, and the encoder's weights,
// configuration and image processor.
func pairFiles(collection rae.Collection, encoder string) []modelartifact.FileSpec {
	return []modelartifact.FileSpec{
		{Path: collection.Decoder, Name: "decoder", Role: artifact.ComponentWeights},
		{Path: collection.Statistics, Name: "statistics", Role: artifact.ComponentWeights},
		{Path: filepath.Join(encoder, "model.safetensors"), Name: "encoder/weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(encoder, "config.json"), Name: "encoder/config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(encoder, "preprocessor_config.json"), Name: "encoder/processor", Role: artifact.ComponentConfig},
	}
}

// resolveAutoencoder reads a directory as the autoencoder it holds, on the
// host topology: a bridged pair, whose parts are its bridge, the encoder the
// bridge maps and the decoder of the collection it maps to; or a
// representation autoencoder's collection, whose pair is its decoder, the
// latent statistics it decodes under and the encoder beside the collection.
// Any other directory is read as a Wan release's one autoencoder file, on the
// device topology.
func resolveAutoencoder(path string) (Source, error) {
	bridged, isBridged, err := rae.ResolveBridgedPair(path)
	if err != nil {
		return Source{}, err
	}
	collection, held, err := rae.ResolveCollection(path)
	if err != nil {
		return Source{}, err
	}
	if !held && !isBridged {
		inventory, err := autoencoderInventory(path)
		return Source{Inventory: inventory}, err
	}
	// A pair spans directories beside each other, so the inventory roots
	// at their common parent; the capability resolves only from the pair's
	// own directory, which is recorded beside the root.
	files := pairFiles(collection, collection.EncoderDirectory())
	if isBridged {
		// The bridged pair runs the source encoder in place of the
		// decoder's own, of which it reads only the image processor.
		files = append([]modelartifact.FileSpec{{Path: bridged.Bridge, Name: "bridge", Role: artifact.ComponentWeights}},
			pairFiles(bridged.Collection, bridged.SourceEncoder)...)
		files = append(files, modelartifact.FileSpec{
			Path: filepath.Join(bridged.Collection.EncoderDirectory(), "preprocessor_config.json"), Name: "decoder/processor", Role: artifact.ComponentConfig,
		})
	}
	inventory, err := modelartifact.FromFiles(filepath.Dir(filepath.Clean(path)), files)
	if err == nil {
		var own artifact.Location
		own, err = artifact.CanonicalLocalLocation(inventory.Manifest.ID, artifact.LocationDirectory, path)
		inventory.Locations = append(inventory.Locations, own)
	}
	return definitionSource(inventory, err, modelrecipe.RepresentationReconstructionDefinition)
}

// hostPair is a representation autoencoder as the host runs one, the
// collection's own pair or a bridged one: a square picture of its side in and
// out, through a latent of its channels over its grid.
type hostPair interface {
	Side() int
	Latent() (channels, grid int)
	RoundTrip(ctx context.Context, picture []float32) ([]float32, error)
	Reconstruct(ctx context.Context, encoded []byte) (latentimage.EncodedImage, error)
}

// loadHostPair loads the pair a directory holds: a bridged pair where the
// directory holds a bridge, else the collection's own.
func loadHostPair(path string) (hostPair, error) {
	if _, bridged, err := rae.ResolveBridgedPair(path); err != nil {
		return nil, err
	} else if bridged {
		return rae.LoadBridgedAutoencoder(path)
	}
	return rae.LoadAutoencoder(path)
}

// representationReconstruction passes a picture through a representation
// autoencoder on the host, the same on every platform.
var representationReconstruction = capabilityruntime.JSONScalar[latentvideo.ReconstructionRequest, hostPair, latentimage.EncodedImage](
	"reconstruction", latentvideo.ValidateReconstructionRequest,
	capabilityruntime.IgnoreInput[latentvideo.ReconstructionRequest](loadHostPair),
	func(runtime *workflowruntime.Runtime, modelID artifact.ID, pair hostPair) error {
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
	_, bridged, err := rae.ResolveBridgedPair(path)
	if err != nil {
		return nil, err
	}
	if _, held, err := rae.ResolveCollection(path); err != nil {
		return nil, err
	} else if !held && !bridged {
		return openDeviceAutoencoder(path)
	}
	pair, err := loadHostPair(path)
	if err != nil {
		return nil, err
	}
	return representationAutoencoder{pair: pair}, nil
}

// representationAutoencoder is a representation autoencoder as an
// evaluation holds one: it takes a single picture of the pair's own size, in
// the signed unit range clips are held in.
type representationAutoencoder struct{ pair hostPair }

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
