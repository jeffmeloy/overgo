package rae

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/latentimage"
	"overgo/internal/modelartifact"
	"overgo/internal/safetensors"
	"overgo/internal/vitencoder"
)

// A bridged pair's directory holds one file: the affine map from another
// encoder's patch tokens to the tokens a collection's decoder reads, and, in
// its metadata, the names of the two directories beside it that it joins.
const (
	bridgeFile          = "encoder_bridge.safetensors"
	bridgeWeightTensor  = "bridge.weight"
	bridgeBiasTensor    = "bridge.bias"
	bridgeSourceKey     = "source_encoder"
	bridgeCollectionKey = "collection"
)

// EncoderBridge is an affine map from one encoder's patch tokens to
// another's: Weight holds a row of source weights for each target channel,
// and Bias a value for each target channel.
type EncoderBridge struct {
	Weight []float32
	Bias   []float32
}

// widths are the token widths the bridge maps from and to.
func (bridge EncoderBridge) widths() (source, target int) {
	target = len(bridge.Bias)
	if target == 0 || len(bridge.Weight)%target != 0 {
		return 0, 0
	}
	return len(bridge.Weight) / target, target
}

// Apply maps rows of source tokens to rows of target tokens.
func (bridge EncoderBridge) Apply(tokens []float32) ([]float32, error) {
	source, target := bridge.widths()
	if source == 0 || len(tokens) == 0 || len(tokens)%source != 0 {
		return nil, fmt.Errorf("rae: %d token values do not fit a bridge from %d channels", len(tokens), source)
	}
	rows := len(tokens) / source
	mapped := make([]float32, rows*target)
	linear(mapped, tokens, bridge.Weight, bridge.Bias, rows, source, target)
	return mapped, nil
}

// SaveEncoderBridge writes a bridge into directory as a bridged pair: the
// map, and the names of the source encoder's directory and the collection's
// directory, both of which sit beside directory.
func SaveEncoderBridge(directory string, bridge EncoderBridge, sourceEncoder, collection string) error {
	source, target := bridge.widths()
	if source == 0 || sourceEncoder == "" || collection == "" || !filepath.IsLocal(sourceEncoder) || !filepath.IsLocal(collection) {
		return errors.New("rae: a bridge needs its weights and the names of the encoder and the collection beside it")
	}
	if err := os.MkdirAll(directory, bridgeDirectoryMode); err != nil {
		return err
	}
	return safetensors.Save(filepath.Join(directory, bridgeFile),
		map[string][]float32{bridgeWeightTensor: bridge.Weight, bridgeBiasTensor: bridge.Bias},
		map[string][]int{bridgeWeightTensor: {target, source}, bridgeBiasTensor: {target}},
		map[string]string{bridgeSourceKey: sourceEncoder, bridgeCollectionKey: collection})
}

// bridgeDirectoryMode keeps a bridged pair's directory to its owner.
const bridgeDirectoryMode = 0o700

// BridgedPair is the files a bridged pair reads: its bridge, the encoder
// whose tokens the bridge maps, and the collection whose decoder reads what
// it maps them to.
type BridgedPair struct {
	Bridge              string
	SourceEncoder       string
	CollectionDirectory string
	Collection          Collection
}

// ResolveBridgedPair reads directory as a bridged pair. held is false when
// the directory holds no bridge, so the caller resolves it as another kind of
// autoencoder; a held pair whose encoder or collection is absent is refused.
func ResolveBridgedPair(directory string) (pair BridgedPair, held bool, err error) {
	directory = filepath.Clean(directory)
	path := filepath.Join(directory, bridgeFile)
	if info, err := os.Stat(path); err != nil || info.IsDir() {
		return BridgedPair{}, false, nil
	}
	source, err := safetensors.OpenFile(path)
	if err != nil {
		return BridgedPair{}, true, err
	}
	metadata := source.Metadata[bridgeFile]
	if err := source.Close(); err != nil {
		return BridgedPair{}, true, err
	}
	encoder, collectionName := metadata[bridgeSourceKey], metadata[bridgeCollectionKey]
	if encoder == "" || collectionName == "" || !filepath.IsLocal(encoder) || !filepath.IsLocal(collectionName) {
		return BridgedPair{}, true, errors.New("rae: the bridge does not name the encoder and the collection it joins")
	}
	parent := filepath.Dir(directory)
	collectionDirectory := filepath.Join(parent, collectionName)
	collection, collectionHeld, err := ResolveCollection(collectionDirectory)
	if err != nil || !collectionHeld {
		return BridgedPair{}, true, errors.Join(err, fmt.Errorf("rae: the bridge's collection %s is absent", collectionName))
	}
	pair = BridgedPair{Bridge: path, SourceEncoder: filepath.Join(parent, encoder), CollectionDirectory: collectionDirectory, Collection: collection}
	if _, err := os.Stat(pair.SourceEncoder); err != nil {
		return BridgedPair{}, true, fmt.Errorf("rae: the bridge's encoder %s: %w", encoder, err)
	}
	return pair, true, nil
}

// pairFiles are the files a decoder and the encoder that feeds it are: the
// collection's decoder and latent statistics, and the encoder's weights,
// configuration and image processor.
func pairFiles(collection Collection, encoder string) []modelartifact.FileSpec {
	return []modelartifact.FileSpec{
		{Path: collection.Decoder, Name: "decoder", Role: artifact.ComponentWeights},
		{Path: collection.Statistics, Name: "statistics", Role: artifact.ComponentWeights},
		{Path: filepath.Join(encoder, "model.safetensors"), Name: "encoder/weights", Role: artifact.ComponentWeights},
		{Path: filepath.Join(encoder, "config.json"), Name: "encoder/config", Role: artifact.ComponentConfig},
		{Path: filepath.Join(encoder, processorConfig), Name: "encoder/processor", Role: artifact.ComponentConfig},
	}
}

// PairInventory names, each by its bytes, the files the autoencoder in
// directory runs through: a bridged pair's bridge, the encoder the bridge maps
// and the decoder of the collection it maps to; or a collection's decoder, the
// latent statistics it decodes under and the encoder beside the collection.
// Serving and training name a pair by this one inventory, so a model trained
// is the model served. held is false when directory is neither.
func PairInventory(directory string) (inventory modelartifact.Inventory, held bool, err error) {
	bridged, isBridged, err := ResolveBridgedPair(directory)
	if err != nil {
		return inventory, isBridged, err
	}
	collection, held, err := ResolveCollection(directory)
	if err != nil || !held && !isBridged {
		return inventory, held, err
	}
	// A pair spans directories beside each other, so the inventory roots
	// at their common parent; the capability resolves only from the pair's
	// own directory, which is recorded beside the root.
	files := pairFiles(collection, collection.encoder)
	if isBridged {
		// The bridged pair runs the source encoder in place of the
		// decoder's own, of which it reads only the image processor.
		files = append([]modelartifact.FileSpec{{Path: bridged.Bridge, Name: "bridge", Role: artifact.ComponentWeights}},
			pairFiles(bridged.Collection, bridged.SourceEncoder)...)
		files = append(files, modelartifact.FileSpec{
			Path: filepath.Join(bridged.Collection.encoder, processorConfig), Name: "decoder/processor", Role: artifact.ComponentConfig,
		})
	}
	inventory, err = modelartifact.FromFiles(filepath.Dir(filepath.Clean(directory)), files)
	if err == nil {
		var own artifact.Location
		own, err = artifact.CanonicalLocalLocation(inventory.Manifest.ID, artifact.LocationDirectory, directory)
		inventory.Locations = append(inventory.Locations, own)
	}
	return inventory, true, err
}

// BridgedAutoencoder is another encoder's tokens, mapped by a bridge, feeding
// a collection's decoder: a pair made of two parts that were not trained
// together. It works at the decoder's size, as the collection's own pair does.
type BridgedAutoencoder struct {
	source  *vitencoder.Encoder
	bridge  EncoderBridge
	decoder *vitDecoder
}

// LoadBridgedAutoencoder reads the bridged pair in directory: its bridge, the
// encoder beside it and the decoder of the collection beside it.
func LoadBridgedAutoencoder(directory string) (*BridgedAutoencoder, error) {
	pair, held, err := ResolveBridgedPair(directory)
	if err != nil {
		return nil, err
	}
	if !held {
		return nil, fmt.Errorf("rae: %s holds no encoder bridge", directory)
	}
	bridge, err := LoadEncoderBridge(pair.Bridge)
	if err != nil {
		return nil, err
	}
	source, err := vitencoder.Load(pair.SourceEncoder)
	if err != nil {
		return nil, err
	}
	decoder, err := loadDecoder(pair.Collection.Decoder, pair.Collection.Statistics, pair.Collection.encoder)
	if err != nil {
		return nil, err
	}
	return bridgeTo(decoder, source, bridge)
}

// LoadEncoderBridge reads the bridge a bridged pair keeps in the file at
// path, for a caller that joins it to parts it has already loaded.
func LoadEncoderBridge(path string) (EncoderBridge, error) {
	file, err := safetensors.OpenFile(path)
	if err != nil {
		return EncoderBridge{}, err
	}
	weight, weightErr := safetensors.ReadF32(file.Tensors[bridgeWeightTensor])
	bias, biasErr := safetensors.ReadF32(file.Tensors[bridgeBiasTensor])
	if err := errors.Join(weightErr, biasErr, file.Close()); err != nil {
		return EncoderBridge{}, fmt.Errorf("rae: read the bridge: %w", err)
	}
	return EncoderBridge{Weight: weight, Bias: bias}, nil
}

// Bridged joins another loaded encoder and a bridge to this pair's decoder,
// as a bridged pair that has not been saved: what a fit judges before it
// keeps a bridge.
func (a *Autoencoder) Bridged(source *vitencoder.Encoder, bridge EncoderBridge) (*BridgedAutoencoder, error) {
	return bridgeTo(a.decoder, source, bridge)
}

// bridgeTo joins an encoder, a bridge and a decoder, refusing parts that do
// not fit: the encoder must give the grid the decoder reads, and the bridge
// must map the encoder's width to the decoder's.
func bridgeTo(decoder *vitDecoder, source *vitencoder.Encoder, bridge EncoderBridge) (*BridgedAutoencoder, error) {
	_, sourceGrid, sourceWidth := source.Shape()
	from, to := bridge.widths()
	if !checked.Equal(sourceGrid, decoder.grid) || !checked.Equal(from, sourceWidth) || !checked.Equal(to, decoder.latent) {
		return nil, fmt.Errorf("rae: an encoder of %d channels over a %d grid and a bridge from %d to %d do not fit a decoder of %d over %d",
			sourceWidth, sourceGrid, from, to, decoder.latent, decoder.grid)
	}
	return &BridgedAutoencoder{source: source, bridge: bridge, decoder: decoder}, nil
}

// Side is the side of the square picture the pair takes and gives back.
func (a *BridgedAutoencoder) Side() int { return a.decoder.grid * a.decoder.patch }

// BridgeParameterCount is the count of the bridge's weights and offsets: what
// training the pair's bridge moves.
func (a *BridgedAutoencoder) BridgeParameterCount() int {
	return len(a.bridge.Weight) + len(a.bridge.Bias)
}

// Latent is the extent of what the decoder reads: its channels and the side
// of its grid.
func (a *BridgedAutoencoder) Latent() (channels, grid int) { return a.decoder.latent, a.decoder.grid }

// sourceTokens are the source encoder's patch tokens, with its own final
// norm, for one picture of Side by Side, channel-first in [0, 1]: what the
// bridge maps, and what a bridge is fitted from.
func (a *BridgedAutoencoder) sourceTokens(ctx context.Context, picture []float32) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return BridgeSourceTokens(a.source, picture, a.Side())
}

// BridgeSourceTokens runs one square picture of side, channel-first in
// [0, 1], through an encoder that is bridged or is to be: resized to the
// encoder's input, normalized by its own mean and deviation, and read as its
// patch tokens with its own final norm. A fit and the pair it makes read the
// source encoder the same way through this.
func BridgeSourceTokens(source *vitencoder.Encoder, picture []float32, side int) ([]float32, error) {
	pixels, err := encoderInput(picture, side, source)
	if err != nil {
		return nil, err
	}
	return source.Patches(pixels, true)
}

// RoundTrip takes one picture of Side by Side, channel-first in [0, 1], and
// returns the picture the decoder makes of the bridged tokens.
func (a *BridgedAutoencoder) RoundTrip(ctx context.Context, picture []float32) ([]float32, error) {
	tokens, err := a.sourceTokens(ctx, picture)
	if err != nil {
		return nil, err
	}
	mapped, err := a.bridge.Apply(tokens)
	if err != nil {
		return nil, err
	}
	return a.decoder.decodeTokens(mapped)
}

// Reconstruct passes one encoded picture through the bridged pair and
// returns what comes out as a PNG of Side by Side.
func (a *BridgedAutoencoder) Reconstruct(ctx context.Context, encoded []byte) (latentimage.EncodedImage, error) {
	return reconstructPicture(ctx, encoded, a.Side(), a.RoundTrip)
}
