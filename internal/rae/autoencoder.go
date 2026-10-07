package rae

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	// A picture arrives as a PNG or a JPEG.
	_ "image/jpeg"
	_ "image/png"

	"overgo/internal/checked"
	"overgo/internal/latentimage"
	"overgo/internal/media"
	"overgo/internal/vitencoder"
)

// cropSides is the two sides a centre crop leaves its surplus on, in equal
// parts.
const cropSides = 2

// Autoencoder is a representation autoencoder's two halves: the frozen
// encoder whose patch tokens are the latent, and the decoder trained to turn
// those tokens back into pixels. It works at one size, the decoder's.
type Autoencoder struct {
	encoder *vitencoder.Encoder
	decoder *vitDecoder
}

// EncoderDirectory is where the collection's encoder sits: beside the
// collection, under the name its reviewed sampler declares.
func (collection Collection) EncoderDirectory() string { return collection.encoder }

// LoadAutoencoder reads the decoder of the collection in directory, with the
// latent statistics it decodes under, and the encoder beside the collection
// whose tokens it reads.
func LoadAutoencoder(directory string) (*Autoencoder, error) {
	collection, held, err := ResolveCollection(directory)
	if err != nil {
		return nil, err
	}
	if !held {
		return nil, fmt.Errorf("rae: %s holds no collection the reviewed sampler serves", directory)
	}
	encoder, err := vitencoder.Load(collection.encoder)
	if err != nil {
		return nil, err
	}
	decoder, err := loadDecoder(collection.Decoder, collection.Statistics, collection.encoder)
	if err != nil {
		return nil, err
	}
	if _, grid, hidden := encoder.Shape(); grid != decoder.grid || hidden != decoder.latent {
		return nil, fmt.Errorf("rae: the encoder gives %d channels over a %d grid, the decoder reads %d over %d", hidden, grid, decoder.latent, decoder.grid)
	}
	return &Autoencoder{encoder: encoder, decoder: decoder}, nil
}

// Side is the side of the square picture the pair takes and gives back.
func (a *Autoencoder) Side() int { return a.decoder.grid * a.decoder.patch }

// Latent is the latent's extent: its channels and the side of its grid.
func (a *Autoencoder) Latent() (channels, grid int) { return a.decoder.latent, a.decoder.grid }

// RoundTrip takes one picture of Side by Side, channel-first in [0, 1], and
// returns the picture the decoder makes of the encoder's tokens for it, in
// the same layout and range. The picture is resized to the encoder's input
// and normalized as the reference does before it encodes.
func (a *Autoencoder) RoundTrip(ctx context.Context, picture []float32) ([]float32, error) {
	pixels, err := a.EncoderInput(ctx, picture)
	if err != nil {
		return nil, err
	}
	tokens, err := a.encoder.Patches(pixels, false)
	if err != nil {
		return nil, err
	}
	return a.Decode(ctx, tokens)
}

// EncoderInput turns one picture of Side by Side, channel-first in [0, 1],
// into what the pair's encoder is given: resized to the encoder's input and
// normalized, as the reference does before it encodes. Another encoder of
// the same input and normalization, bridged to this decoder, is given the
// same pixels.
func (a *Autoencoder) EncoderInput(ctx context.Context, picture []float32) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return encoderInput(picture, a.Side(), a.encoder)
}

// encoderInput resizes a square picture of side, channel-first in [0, 1], to
// the input an encoder takes and normalizes it by that encoder's own mean
// and deviation.
func encoderInput(picture []float32, side int, encoder *vitencoder.Encoder) ([]float32, error) {
	input, _, _ := encoder.Shape()
	mean, deviation := encoder.Normalization()
	pixels, err := vitencoder.InterpolateBicubic(picture, len(mean), side, side, input, input)
	if err != nil {
		return nil, err
	}
	for channel := range mean {
		plane := pixels[channel*input*input:][:input*input]
		for index, value := range plane {
			plane[index] = (value - mean[channel]) / deviation[channel]
		}
	}
	return pixels, nil
}

// Tokens are the encoder's tokens for the pixels EncoderInput gives: the
// latent, one row of Latent channels for each patch of the grid.
func (a *Autoencoder) Tokens(pixels []float32) ([]float32, error) {
	return a.encoder.Patches(pixels, false)
}

// Decode turns tokens, the pair's own or another encoder's bridged to them,
// into the picture the decoder makes of them, channel-first in [0, 1].
func (a *Autoencoder) Decode(ctx context.Context, tokens []float32) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return a.decoder.decodeTokens(tokens)
}

// Reconstruct passes one encoded picture through the pair and returns what
// comes out as a PNG of Side by Side: the picture is scaled until its shorter
// side is Side and the middle square of it is taken, which is what the pair
// works at.
func (a *Autoencoder) Reconstruct(ctx context.Context, encoded []byte) (latentimage.EncodedImage, error) {
	return reconstructPicture(ctx, encoded, a.Side(), a.RoundTrip)
}

// reconstructPicture scales an encoded picture until its shorter side is
// side, takes the middle square, passes it through roundTrip and encodes
// what comes back as a PNG.
func reconstructPicture(ctx context.Context, encoded []byte, side int, roundTrip func(context.Context, []float32) ([]float32, error)) (latentimage.EncodedImage, error) {
	decoded, _, err := image.Decode(bytes.NewReader(encoded))
	if err != nil {
		return latentimage.EncodedImage{}, fmt.Errorf("rae: decode the picture: %w", err)
	}
	height, width := decoded.Bounds().Dy(), decoded.Bounds().Dx()
	if !checked.PositiveInts(height, width) {
		return latentimage.EncodedImage{}, errors.New("rae: the picture is empty")
	}
	scaledHeight, scaledWidth := side, side
	if height < width {
		scaledWidth = max(side, width*side/height)
	} else {
		scaledHeight = max(side, height*side/width)
	}
	scaled := media.ResizeBicubic(decoded, scaledWidth, scaledHeight)
	top, left := (scaledHeight-side)/cropSides, (scaledWidth-side)/cropSides
	plane := side * side
	picture := make([]float32, media.RGBChannels*plane)
	for y := range side {
		for x := range side {
			offset := scaled.PixOffset(left+x, top+y)
			for channel := range media.RGBChannels {
				picture[channel*plane+y*side+x] = float32(scaled.Pix[offset+channel]) / float32(^uint8(0))
			}
		}
	}
	pixels, err := roundTrip(ctx, picture)
	if err != nil {
		return latentimage.EncodedImage{}, err
	}
	for index, value := range pixels {
		pixels[index] = signedSpan*value - 1
	}
	return latentimage.EncodePlanarPNG(pixels, side, side)
}
