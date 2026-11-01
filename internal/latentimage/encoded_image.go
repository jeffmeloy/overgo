package latentimage

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/media"
)

// EncodedImage is a typed PNG output. Data owns the published artifact bytes.
type EncodedImage struct {
	Data      []byte  `json:"data"`
	MediaType string  `json:"media_type"`
	Channels  int     `json:"channels"`
	Height    int     `json:"height"`
	Width     int     `json:"width"`
	Minimum   float32 `json:"minimum"`
	Maximum   float32 `json:"maximum"`
}

var encodedPNGContract = artifact.DocumentContract{
	Kind: artifact.KindOutput, MediaType: media.PNGMediaType, Schema: "overgo.encoded-image.png.v1",
}

// PNGContent validates and publishes the encoded PNG bytes without a JSON copy.
func PNGContent(image EncodedImage) (artifact.Content, error) {
	if !media.ValidRGBRaster(image.MediaType, image.Channels, image.Height, image.Width) {
		return artifact.Content{}, errors.New("latent image: invalid encoded PNG")
	}
	return encodedPNGContract.OwnedContentBytes(image.Data)
}

func encodePNG(pixels []float32, height, width int) (EncodedImage, error) {
	return encodeRGB(pixels, height, width, false, false)
}

func encodeUnitPNG(pixels []float32, height, width int) (EncodedImage, error) {
	return encodeRGB(pixels, height, width, false, true)
}

// EncodePlanarPNG publishes normalized CHW RGB without an HWC copy.
func EncodePlanarPNG(pixels []float32, height, width int) (EncodedImage, error) {
	return encodeRGB(pixels, height, width, true, false)
}

func encodeRGB(pixels []float32, height, width int, planar, unit bool) (EncodedImage, error) {
	var data []byte
	var minimum, maximum float32
	var err error
	if unit {
		data, minimum, maximum, err = media.EncodeUnitRGBPNG(pixels, height, width)
	} else {
		data, minimum, maximum, err = media.EncodeNormalizedRGBPNG(pixels, height, width, planar)
	}
	if err != nil {
		return EncodedImage{}, err
	}
	return EncodedImage{
		Data: data, MediaType: media.PNGMediaType,
		Channels: media.RGBChannels, Height: height, Width: width, Minimum: minimum, Maximum: maximum,
	}, nil
}
