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
	// Preset is what the generation that drew the picture ran; a picture decoded from its bytes holds none.
	Preset media.Preset `json:"preset,omitzero"`
	// Prompt and NegativePrompt are what the picture's conditioning was drawn from, as the MOSS Ultra workflow shows them
	// ("Prompt used", "Negative prompt used"), where its capability reports them (Request.Drawn); a picture decoded from
	// its bytes holds none.
	Prompt         string `json:"prompt,omitzero"`
	NegativePrompt string `json:"negative_prompt,omitzero"`
}

// RanPreset is the preset the picture was drawn at (media.PresetOf).
func (image EncodedImage) RanPreset() media.Preset { return image.Preset }

// DrawnFrom is the prompt and the negative prompt the picture was drawn from, both empty where its capability reports
// none, as a page shows them beside the picture (moss-prompts-used).
func (image EncodedImage) DrawnFrom() (prompt, negative string) {
	return image.Prompt, image.NegativePrompt
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
