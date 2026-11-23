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
	// ("Prompt used", "Negative prompt used"), as its draw sets them; Sharpened says its finishing sharpened it. The
	// picture stored keeps all three in its own bytes (PNGContent), as ComfyUI writes its prompt into the pictures it
	// saves, and a picture its stored run answers reads them back (DecodeContent), so they travel with the draw's result
	// and are never read back from the request its run stored, which a request above a mebibyte is stored as the digest
	// of (draw-result-carries, SQA-464).
	Prompt         string `json:"prompt,omitzero"`
	NegativePrompt string `json:"negative_prompt,omitzero"`
	Sharpened      bool   `json:"sharpened,omitzero"`
	// Drawn is the picture as it was drawn, before its finishing's sharpening, as PNG bytes of the same extent, where it
	// was sharpened: the MOSS Ultra workflow saves both and compares them (moss-sharpened-pair). A picture its stored run
	// answers holds it where its capability finds it (the sharpened picture derived from it).
	Drawn []byte `json:"drawn,omitempty"`
}

// BeforeFinishing is the picture as it was drawn before its finishing sharpened it, an EncodedImage, and whether it was
// sharpened, so a page shows the two side by side.
func (image EncodedImage) BeforeFinishing() (any, bool) {
	if len(image.Drawn) == 0 {
		return nil, false
	}
	return EncodedImage{Data: image.Drawn, MediaType: image.MediaType, Channels: image.Channels, Height: image.Height, Width: image.Width}, true
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

// PNGContent validates and publishes the encoded PNG bytes without a JSON copy, what the picture was drawn from written
// into them as text where it says (pictureTexts), so a picture saying nothing keeps its bytes as they are, and a
// picture published again is the same content.
func PNGContent(image EncodedImage) (artifact.Content, error) {
	if !media.ValidRGBRaster(image.MediaType, image.Channels, image.Height, image.Width) {
		return artifact.Content{}, errors.New("latent image: invalid encoded PNG")
	}
	data, err := withPictureTexts(image.Data, image.pictureTexts())
	if err != nil {
		return artifact.Content{}, err
	}
	return encodedPNGContract.OwnedContentBytes(data)
}

// pictureTexts are what the picture was drawn from, as its stored bytes carry them, by keyword: none it does not say.
func (image EncodedImage) pictureTexts() [][2]string {
	var texts [][2]string
	if image.Prompt != "" {
		texts = append(texts, [2]string{textPromptUsed, image.Prompt})
	}
	if image.NegativePrompt != "" {
		texts = append(texts, [2]string{textNegativeUsed, image.NegativePrompt})
	}
	if image.Sharpened {
		texts = append(texts, [2]string{textFinishing, finishingSharpened})
	}
	return texts
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
