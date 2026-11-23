package latentimage

import (
	"bytes"
	"errors"
	"image/png"

	"overgo/internal/artifact"
	"overgo/internal/media"
)

// DecodeContent rebuilds the encoded image from its recorded PNG artifact:
// a request the store already answered replays from the document, and
// the image's shape comes from the PNG header, not from a JSON copy.
func (image *EncodedImage) DecodeContent(content artifact.Content) error {
	if content.Descriptor.MediaType != media.PNGMediaType {
		return errors.New("latent image: recorded content is not a PNG")
	}
	config, err := png.DecodeConfig(bytes.NewReader(content.Data))
	if err != nil {
		return err
	}
	// What the picture was drawn from travels in its bytes (PNGContent), so a stored answer gives it back.
	texts := readPictureTexts(content.Data)
	*image = EncodedImage{
		Data: content.Data, MediaType: media.PNGMediaType,
		Channels: media.RGBChannels, Height: config.Height, Width: config.Width,
		Prompt: texts[textPromptUsed], NegativePrompt: texts[textNegativeUsed], Sharpened: texts[textFinishing] == finishingSharpened,
	}
	return nil
}
