package latentvideo

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/media"
)

// DecodeContent rebuilds the encoded video from its recorded GIF artifact:
// a request the store already answered replays from the document, and
// the clip's frame count and shape come from the GIF itself.
func (video *EncodedVideo) DecodeContent(content artifact.Content) error {
	if content.Descriptor.MediaType != media.GIFMediaType {
		return errors.New("latent video: recorded content is not a GIF")
	}
	shape, err := media.DecodeGIFShape(content.Data)
	if err != nil {
		return err
	}
	*video = EncodedVideo{
		Data: content.Data, MediaType: media.GIFMediaType, Frames: shape.Frames,
		Channels: media.RGBChannels, Height: shape.Height, Width: shape.Width,
	}
	return nil
}
