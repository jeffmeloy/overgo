package latentvideo

import (
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/media"
)

// DecodeContent rebuilds the encoded video from its recorded clip artifact,
// a GIF or an MP4: a request the store already answered replays from the
// document, and the clip's frame count, shape and frame rate come from the
// clip itself, so the replayed clip publishes as the one first served.
func (video *EncodedVideo) DecodeContent(content artifact.Content) error {
	var frames, height, width, fps int
	switch content.Descriptor.MediaType {
	case media.GIFMediaType:
		shape, err := media.DecodeGIFShape(content.Data)
		if err != nil {
			return err
		}
		frames, height, width, fps = shape.Frames, shape.Height, shape.Width, shape.FPS
	case media.MP4MediaType:
		shape, err := media.DecodeMP4Shape(content.Data)
		if err != nil {
			return err
		}
		frames, height, width, fps = shape.Frames, shape.Height, shape.Width, shape.FPS
	default:
		return fmt.Errorf("latent video: recorded content is %q, neither a GIF nor an MP4", content.Descriptor.MediaType)
	}
	*video = EncodedVideo{
		Data: content.Data, MediaType: content.Descriptor.MediaType, Frames: frames,
		Channels: media.RGBChannels, Height: height, Width: width, FPS: fps,
	}
	return nil
}
