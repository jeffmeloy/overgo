package oscillatorimage

import (
	"overgo/internal/artifact"
	"overgo/internal/latentvideo"
)

// DecodeContent rebuilds the encoded video from its recorded GIF artifact
// so a request the store already answered replays from the document; the
// oscillator's clip is the latent video's GIF form under its own type, and
// the changed-pixel count is not recorded in the GIF, so it stays unset.
func (video *EncodedVideo) DecodeContent(content artifact.Content) error {
	var decoded latentvideo.EncodedVideo
	if err := decoded.DecodeContent(content); err != nil {
		return err
	}
	*video = EncodedVideo{FPS: videoFPS}
	video.Data, video.MediaType = decoded.Data, decoded.MediaType
	video.Frames, video.Channels, video.Height, video.Width = decoded.Frames, decoded.Channels, decoded.Height, decoded.Width
	return nil
}
