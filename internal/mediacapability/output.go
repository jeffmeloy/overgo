package mediacapability

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/oscillatorimage"
	"overgo/internal/speechsynth"
)

// encodedWAVContract is the stored form of a synthesized speech clip: the
// same output kind the encoded image and video contracts publish.
var encodedWAVContract = artifact.DocumentContract{
	Kind: artifact.KindOutput, MediaType: media.WAVMediaType, Schema: "overgo.encoded-audio.wav.v1",
}

// OutputContent turns what an executor produced into the media artifact
// the store publishes and the page renders: encoded images as PNG, video
// as GIF, speech as WAV. The output's own type decides; a value outside
// the catalog's output vocabulary is refused rather than guessed at.
func OutputContent(output any) (artifact.Content, error) {
	switch value := output.(type) {
	case latentimage.EncodedImage:
		return latentimage.PNGContent(value)
	case latentvideo.EncodedVideo:
		return latentvideo.GIFContent(value)
	case oscillatorimage.EncodedVideo:
		return latentvideo.GIFContent(latentvideo.EncodedVideo{
			Data: value.Data, MediaType: value.MediaType, Frames: value.Frames,
			Channels: value.Channels, Height: value.Height, Width: value.Width,
		})
	case speechsynth.Audio:
		if value.Channels != 1 {
			return artifact.Content{}, fmt.Errorf("media capability: speech output has %d channels; one is published", value.Channels)
		}
		data, err := media.EncodeWAVPCM16(value.PCM, value.SampleRate)
		if err != nil {
			return artifact.Content{}, err
		}
		return encodedWAVContract.OwnedContentBytes(data)
	case nil:
		return artifact.Content{}, errors.New("media capability: the executor produced no output")
	default:
		return artifact.Content{}, fmt.Errorf("media capability: output type %T is not a media artifact", output)
	}
}
