package mediacapability

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/speechsynth"
)

// OutputContent turns what an executor produced into the artifact the
// store publishes and the page renders: encoded images as PNG, video as the
// GIF or MP4 it was written as, speech as WAV, a text answer as plain text
// (the contract the transcription workflow shares). The output's own type decides; a value
// outside the catalog's output vocabulary is refused rather than guessed
// at.
func OutputContent(output any) (artifact.Content, error) {
	switch value := output.(type) {
	case string:
		return modelrecipe.TextAnswerContract.OwnedContentBytes([]byte(value))
	case latentimage.EncodedImage:
		return latentimage.PNGContent(value)
	case latentvideo.EncodedVideo:
		return latentvideo.VideoContent(value)
	case oscillatorimage.EncodedVideo:
		return latentvideo.VideoContent(latentvideo.EncodedVideo{
			Data: value.Data, MediaType: value.MediaType, Frames: value.Frames,
			Channels: value.Channels, Height: value.Height, Width: value.Width, FPS: value.FPS,
		})
	case speechsynth.Audio:
		return speechsynth.WAVContent(value)
	case nil:
		return artifact.Content{}, errors.New("media capability: the executor produced no output")
	default:
		return artifact.Content{}, fmt.Errorf("media capability: output type %T is not a media artifact", output)
	}
}
