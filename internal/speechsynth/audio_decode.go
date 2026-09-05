package speechsynth

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/media"
)

// DecodeContent rebuilds the synthesized audio from its recorded WAV
// artifact so a request the store already answered replays from the
// document, through the bounded audio decoder; the recorded bytes bound
// the samples it may produce.
func (audio *Audio) DecodeContent(content artifact.Content) error {
	if content.Descriptor.MediaType != media.WAVMediaType {
		return errors.New("speechsynth: recorded content is not a WAV")
	}
	decoded, _, err := media.DecodeAudio(context.Background(), content.Data, uint64(len(content.Data)))
	if err != nil {
		return err
	}
	*audio = Audio{PCM: decoded.Samples, SampleRate: int(decoded.Format.SampleRate), Channels: int(decoded.Format.Channels)}
	return nil
}
