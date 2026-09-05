package speechsynth

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/media"
)

// DecodeContent rebuilds the synthesized audio from its recorded WAV
// artifact so a request the store already answered replays from the
// document; the published clip is mono.
func (audio *Audio) DecodeContent(content artifact.Content) error {
	if content.Descriptor.MediaType != media.WAVMediaType {
		return errors.New("speechsynth: recorded content is not a WAV")
	}
	samples, sampleRate, err := media.DecodeWAV(content.Data)
	if err != nil {
		return err
	}
	*audio = Audio{PCM: samples, SampleRate: sampleRate, Channels: 1}
	return nil
}
