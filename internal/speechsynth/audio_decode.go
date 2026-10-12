package speechsynth

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/media"
)

// encodedWAVContract is the stored form of a synthesized speech clip: the
// same output kind the encoded image and video contracts publish.
var encodedWAVContract = artifact.DocumentContract{
	Kind: artifact.KindOutput, MediaType: media.WAVMediaType, Schema: "overgo.encoded-audio.wav.v1",
}

// WAVContent is the WAV a recording is published as, and the form a whole
// reading is stored in, so the answer the store keeps is the bytes that were
// published. The WAV written is mono.
func WAVContent(audio Audio) (artifact.Content, error) {
	if audio.Channels != 1 {
		return artifact.Content{}, fmt.Errorf("speechsynth: speech output has %d channels; one is published", audio.Channels)
	}
	data, err := media.EncodeWAVPCM16(audio.PCM, audio.SampleRate)
	if err != nil {
		return artifact.Content{}, err
	}
	return encodedWAVContract.OwnedContentBytes(data)
}

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
