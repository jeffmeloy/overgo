package mediacapability

import (
	"bytes"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/speechsynth"
)

// TestSpeechOutputIsItsWAV holds what a speech request publishes: one WAV
// under the encoded-audio contract, its samples written as sixteen-bit values
// low byte first and held to the range such a value has. A recording of more
// than one channel is refused, since the WAV written is mono. The published
// bytes are also what a stored answer is rebuilt from, so reading them back
// and publishing again must give the same bytes.
//
// The byte expectations are the ones the retired document recording's test
// held its own encoder to; this is the encoder a served request uses.
func TestSpeechOutputIsItsWAV(t *testing.T) {
	const rate = 24000
	// Silence, half scale either way, and a sample past each end of the range.
	audio := speechsynth.Audio{PCM: []float32{0, 0.5, -0.5, 2, -2}, SampleRate: rate, Channels: 1, Complete: true}
	content, err := OutputContent(audio)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := content.Descriptor
	if descriptor.ID.Kind() != artifact.KindOutput || descriptor.MediaType != media.WAVMediaType || descriptor.Schema != "overgo.encoded-audio.wav.v1" {
		t.Fatalf("speech is published as %s %q under %q", descriptor.ID.Kind(), descriptor.MediaType, descriptor.Schema)
	}
	// 0x0000, 0x4000, 0xC000, then the largest and the smallest value: 0x7FFF
	// and 0x8000.
	want := []byte{0, 0, 0, 64, 0, 192, 255, 127, 0, 128}
	if !bytes.HasSuffix(content.Data, want) {
		t.Fatalf("the samples are written as %v, want %v", content.Data[max(0, len(content.Data)-len(want)):], want)
	}
	decoded, status, err := media.DecodeAudio(t.Context(), content.Data, uint64(len(audio.PCM)))
	if err != nil || len(decoded.Samples) != len(audio.PCM) || decoded.Format.SampleRate != rate || decoded.Format.Channels != 1 {
		t.Fatalf("the published WAV reads as %d samples at %d Hz in %d channels (%s): %v", len(decoded.Samples), decoded.Format.SampleRate, decoded.Format.Channels, status, err)
	}
	var stored speechsynth.Audio
	if err := stored.DecodeContent(content); err != nil {
		t.Fatal(err)
	}
	again, err := OutputContent(stored)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(again.Data, content.Data) || again.Descriptor != content.Descriptor {
		t.Fatal("a recording read back from its WAV is published as other bytes")
	}
	audio.Channels++
	if _, err := OutputContent(audio); err == nil {
		t.Fatal("a recording of two channels was published as a mono WAV")
	}
}
