package main

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/projector"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAProjectionFixtureMayBeAWAV: a projection suite's sound may be a WAV file (kind audio-wav) as well as raw float32
// samples, as docs/verification admits WAV and not raw samples: a sound case and a picture-with-sound case of WAV files
// read as the modes they declare, and a WAV given as a picture is refused.
func TestAProjectionFixtureMayBeAWAV(t *testing.T) {
	t.Parallel()
	capabilities := projector.SessionCapabilities{Image: true, Audio: true, MediaHistory: true}
	sound := projectionFile{Kind: "audio-wav", Path: "speech.wav", Artifact: testutil.ArtifactID(t, artifact.KindFile, "speech")}
	picture := projectionFile{Kind: "image", Path: "picture.png", Artifact: testutil.ArtifactID(t, artifact.KindFile, "picture")}
	encode := func(input projectionInput) string { return string(testcheck.Must(json.Marshal(input))(t)) }
	for _, input := range []projectionInput{
		{Kind: "audio", Files: []projectionFile{sound}, Text: []string{"", "Which word is spoken first?"}},
		{Kind: "mixed", Files: []projectionFile{picture, sound}, Text: []string{"", "", "Which word is spoken first?"}},
	} {
		decoded, err := decodeProjectionInput(encode(input), capabilities)
		if err != nil || decoded.Kind != input.Kind {
			t.Fatalf("a %s case of a WAV file read as %q: %v", input.Kind, decoded.Kind, err)
		}
	}
	if _, err := decodeProjectionInput(encode(projectionInput{Kind: "image", Files: []projectionFile{sound}, Text: []string{"", "describe"}}), capabilities); err == nil {
		t.Fatal("a WAV file given as a picture was accepted")
	}
}
