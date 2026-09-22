package speechrecognition

import (
	"testing"

	"overgo/internal/artifact"
)

// TestTranscriptStageEncodesItsOutput holds the transcription stage to the
// rule every scalar capability stage answers to: what it returns leaves the
// runtime with a content identity. A stage registered without an encoder
// executes a model to completion and is then refused at the last step, so the
// encoding is checked here rather than discovered by a verify run.
func TestTranscriptStageEncodesItsOutput(t *testing.T) {
	t.Parallel()
	for _, text := range []string{"a transcript", ""} {
		content, err := transcriptContent(text)
		if err != nil {
			t.Fatal(err)
		}
		if err := content.Validate(); err != nil {
			t.Fatal(err)
		}
		if content.Descriptor.ID.Kind() != artifact.KindFile {
			t.Fatalf("transcript identity kind = %q", content.Descriptor.ID.Kind())
		}
	}
	first, err := transcriptContent("one")
	if err != nil {
		t.Fatal(err)
	}
	second, err := transcriptContent("another")
	if err != nil {
		t.Fatal(err)
	}
	if first.Descriptor.ID == second.Descriptor.ID {
		t.Fatal("two transcripts share one identity")
	}
	if err := RegisterRuntime(nil, first.Descriptor.ID, nil); err == nil {
		t.Fatal("a runtime binding without a recognizer was accepted")
	}
}
