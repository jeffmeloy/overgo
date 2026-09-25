package speechrecognition

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

func TestRecognizerLoadAdmission(t *testing.T) {
	t.Parallel()
	canceled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	directory := filepath.Join(t.TempDir(), "absent-checkpoint")
	for _, form := range []struct {
		name    string
		binding *TransducerBinding
	}{{"connectionist", nil}, {"recurrent", &TransducerBinding{}}} {
		t.Run(form.name, func(t *testing.T) {
			for _, test := range []struct {
				name    string
				ctx     context.Context
				memory  uint64
				message string
			}{
				{"canceled", canceled, 1, ""},
				{"nil-context", nil, 1, "context and host memory ceiling"},
				{"zero-memory", t.Context(), 0, "context and host memory ceiling"},
			} {
				t.Run(test.name, func(t *testing.T) {
					model, err := LoadRecognizer(test.ctx, directory, RecognizerSpec{Transducer: form.binding}, test.memory)
					if model != nil || err == nil {
						t.Fatalf("inadmissible load returned model %v and error %v", model, err)
					}
					if test.message == "" {
						if !errors.Is(err, context.Canceled) {
							t.Fatalf("canceled load accessed checkpoint: %v", err)
						}
					} else if !strings.Contains(err.Error(), test.message) {
						t.Fatalf("invalid admission accessed checkpoint: %v", err)
					}
				})
			}
		})
	}
}

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
