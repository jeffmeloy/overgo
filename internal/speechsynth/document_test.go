package speechsynth

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSpeechDocumentSourceCoverage(t *testing.T) {
	synth, err := LoadSynthesizer(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{
		"Hello. This speech was generated locally.",
		strings.Repeat("Every sentence belongs to this document. Preserve the final sentence!\n", 5),
		"  Café, naïve, 日本語.\r\n" + strings.Repeat("longword", 100) + "\t\n",
	} {
		plan, err := synth.PlanDocument(t.Context(), DocumentRequest{Text: text, Voice: "alba", Seed: 7})
		if err != nil {
			t.Fatal(err)
		}
		prior := 0
		for _, part := range plan.Segments {
			if part.Start != prior || part.End <= part.Start || part.End > len(text) {
				t.Fatalf("source gap: %+v after %d", part, prior)
			}
			span := text[part.Start:part.End]
			if !utf8.ValidString(span) || strings.TrimSpace(span) == "" {
				t.Fatalf("invalid span %q", span)
			}
			tokens, err := synth.tokenizer.Encode(span)
			if err != nil || len(tokens) != part.Tokens || part.Tokens > documentSegmentTokens || part.MaxFrames <= 0 {
				t.Fatalf("invalid segment %+v: %v", part, err)
			}
			prior = part.End
		}
		if prior != len(text) || plan.Request.Text != text {
			t.Fatal("source changed or was truncated")
		}
	}
}

func TestSpeechDocumentRejectsInvalidSource(t *testing.T) {
	synth, err := LoadSynthesizer(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, request := range []DocumentRequest{
		{Text: " \n", Voice: "alba"},
		{Text: string([]byte{255}), Voice: "alba"},
		{Text: "hello\x00world", Voice: "alba"},
		{Text: "Hello", Voice: "../alba"},
		{Text: "Hello", Voice: "alba", Temperature: math.NaN()},
		{Text: "Hello", Voice: "alba", Temperature: math.Inf(1)},
	} {
		if _, err := synth.PlanDocument(t.Context(), request); err == nil {
			t.Fatalf("accepted invalid document: %+v", request)
		}
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cause := errors.New("document stopped")
	cancel(cause)
	if _, err := synth.PlanDocument(ctx, DocumentRequest{Text: "Hello", Voice: "alba"}); !errors.Is(err, cause) {
		t.Fatalf("lost cancellation: %v", err)
	}
}
