package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
)

func TestTranscriptionInputsBindContainerAndSelector(t *testing.T) {
	base := t.TempDir()
	const payload = "immutable encoded audio"
	if err := os.WriteFile(filepath.Join(base, "audio.wav"), []byte(payload), 0600); err != nil {
		t.Fatal(err)
	}
	id, _, err := artifact.Identify(artifact.KindFile, strings.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	spec := transcriptionResourceManifestInput{
		Name: "utterance", Path: "audio.wav", Origin: dataset.AudioPayloadOrigin{Container: id},
		Policy: dataset.AudioInspectionPolicy{MaximumEncodedBytes: uint64(len(payload)), MaximumSamples: 1, ClipThreshold: 1,
			Admission: recipecontract.AudioAdmissionPolicy{MinimumChannels: 1, MaximumChannels: 1, MaximumAbsoluteDCOffset: 1}},
	}
	inputs, err := readTranscriptionInputs(t.Context(), base, []transcriptionResourceManifestInput{spec, spec})
	if err != nil || len(inputs) != 2 || string(inputs[0].Data) != payload || string(inputs[1].Data) != payload || inputs[0].Origin != spec.Origin {
		t.Fatalf("bound inputs=%+v error=%v", inputs, err)
	}
	inputs[0].Data[0]++
	if string(inputs[1].Data) != payload {
		t.Fatal("duplicate selectors share mutable input bytes")
	}
	for _, test := range []struct {
		name   string
		change func(*transcriptionResourceManifestInput)
	}{
		{"wrong content", func(s *transcriptionResourceManifestInput) {
			s.Origin.Container, _, _ = artifact.Identify(artifact.KindFile, strings.NewReader("other audio"))
		}},
		{"wrong ordinal", func(s *transcriptionResourceManifestInput) { s.Origin.ValueIndex = 1 }},
		{"oversize", func(s *transcriptionResourceManifestInput) { s.Policy.MaximumEncodedBytes-- }},
		{"missing path", func(s *transcriptionResourceManifestInput) { s.Path = "missing.wav" }},
		{"missing container", func(s *transcriptionResourceManifestInput) { s.Origin.Container = artifact.ID{} }},
	} {
		t.Run(test.name, func(t *testing.T) {
			invalid := spec
			test.change(&invalid)
			if _, err := readTranscriptionInputs(t.Context(), base, []transcriptionResourceManifestInput{invalid}); err == nil {
				t.Fatal("invalid source admitted")
			}
		})
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, err := readTranscriptionInputs(ctx, base, []transcriptionResourceManifestInput{spec}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled source read: %v", err)
	}
}
