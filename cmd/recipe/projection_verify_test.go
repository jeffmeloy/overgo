package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestProjectionVerificationRequiresExecution(t *testing.T) {
	err := verifyCapability(t.TempDir(), "unused", recipe.TaskProjection, capability{}, "")
	if err == nil || !strings.Contains(err.Error(), "executor") {
		t.Fatalf("compile-only projection reached verification: %v", err)
	}
	capabilities := projector.SessionCapabilities{Image: true, MultiImage: true, Audio: true, Video: true, MediaHistory: true}
	file := projectionFile{Kind: "image", Path: "fixture.png", Artifact: testutil.ArtifactID(t, artifact.KindFile, "image")}
	audio := projectionFile{Kind: "audio-f32le", Path: "fixture.f32", Artifact: testutil.ArtifactID(t, artifact.KindFile, "audio")}
	inputs := []projectionInput{
		{Kind: "image", Files: []projectionFile{file}, Text: []string{"", "describe"}},
		{Kind: "images", Files: []projectionFile{file, file}, Text: []string{"", "", "count"}},
		{Kind: "audio", Files: []projectionFile{audio}, Text: []string{"", "transcribe"}},
		{Kind: "video", Files: []projectionFile{file, file}, Text: []string{"", "describe"}, FPS: 1},
		{Kind: "mixed", Files: []projectionFile{file, audio}, Text: []string{"", "", "transcribe"}},
	}
	encode := func(input projectionInput) string {
		data, err := json.Marshal(input)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	var suite evaluation.ExactSuite
	for _, input := range inputs {
		suite.Cases = append(suite.Cases, evaluation.ExactCase{Name: input.Kind, Prompt: encode(input)})
	}
	if err := requireProjectionCoverage(suite, capabilities); err != nil {
		t.Fatal(err)
	}
	for index, input := range inputs {
		t.Run(input.Kind, func(t *testing.T) {
			partial := suite
			partial.Cases = slices.Delete(slices.Clone(suite.Cases), index, index+1)
			if err := requireProjectionCoverage(partial, capabilities); err == nil {
				t.Fatal("missing declared modality accepted")
			}
			input.Files = nil
			if _, err := decodeProjectionInput(encode(input), capabilities); err == nil {
				t.Fatal("empty fixture accepted")
			}
		})
	}
	for _, invalid := range []projectionInput{
		{Kind: "unknown", Files: []projectionFile{file}, Text: []string{"", "describe"}},
		{Kind: "image", Files: []projectionFile{audio}, Text: []string{"", "describe"}},
		{Kind: "images", Files: []projectionFile{file}, Text: []string{"", "describe"}},
		{Kind: "video", Files: []projectionFile{file, file}, Text: []string{"", "describe"}},
		{Kind: "mixed", Files: []projectionFile{file, audio}, Text: []string{"", "describe"}},
		{Kind: "image", Files: []projectionFile{{Kind: "image", Path: "unbound", Artifact: testutil.ArtifactID(t, artifact.KindOutput, "wrong-kind")}}, Text: []string{"", "describe"}},
	} {
		if _, err := decodeProjectionInput(encode(invalid), capabilities); err == nil {
			t.Fatalf("invalid input accepted: %+v", invalid)
		}
	}
	// These fail before any projector or language execution; the embedded nil
	// session makes accidental execution panic instead of manufacturing a pass.
	runtime := &projectedExactRuntime{projection: projectionCapabilitiesOnly{capabilities: capabilities}}
	file.Path = filepath.Join(t.TempDir(), "fixture.png")
	input := projectionInput{Kind: "image", Files: []projectionFile{file}, Text: []string{"", "describe"}}
	if _, _, err := runtime.Generate(t.Context(), encode(input), inference.GenerateOptions{}); err == nil {
		t.Fatal("missing fixture accepted")
	}
	if err := os.WriteFile(file.Path, []byte("changed bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := runtime.Generate(t.Context(), encode(input), inference.GenerateOptions{}); err == nil || !strings.Contains(err.Error(), "differ") {
		t.Fatalf("changed fixture accepted: %v", err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, _, err := runtime.Generate(ctx, encode(input), inference.GenerateOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
}

type projectionCapabilitiesOnly struct {
	projector.Session
	capabilities projector.SessionCapabilities
}

func (session projectionCapabilitiesOnly) Capabilities() projector.SessionCapabilities {
	return session.capabilities
}
