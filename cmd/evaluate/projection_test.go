package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/inference"
)

// TestMediaPathStaysInsideItsRoot holds a case's media name to the media root:
// a file of the root resolves, and a name that climbs out of it, is absolute,
// or names no file does not.
func TestMediaPathStaysInsideItsRoot(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "clip.mp4"), []byte("clip"), 0o600); err != nil {
		t.Fatal(err)
	}
	if path, err := mediaPath(root, "clip.mp4"); err != nil || path != filepath.Join(root, "clip.mp4") {
		t.Fatalf("clip.mp4 resolved to (%q, %v)", path, err)
	}
	for name, media := range map[string]string{
		"a climbing name": "../clip.mp4",
		"an absolute":     filepath.Join(root, "clip.mp4"),
		"a missing file":  "other.mp4",
		"an empty name":   "",
	} {
		if path, err := mediaPath(root, media); err == nil {
			t.Errorf("%s resolved to %q", name, path)
		}
	}
	if path, err := mediaPath("", "clip.mp4"); err == nil {
		t.Errorf("a name resolved to %q with no media root", path)
	}
}

// TestProjectionRuntimeAnswersOnlyAQuestionAboutNamedMedia holds the runtime
// to its contract: a prompt that names no video is refused, and no continuation
// likelihood is scored, before any model is asked.
func TestProjectionRuntimeAnswersOnlyAQuestionAboutNamedMedia(t *testing.T) {
	t.Parallel()
	var runtime projectionRuntime
	if _, _, err := runtime.Generate(t.Context(), "Which description matches?", inference.GenerateOptions{}); err == nil {
		t.Error("a prompt that names no video was answered")
	}
	if _, _, err := runtime.Generate(t.Context(), "<overgo-media kind=video name=../clip.mp4>\nWhich?", inference.GenerateOptions{}); err == nil {
		t.Error("a video outside the media root was answered")
	}
	if _, err := runtime.ScoreContinuations(t.Context(), "prompt", []string{"candidate"}); err == nil {
		t.Error("a projection evaluation scored a continuation")
	}
}
