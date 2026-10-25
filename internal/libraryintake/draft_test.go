package libraryintake

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/gguf"
)

// writeArchitectureFixture writes a GGUF declaring architecture in dir.
func writeArchitectureFixture(t *testing.T, dir, name, architecture string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	metadata := []gguf.Metadata{{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: architecture}}}
	if err := gguf.WriteFileExclusive(path, metadata, nil, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestDraftBesideTheModel holds the library's draft discovery: a draft GGUF in
// a model's directory is the model's draft, not a second model, so the
// directory still names its one model; a model with no draft beside it has
// none, and two drafts are refused rather than guessed between.
func TestDraftBesideTheModel(t *testing.T) {
	dir := t.TempDir()
	model := writeArchitectureFixture(t, dir, "model.gguf", "qwen4exp")
	if draft, err := draftBeside(model); err != nil || draft != "" {
		t.Fatalf("a model alone has draft %q (%v)", draft, err)
	}
	draft := writeArchitectureFixture(t, dir, "draft.gguf", "qwen4exp-mtp")
	if found, err := draftBeside(model); err != nil || found != draft {
		t.Fatalf("draftBeside = %q (%v), want %q", found, err, draft)
	}
	if found, projector, err := ModelFiles(dir, ""); err != nil || found != model || projector != "" {
		t.Fatalf("ModelFiles over a model and its draft = %q, %q (%v), want the model alone", found, projector, err)
	}
	writeArchitectureFixture(t, dir, "second-draft.gguf", "qwen4exp-mtp")
	if _, err := draftBeside(model); err == nil || !strings.Contains(err.Error(), "2 draft GGUF files") {
		t.Fatalf("two drafts beside the model: %v", err)
	}
}
