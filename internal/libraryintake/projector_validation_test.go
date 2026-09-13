package libraryintake

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/gguf"
)

func writeProjectorFixture(t *testing.T, name string, projector bool) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	metadata := []gguf.Metadata{{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "clip"}}}
	if projector {
		metadata = append(metadata, gguf.Metadata{Key: "clip.projector_type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "mlp"}})
	}
	if err := gguf.WriteFileExclusive(path, metadata, nil, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestValidateTextWithProjector: a projector beside the model no longer
// refuses text validation; a file that declares no projection is refused
// before the source or the model is inspected, and a declared projector
// lets validation proceed to the model itself.
func TestValidateTextWithProjector(t *testing.T) {
	model := filepath.Join(t.TempDir(), "model.gguf")
	_, execute, err := Validate(t.Context(), nil, model, writeProjectorFixture(t, "not-a-projector.gguf", false), []string{"hello"}, 16)
	if _, rejected := errors.AsType[refusal](err); !rejected || !strings.Contains(err.Error(), "does not declare a projector") || execute != nil {
		t.Fatalf("undeclared projector: %v", err)
	}
	_, execute, err = Validate(t.Context(), nil, model, writeProjectorFixture(t, "projector.gguf", true), []string{"hello"}, 16)
	if err == nil || execute != nil {
		t.Fatalf("validation over an absent model returned an executable: %v", err)
	}
	if strings.Contains(err.Error(), "executed media suite") || strings.Contains(err.Error(), "projector") {
		t.Fatalf("a declared projector still refuses text validation: %v", err)
	}
}
