package libraryintake

import (
	"errors"
	"strings"
	"testing"
)

func TestLibraryProjectionRefusesUndeclaredProjector(t *testing.T) {
	// A projector that cannot be read declares nothing and refuses before
	// source inspection, loading model bytes or publishing evidence; no
	// repository is needed to refuse.
	id, execute, err := Validate(t.Context(), nil, "model.gguf", "projector.gguf", []string{"hello"}, 16)
	_, rejected := errors.AsType[refusal](err)
	if !rejected || !strings.Contains(err.Error(), "does not declare a projector") {
		t.Fatalf("undeclared projector contract: %v", err)
	}
	if execute != nil || id.String() != "" {
		t.Fatalf("refused intake returned an executable or identity: %v", id)
	}
}
