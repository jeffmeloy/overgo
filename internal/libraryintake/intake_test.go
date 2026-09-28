package libraryintake

import (
	"errors"
	"strings"
	"testing"
)

func TestLibraryProjectionRefusesUndeclaredProjector(t *testing.T) {
	// A projector that cannot be read declares nothing and refuses before
	// source inspection, loading model bytes or publishing evidence; no
	// repository is needed to refuse. Reaching the refusal also decodes the
	// embedded prompt policy, which carries a doc field the decoder accepts.
	id, execute, err := Validate(t.Context(), nil, "model.gguf", "projector.gguf")
	_, rejected := errors.AsType[refusal](err)
	if !rejected || !strings.Contains(err.Error(), "does not declare a projector") {
		t.Fatalf("undeclared projector contract: %v", err)
	}
	if execute != nil || id.String() != "" {
		t.Fatalf("refused intake returned an executable or identity: %v", id)
	}
}
