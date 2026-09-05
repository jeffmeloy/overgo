package libraryintake

import (
	"errors"
	"strings"
	"testing"
)

func TestLibraryProjectionRequiresExecutedSuite(t *testing.T) {
	// Missing execution fixtures must refuse before source inspection, loading
	// model bytes or publishing evidence; no repository is needed to refuse.
	id, execute, err := Validate(t.Context(), nil, "model.gguf", "projector.gguf", []string{"hello"}, 16)
	_, rejected := errors.AsType[refusal](err)
	if !rejected || !strings.Contains(err.Error(), "executed media suite") {
		t.Fatalf("missing projection execution contract: %v", err)
	}
	if execute != nil || id.String() != "" {
		t.Fatalf("refused intake returned an executable or identity: %v", id)
	}
}
