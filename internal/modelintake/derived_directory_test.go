package modelintake

import (
	"strings"
	"testing"

	// gguf.Open serves a directory through the view the converter registers.
	_ "overgo/internal/hfconvert"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestASafetensorsDirectoryNothingRecordsResolves: a Hugging Face directory the store has never seen resolves from its
// own config.json and the prototype compiled for its architecture, and says it is not verified; a directory whose model
// type nothing maps to a prototype is refused by that name.
func TestASafetensorsDirectoryNothingRecordsResolves(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	loaded, err := modelrecipe.ResolveActiveGGUF(t.Context(), store, testutil.Qwen35Directory(t))
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	state, err := loaded.State()
	if err != nil {
		t.Fatal(err)
	}
	if state.Verification != modelrecipe.StateNotVerified {
		t.Fatalf("state = %+v", state)
	}
	unmapped := t.TempDir()
	testutil.WriteTextFile(t, unmapped, "config.json", `{"model_type":"needle"}`)
	if _, err := modelrecipe.ResolveActiveGGUF(t.Context(), store, unmapped); err == nil || !strings.Contains(err.Error(), `"needle"`) {
		t.Fatalf("a directory nothing maps resolved with %v", err)
	}
}
