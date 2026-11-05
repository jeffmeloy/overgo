package modelintake

import (
	"strings"
	"testing"

	// gguf.Open serves a directory through the view the converter registers.
	_ "overgo/internal/hfconvert"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestASafetensorsDirectoryNothingRecordsResolves: a Hugging Face directory the store has never seen resolves from its
// own config.json and the prototype compiled for its architecture, takes its sampling from its own generation_config.json
// and says it is not verified; a directory whose model type nothing maps to a prototype is refused by that name.
func TestASafetensorsDirectoryNothingRecordsResolves(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	directory := testutil.Qwen35Directory(t)
	testutil.WriteTextFile(t, directory, "generation_config.json", `{"do_sample":true,"temperature":0.25}`)
	loaded, err := modelrecipe.ResolveActiveGGUF(t.Context(), store, directory)
	testcheck.NoError(t, err)
	defer loaded.Close()
	state, err := loaded.State()
	testcheck.NoError(t, err)
	if state.Verification != modelrecipe.StateNotVerified {
		t.Fatalf("state = %+v", state)
	}
	policy, err := loaded.RuntimePolicy()
	testcheck.NoError(t, err)
	if policy.Serving.Sampling.Temperature != 0.25 {
		t.Fatalf("the directory's sampling did not reach the policy: %+v", policy.Serving.Sampling)
	}
	unmapped := t.TempDir()
	testutil.WriteTextFile(t, unmapped, "config.json", `{"model_type":"needle"}`)
	if _, err := modelrecipe.ResolveActiveGGUF(t.Context(), store, unmapped); err == nil || !strings.Contains(err.Error(), `"needle"`) {
		t.Fatalf("a directory nothing maps resolved with %v", err)
	}
}
