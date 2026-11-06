package modelintake

import (
	"strings"
	"testing"

	// gguf.Open serves a directory through the view the converter registers.
	_ "overgo/internal/hfconvert"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestASafetensorsDirectoryNothingRecordsResolves: a Hugging Face directory the store has never seen resolves from its
// own config.json and the prototype compiled for its architecture, takes its sampling from its own generation_config.json
// and says it is not verified; a directory whose model type nothing maps to a prototype is refused by that name.
func TestASafetensorsDirectoryNothingRecordsResolves(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	directory := testutil.Qwen35Directory(t)
	testutil.WriteTextFile(t, directory, "generation_config.json", `{"do_sample":true,"temperature":0.25}`)
	loaded := testcheck.Must(modelrecipe.ResolveActiveGGUF(t.Context(), store, directory))(t)
	defer loaded.Close()
	state := testcheck.Must(loaded.State())(t)
	if state.Verification != modelrecipe.StateNotVerified {
		t.Fatalf("state = %+v", state)
	}
	policy := testcheck.Must(loaded.RuntimePolicy())(t)
	if policy.Serving.Sampling.Temperature != 0.25 {
		t.Fatalf("the directory's sampling did not reach the policy: %+v", policy.Serving.Sampling)
	}
	unmapped := t.TempDir()
	testutil.WriteTextFile(t, unmapped, "config.json", `{"model_type":"needle"}`)
	if _, err := modelrecipe.ResolveActiveGGUF(t.Context(), store, unmapped); err == nil || !strings.Contains(err.Error(), `"needle"`) {
		t.Fatalf("a directory nothing maps resolved with %v", err)
	}
}
