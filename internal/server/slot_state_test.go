package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/objectstore"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// TestSlotSaveRestoresPromptCache saves a slot's prompt state on one server,
// restores it on a fresh runner after a restart, and demands that the restored
// prefix is served from the cache with the greedy tokens of an uncached run.
// A prompt equal to the saved tokens recomputes one token. Overwriting a name
// retires the state no other name holds. Another model, a bad slot, action or
// filename, a busy slot, more than one slot and a server without a store are
// refused.
func TestSlotSaveRestoresPromptCache(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	configured := func(config *Config) { config.OvergoDBPath = directory }
	serveRunner := func(runner *inference.Runner, configure ...func(*Config)) *Handler {
		t.Helper()
		// The served model is registered, as a model the server resolves from the store is.
		if _, err := artifact.Publish(t.Context(), store, artifact.Batch{
			Key: "test/slot-model/" + runner.ModelID().String(), Artifacts: []artifact.Descriptor{{ID: runner.ModelID()}},
		}); err != nil {
			t.Fatal(err)
		}
		return newTestHandlerForRepository(t, store, runner, append([]func(*Config){configured}, configure...)...)
	}
	serve := func(contextLength uint32, configure ...func(*Config)) (*Handler, *inference.Runner) {
		t.Helper()
		runner := openSessionRunner(t, contextLength)
		return serveRunner(runner, configure...), runner
	}
	complete := func(handler *Handler, prompt []tokenizer.TokenID) nativeCompletionResponse {
		t.Helper()
		response := serveTestRequest(handler, http.MethodPost, "/completion", marshalTestJSON(t, map[string]any{
			"prompt": prompt, "n_predict": 3, "cache_prompt": true, "temperature": 0,
			"ignore_eos": true, "return_tokens": true,
		}))
		var result nativeCompletionResponse
		if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &result) != nil || len(result.Tokens) != 3 {
			t.Fatalf("completion status=%d body=%s", response.Code, response.Body.String())
		}
		return result
	}
	slot := func(handler *Handler, target, filename string, status int) map[string]any {
		t.Helper()
		response := serveTestRequest(handler, http.MethodPost, target, marshalTestJSON(t, map[string]string{"filename": filename}))
		var result map[string]any
		if response.Code != status || json.Unmarshal(response.Body.Bytes(), &result) != nil {
			t.Fatalf("%s %q status=%d want %d body=%s", target, filename, response.Code, status, response.Body.String())
		}
		return result
	}
	const save, restore = "/slots/0?action=save", "/slots/0?action=restore"
	objectFile := func(id string) string {
		parsed, err := artifact.ParseID(id)
		if err != nil {
			t.Fatal(err)
		}
		return filepath.Join(directory, objectstore.Directory, parsed.Kind().String(), parsed.DigestHex())
	}

	first, firstRunner := serve(64)
	prompt := []tokenizer.TokenID{1, 4, 5, 6, 7, 4, 5}
	served := complete(first, prompt)
	// The pool keeps the prompt and every generated token but the last, whose state is not computed yet.
	saved := slices.Concat(prompt, served.Tokens[:2])
	savedState := slot(first, save, "prefix", http.StatusOK)
	stateID, _ := savedState["id"].(string)
	if savedState["n_saved"] != float64(len(saved)) || savedState["n_written"].(float64) <= 0 || stateID == "" {
		t.Fatalf("save = %v, want %d tokens", savedState, len(saved))
	}
	if copied := slot(first, save, "copy", http.StatusOK); copied["id"] != stateID {
		t.Fatalf("the same state under a second name = %v, want %s", copied, stateID)
	}
	parsedState, err := artifact.ParseID(stateID)
	if err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(t.Context(), parsedState)
	if err != nil || len(parents) != 2 {
		t.Fatalf("saved state lineage = %v err=%v, want the model and the recipe", parents, err)
	}
	for _, refused := range []struct {
		target, filename string
		status           int
	}{
		{"/slots/0?action=erase", "prefix", http.StatusBadRequest},
		{"/slots/1?action=save", "prefix", http.StatusBadRequest},
		{"/slots/x?action=save", "prefix", http.StatusBadRequest},
		{save, "../prefix", http.StatusBadRequest},
		{save, "a/b", http.StatusBadRequest},
		{save, "", http.StatusBadRequest},
		{restore, "never-saved", http.StatusNotFound},
	} {
		slot(first, refused.target, refused.filename, refused.status)
	}
	lease, leased := first.sessions.TryLease(0)
	if !leased {
		t.Fatal("slot 0 is not free")
	}
	slot(first, save, "prefix", http.StatusConflict)
	_ = lease.Release()

	// A restart: the handler and its runner close, and a fresh runner of the same model restores the state.
	_ = first.Close()
	_ = firstRunner.Close()
	second, _ := serve(64)
	// An empty pool has nothing to save, and the names keep what they hold.
	slot(second, save, "prefix", http.StatusConflict)
	restored := slot(second, restore, "prefix", http.StatusOK)
	if restored["n_restored"] != float64(len(saved)) || restored["id"] != stateID || restored["n_read"] != savedState["n_written"] {
		t.Fatalf("restore = %v, save = %v", restored, savedState)
	}
	extended := slices.Concat(saved, []tokenizer.TokenID{6, 7})
	continued := complete(second, extended)
	fresh, _ := serve(64)
	uncached := complete(fresh, extended)
	if continued.TokensCached != len(saved) || uncached.TokensCached != 0 || !slices.Equal(continued.Tokens, uncached.Tokens) {
		t.Fatalf("restored run cached %d tokens %v, uncached run cached %d tokens %v; want %d cached and equal tokens",
			continued.TokensCached, continued.Tokens, uncached.TokensCached, uncached.Tokens, len(saved))
	}
	// A prompt equal to the restored tokens recomputes the last one, whose hidden row the state does not keep.
	slot(second, restore, "prefix", http.StatusOK)
	exact := complete(second, saved)
	freshExact, _ := serve(64)
	if uncachedExact := complete(freshExact, saved); exact.TokensCached != len(saved)-1 || !slices.Equal(exact.Tokens, uncachedExact.Tokens) {
		t.Fatalf("exact restored run cached %d tokens %v, uncached %v; want %d cached and equal tokens",
			exact.TokensCached, exact.Tokens, uncachedExact.Tokens, len(saved)-1)
	}

	// Overwriting a name keeps the state another name holds and retires the state no name holds.
	overwritten := slot(second, save, "prefix", http.StatusOK)
	nextID, _ := overwritten["id"].(string)
	if nextID == "" || nextID == stateID {
		t.Fatalf("overwrite = %v, want a state other than %s", overwritten, stateID)
	}
	if repeated := slot(second, save, "prefix", http.StatusOK); repeated["id"] != nextID {
		t.Fatalf("a repeated save = %v, want %s", repeated, nextID)
	}
	if _, err := os.Stat(objectFile(stateID)); err != nil {
		t.Fatalf("the state the name copy holds is gone: %v", err)
	}
	slot(second, save, "copy", http.StatusOK)
	locations, err := store.Locations(t.Context(), parsedState)
	if _, statErr := os.Stat(objectFile(stateID)); !errors.Is(statErr, os.ErrNotExist) || err != nil || len(locations) != 0 {
		t.Fatalf("the state no name holds keeps file error %v and locations %v (err %v)", statErr, locations, err)
	}
	if _, err := os.Stat(objectFile(nextID)); err != nil {
		t.Fatalf("the state both names hold is gone: %v", err)
	}

	// Another model never finds the state: its names are its own.
	other, _ := serve(128)
	slot(other, restore, "prefix", http.StatusNotFound)
	// The same model under another recipe finds the state and refuses it, since the recipe binds its numeric path.
	path := testutil.HermeticLlamaGGUFWith(t, 64, testutil.GGUFScalar("tokenizer.chat_template", gguf.ValueTypeString, joinedMessagesTemplate))
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(path, recipe.PlacementHybrid, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference)
	if err != nil {
		t.Fatal(err)
	}
	otherRecipe, err := inference.OpenWithProgram(t.Context(), &loaded, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer otherRecipe.Close()
	if otherRecipe.ModelID() != firstRunner.ModelID() {
		t.Fatalf("the other recipe serves model %s, not %s", otherRecipe.ModelID(), firstRunner.ModelID())
	}
	refused := slot(serveRunner(otherRecipe), restore, "prefix", http.StatusConflict)
	if message, _ := refused["error"].(map[string]any)["message"].(string); !strings.Contains(message, "recipe") {
		t.Fatalf("the other recipe's refusal = %v, want the recipe named", refused)
	}
	// The prompt cache belongs to the runner, so more than one slot is refused.
	multiple, _ := serve(64, func(config *Config) { config.MaxConcurrent = 2 })
	slot(multiple, save, "prefix", http.StatusNotImplemented)
	// A server without a store has nowhere to keep a state.
	slot(newTestHandler(t, openSessionRunner(t, 64)), save, "prefix", http.StatusServiceUnavailable)
}
