package discovery

import (
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
)

const remoteTestCodeCommit = "0123456789abcdef0123456789abcdef01234567"

// TestServableListsRemoteModelsWithTheirRefusal pins the remote branch
// of the servable predicate and the capability catalog: a declared remote
// model is present at its remote location without bytes on disk, listed
// refused by its key's variable name while the environment lacks the key
// and servable once it holds it, in both listings.
func TestServableListsRemoteModelsWithTheirRefusal(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	t.Setenv("OVERGO_REMOTE_DISCOVERY_TEST_KEY", "")
	declaration, err := remoteprovider.Declare(t.Context(), store, remoteprovider.Provider{
		Name: "fake", Endpoint: "https://fake.example/api/v1", KeyEnvironment: "OVERGO_REMOTE_DISCOVERY_TEST_KEY", Model: "vendor/model",
	}, remoteTestCodeCommit)
	if err != nil {
		t.Fatal(err)
	}
	entries, err := Servable(t.Context(), store, 16)
	if err != nil || len(entries) != 1 {
		t.Fatalf("servable = %+v, %v", entries, err)
	}
	if entry := entries[0]; entry.Model != declaration.Model || !entry.Present || entry.Location != declaration.Location ||
		entry.Recipe != declaration.Recipe.ID || !strings.Contains(entry.Stale, "OVERGO_REMOTE_DISCOVERY_TEST_KEY is not set") {
		t.Fatalf("keyless entry = %+v", entry)
	}
	catalog, _, err := CapabilityCatalog(t.Context(), store, 16, nil)
	if err != nil || len(catalog) != 1 || !catalog[0].Present || catalog[0].Location != declaration.Location ||
		len(catalog[0].Capabilities) != 1 || catalog[0].Capabilities[0].Task != recipe.TaskInference ||
		!strings.Contains(catalog[0].Capabilities[0].Stale, "OVERGO_REMOTE_DISCOVERY_TEST_KEY is not set") {
		t.Fatalf("keyless catalog = %+v, %v", catalog, err)
	}
	t.Setenv("OVERGO_REMOTE_DISCOVERY_TEST_KEY", "secret")
	entries, err = Servable(t.Context(), store, 16)
	if err != nil || len(entries) != 1 || entries[0].Stale != "" {
		t.Fatalf("keyed entry = %+v, %v", entries, err)
	}
	catalog, _, err = CapabilityCatalog(t.Context(), store, 16, nil)
	if err != nil || len(catalog) != 1 || catalog[0].Capabilities[0].Stale != "" {
		t.Fatalf("keyed catalog = %+v, %v", catalog, err)
	}
}
