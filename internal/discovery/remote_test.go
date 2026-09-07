package discovery

import (
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
)

const remoteTestCodeCommit = "0123456789abcdef0123456789abcdef01234567"

// TestServableListsRemoteModelsWithTheirRefusal: remote branch of Servable
// + CapabilityCatalog. Declared remote model: present at its remote
// location, no bytes on disk; refused by key variable name without the
// key; servable with it; retired -> gone from both listings.
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
		!strings.Contains(catalog[0].Capabilities[0].Stale, "OVERGO_REMOTE_DISCOVERY_TEST_KEY is not set") || catalog[0].KeyEnvironment != "OVERGO_REMOTE_DISCOVERY_TEST_KEY" {
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
	// Retired: out of both listings; history kept.
	if err := remoteprovider.Retire(t.Context(), store, 16, declaration.Location, remoteTestCodeCommit, "the provider endpoint closed"); err != nil {
		t.Fatal(err)
	}
	if entries, err := Servable(t.Context(), store, 16); err != nil || len(entries) != 0 {
		t.Fatalf("retired servable = %+v, %v", entries, err)
	}
	if catalog, _, err := CapabilityCatalog(t.Context(), store, 16, nil); err != nil || len(catalog) != 0 {
		t.Fatalf("retired catalog = %+v, %v", catalog, err)
	}
	if err := remoteprovider.Retire(t.Context(), store, 16, "remote://nobody/model", remoteTestCodeCommit, "absent"); err == nil ||
		!strings.Contains(err.Error(), "no declared model") {
		t.Fatalf("retiring an undeclared model: %v", err)
	}
}
