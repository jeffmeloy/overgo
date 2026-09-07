package server

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/libraryintake"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

// storedProjectorPair finds, from the store alone, a model whose active
// projection recipe binds a projector with bytes on disk.
func storedProjectorPair(t *testing.T) (modelPath, projectorPath string) {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatalf("projector pair unavailable: %v", err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatalf("projector pair unavailable: %v", err)
	}
	defer store.Close()
	ctx := t.Context()
	memo := discovery.LoadMemo(ctx, store)
	entries, _, err := discovery.CapabilityCatalog(ctx, store, 256, memo)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if !entry.Present || entry.Location == "" {
			continue
		}
		if declared, ok, err := discovery.ActiveProjector(ctx, store, entry.Model, memo); err == nil && ok {
			return entry.Location, declared
		}
	}
	t.Fatal("projector pair unavailable: the store activates no projector with bytes on disk")
	return "", ""
}

// TestFrontPageLibraryProjector pins the projector half of the library
// lifecycle (professional GUI campaign, gui-register-projectors): the
// files a register names are sorted by their own metadata into the model
// and its projector, a projector named explicitly must declare itself
// one, and the Library tab carries the projector through register and
// validate.
func TestFrontPageLibraryProjector(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": reading model files is integration")
	}
	modelPath, projectorPath := storedProjectorPair(t)
	model, projector, err := libraryintake.ModelFiles(modelPath, projectorPath)
	if err != nil || model != modelPath || projector != projectorPath {
		t.Fatalf("libraryintake.ModelFiles(%s, %s) = (%s, %s, %v)", modelPath, projectorPath, model, projector, err)
	}
	if _, _, err := libraryintake.ModelFiles(projectorPath, ""); err == nil {
		t.Fatal("a projector registered as the model")
	}
	notes := filepath.Join(t.TempDir(), "notes.txt")
	if err := os.WriteFile(notes, []byte("not a projector"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := libraryintake.ModelFiles(modelPath, notes); err == nil {
		t.Fatal("a text file passed as the projector")
	}
	// The pair's own directory sorts into the same two files when it holds
	// exactly that pair; a directory holding more is refused by count.
	if dirModel, dirProjector, err := libraryintake.ModelFiles(filepath.Dir(modelPath), ""); err == nil && (dirModel != modelPath || dirProjector != projectorPath) {
		t.Fatalf("directory classification = (%s, %s)", dirModel, dirProjector)
	}

	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	quoted := func(value string) string {
		data, _ := json.Marshal(value)
		return string(data)
	}
	if refused := serveTestRequest(handler, http.MethodPost, "/library/register",
		`{"kind":"model","path":`+quoted(modelPath)+`,"projector":`+quoted(notes)+`}`); refused.Code == http.StatusOK {
		t.Fatalf("register accepted a text file as the projector: %s", refused.Body.String())
	}
	library := serveTestRequest(handler, http.MethodGet, "/mod/discovery.js", "").Body.String()
	for _, needle := range []string{`projector: job.projector`, `"register a local model"`, `stage.registered.projector`} {
		if !strings.Contains(library, needle) {
			t.Errorf("library module missing %q", needle)
		}
	}
}
