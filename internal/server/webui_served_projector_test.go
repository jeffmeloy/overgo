package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/modelswap"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestModelJourneyServedProjector proves that a model whose store declares a
// projector serves image input through the swap proxy with no -mmproj on
// any command line (professional GUI campaign, gui-serve-projectors): the
// server resolves the active projection recipe's bytes itself, and the
// capability document the front page derives from reports the image
// modality. The smallest servable model with a projection activation and
// bytes on disk is served; without a store, a buildable server or such a
// model the check reports UNAVAILABLE. It serves a model, so it runs in the
// lane's journey mode with the serving owners, not with every server suite.
func TestModelJourneyServedProjector(t *testing.T) {
	if os.Getenv(webuilane.ModelJourneyEnvironment) != "1" {
		t.Skip(testskip.ShortIntegration + ": a served model runs through cmd/webui-lane -journeys")
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatalf("served projector unavailable: %v", err)
	}
	store := roots.Store
	if _, err := os.Stat(store); err != nil {
		t.Fatalf("served projector unavailable: no store at %s", store)
	}
	ctx := t.Context()
	name, location, projectorPath := smallestDeclaredProjector(t, ctx, store)
	if location == "" {
		t.Fatal("served projector unavailable: no servable model declares a projector with bytes on disk")
	}
	binary := filepath.Join(t.TempDir(), "overgo-server.exe")
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"build", "-o", binary, "overgo/cmd/server"}, Stdout: os.Stderr, Stderr: os.Stderr,
	})
	if err != nil || receipt.ExitCode != 0 {
		t.Fatalf("served projector unavailable: the server binary did not build: %v (exit %d)", err, receipt.ExitCode)
	}
	t.Logf("served projector: model %s at %s, declared projector %s", name, location, projectorPath)
	supervisor, err := modelswap.New(modelswap.ServerLauncher{Binary: binary, Store: store, Dir: filepath.Dir(store)}, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	repository, err := overgodb.OpenReadOnly(store)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	proxy := &modelswap.Proxy{
		Supervisor: supervisor, Resolver: &modelswap.CatalogResolver{Store: repository, Limit: 256},
		Default: modelswap.Servable{Name: name, Location: location},
	}
	front := httptest.NewServer(proxy)
	defer front.Close()

	var document struct {
		Model struct {
			Modalities map[string]bool `json:"modalities"`
			Media      struct {
				Accept   []string          `json:"accept"`
				Refusals map[string]string `json:"refusals"`
			} `json:"media"`
		} `json:"model"`
	}
	// The proxy admits the request once the served child has announced its
	// listener, so one request reads the capability document.
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, front.URL+"/workspace/manifest", nil)
	if err != nil {
		t.Fatal(err)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("the capability document did not answer: %v", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(response.Body)
		t.Fatalf("the capability document answered %d: %s", response.StatusCode, body)
	}
	if err := json.NewDecoder(response.Body).Decode(&document); err != nil {
		t.Fatal(err)
	}
	if !document.Model.Modalities["image"] {
		t.Fatalf("the served model %s refuses images although the store declares its projector: refusals=%v accept=%v",
			name, document.Model.Media.Refusals, document.Model.Media.Accept)
	}
	t.Logf("served projector: %s accepts %v", name, document.Model.Media.Accept)
}

// smallestDeclaredProjector picks the servable inference model with bytes
// on disk, a projection activation and that projector's bytes on disk,
// smallest model file first; empty when the store declares none.
func smallestDeclaredProjector(t *testing.T, ctx context.Context, store string) (name, location, projectorPath string) {
	t.Helper()
	db, err := overgodb.OpenReadOnly(store)
	if err != nil {
		t.Fatalf("served projector unavailable: the store did not open: %v", err)
	}
	defer db.Close()
	memo := discovery.LoadMemo(ctx, db)
	entries, _, err := discovery.CapabilityCatalog(ctx, db, 256, memo)
	if err != nil {
		t.Fatal(err)
	}
	var best int64
	for _, entry := range entries {
		if !entry.Present || entry.Location == "" {
			continue
		}
		inference := false
		for _, capability := range entry.Capabilities {
			inference = inference || capability.Task == recipe.TaskInference && capability.Stale == ""
		}
		info, statErr := os.Stat(entry.Location)
		if !inference || statErr != nil {
			continue
		}
		declared, ok, err := discovery.ActiveProjector(ctx, db, entry.Model, memo)
		if err != nil {
			t.Logf("served projector: %s declares a projector the store cannot serve: %v", filepath.Base(entry.Location), err)
			continue
		}
		if !ok {
			continue
		}
		if location == "" || info.Size() < best {
			name, location, projectorPath, best = filepath.Base(entry.Location), entry.Location, declared, info.Size()
		}
	}
	return name, location, projectorPath
}
