package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
	"overgo/internal/recipe"
)

// TestIdleShellAnswersWhileNothingServes pins the idle shell: the client and
// its boot routes answer, the catalog lists the store's rows, and an API
// route refuses with the reason.
func TestIdleShellAnswersWhileNothingServes(t *testing.T) {
	model, err := artifact.IdentifyBytes(artifact.KindModel, []byte("idle-model"))
	if err != nil {
		t.Fatal(err)
	}
	served, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("idle-recipe"))
	if err != nil {
		t.Fatal(err)
	}
	shell := &IdleShell{Catalog: func(context.Context) ([]discovery.CatalogEntry, bool, error) {
		return []discovery.CatalogEntry{{Model: model, Location: "C:/models/idle.gguf", Present: true,
			Capabilities: []discovery.Capability{{Task: recipe.TaskInference, Recipe: served, Tier: "verified"}}}}, true, nil
	}}
	front := httptest.NewServer(shell)
	defer front.Close()

	page := get(t, front.URL+"/app.html")
	if page.StatusCode != http.StatusOK || !strings.Contains(page.Header.Get("Content-Type"), "text/html") || page.Header.Get("Content-Security-Policy") == "" {
		t.Fatalf("shell = %d %q", page.StatusCode, page.Header)
	}
	var health struct {
		Status string `json:"status"`
		Model  string `json:"model"`
	}
	decode(t, front.URL+"/health", &health)
	if health.Status != "ok" || health.Model != "" {
		t.Fatalf("health = %+v", health)
	}
	var manifest workspaceManifestResponse
	decode(t, front.URL+"/workspace/manifest", &manifest)
	if len(manifest.Tabs) == 0 || manifest.Model != nil {
		t.Fatalf("manifest = %+v", manifest)
	}
	for _, tab := range manifest.Tabs {
		if library := tab.ID == "library"; tab.Enabled != library || (tab.Refusal != idleRefusal) == !library {
			t.Fatalf("tab %s = %+v", tab.ID, tab)
		}
	}
	var catalog struct {
		Models    []catalogModel `json:"models"`
		Truncated bool           `json:"truncated"`
	}
	decode(t, front.URL+"/catalog/models", &catalog)
	if len(catalog.Models) != 1 || !catalog.Truncated || catalog.Models[0].Recipe != served.String() || !catalog.Models[0].Present {
		t.Fatalf("catalog = %+v", catalog)
	}
	refused := get(t, front.URL+"/v1/chat/completions")
	if refused.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("api route while nothing serves = %d", refused.StatusCode)
	}
	var envelope struct {
		Error struct {
			Code    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(refused.Body).Decode(&envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Error.Code != "no_model_serves" || envelope.Error.Message != idleRefusal {
		t.Fatalf("refusal = %+v", envelope)
	}
	missing := get(t, front.URL+"/absent.js")
	if missing.StatusCode != http.StatusNotFound {
		t.Fatalf("absent asset = %d", missing.StatusCode)
	}

	broken := &IdleShell{Catalog: func(context.Context) ([]discovery.CatalogEntry, bool, error) {
		return nil, false, errors.New("store gone")
	}}
	brokenFront := httptest.NewServer(broken)
	defer brokenFront.Close()
	if failed := get(t, brokenFront.URL+"/catalog/models"); failed.StatusCode != http.StatusInternalServerError {
		t.Fatalf("catalog failure = %d", failed.StatusCode)
	}
	if refused := post(t, brokenFront.URL+"/library/register", `{"kind":"provider","name":"p","endpoint":"http://127.0.0.1:1","key_environment":"K","models":["m"]}`); refused.StatusCode != http.StatusNotImplemented {
		t.Fatalf("library write without a store = %d", refused.StatusCode)
	}
}

func TestIdleShellWriterLifetime(t *testing.T) {
	root := t.TempDir()
	writer, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	lock, err := processlock.Acquire(filepath.Join(root, "overgodb.lock"), 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	shell := &IdleShell{
		Repository: writer,
		Intake: LibraryIntake{
			ModelFiles: func(path, projector string) (string, string, error) { return path, projector, nil },
			Register: func(_ context.Context, _ *overgodb.Store, path, _ string) (map[string]any, error) {
				other, err := overgodb.Open(root)
				if err != nil {
					t.Fatalf("idle intake handle reserves writer: %v", err)
				}
				_ = other.Close()
				return nil, errors.New("intake refused")
			},
		},
	}
	requestWrite := func(ctx context.Context) *httptest.ResponseRecorder {
		response := httptest.NewRecorder()
		shell.ServeHTTP(response, httptest.NewRequestWithContext(ctx, http.MethodPost, "/library/register", strings.NewReader(`{"kind":"model","path":"local.gguf"}`)))
		return response
	}
	// A competing writer does not prevent the idle workbench from answering.
	response := httptest.NewRecorder()
	shell.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/health", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("idle health with another writer = %d", response.Code)
	}
	// A request whose caller has already left waits for no writer: the
	// contention is reported at once with the store busy.
	ctx, leave := context.WithCancelCause(t.Context())
	leave(errors.New("the request's caller left"))
	busy := requestWrite(ctx)
	if busy.Code != http.StatusServiceUnavailable || !strings.Contains(busy.Body.String(), `"store_busy"`) {
		t.Fatalf("live contention = %d %s", busy.Code, busy.Body)
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	refused := requestWrite(t.Context())
	if strings.Contains(refused.Body.String(), `"store_busy"`) || !strings.Contains(refused.Body.String(), "intake refused") {
		t.Fatalf("retry after writer closes = %d %s", refused.Code, refused.Body)
	}
	// Even a refused intake releases the writer before the next operation.
	writer, err = overgodb.Open(root)
	if err != nil {
		t.Fatalf("idle workbench retained writer after request: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	shell.Repository = writer
	unavailable := requestWrite(t.Context())
	if !strings.Contains(unavailable.Body.String(), `"store_unavailable"`) || strings.Contains(unavailable.Body.String(), `"store_busy"`) {
		t.Fatalf("non-lock error misclassified = %d %s", unavailable.Code, unavailable.Body)
	}
}

// TestIdleShellLibraryWritesOverTheOpenedStore pins the cold page's
// library: a declaration, a listing and a retirement go through the
// intake over the borrowed store, and a local
// registration reaches the model intake without a validation step.
func TestIdleShellLibraryWritesOverTheOpenedStore(t *testing.T) {
	root := t.TempDir()
	repository, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer repository.Close()
	var declared, retired, registered int
	shell := &IdleShell{
		Repository: repository,
		Intake: LibraryIntake{
			ModelFiles: func(path, projector string) (string, string, error) { return path, projector, nil },
			Register: func(_ context.Context, store *overgodb.Store, path, projector string) (map[string]any, error) {
				registered++
				return map[string]any{"path": path}, nil
			},
			DeclareProvider: func(_ context.Context, store *overgodb.Store, declaration ProviderDeclaration) ([]DeclaredProvider, error) {
				declared++
				if store == nil || declaration.Name != "page" || len(declaration.Models) != 1 {
					t.Fatalf("declaration = %+v over %v", declaration, store)
				}
				return []DeclaredProvider{{Location: "remote://page/" + declaration.Models[0], Refusal: "key absent"}}, nil
			},
			ListProviderModels: func(_ context.Context, endpoint, variable string) ([]ProviderModel, error) {
				return []ProviderModel{{ID: "vendor/one", ContextLength: 4096}}, nil
			},
			RetireProvider: func(_ context.Context, store *overgodb.Store, location, reason string) error { retired++; return nil },
		},
	}
	front := httptest.NewServer(shell)
	defer front.Close()

	var declaration struct {
		Declared []DeclaredProvider `json:"declared"`
	}
	decodeResponse(t, post(t, front.URL+"/library/register", `{"kind":"provider","name":"page","endpoint":"http://127.0.0.1:1","key_environment":"K","models":["vendor/one"]}`), &declaration)
	if len(declaration.Declared) != 1 || declaration.Declared[0].Location != "remote://page/vendor/one" {
		t.Fatalf("declared = %+v", declaration)
	}
	var listing struct {
		Models []ProviderModel `json:"models"`
	}
	decode(t, front.URL+"/library/providers/models?endpoint=http://127.0.0.1:1&key_environment=K", &listing)
	if len(listing.Models) != 1 || listing.Models[0].ContextLength != 4096 {
		t.Fatalf("listing = %+v", listing)
	}
	if retirement := post(t, front.URL+"/library/providers/retire", `{"location":"remote://page/vendor/one","reason":"closed"}`); retirement.StatusCode != http.StatusOK {
		t.Fatalf("retirement = %d", retirement.StatusCode)
	}
	if registration := post(t, front.URL+"/library/register", `{"kind":"model","path":"C:/models/local.gguf"}`); registration.StatusCode != http.StatusOK {
		t.Fatalf("registration = %d", registration.StatusCode)
	}
	if declared != 1 || retired != 1 || registered != 1 {
		t.Fatalf("declared=%d retired=%d registered=%d", declared, retired, registered)
	}
	if invalid := post(t, front.URL+"/library/register", `{"kind":"other"}`); invalid.StatusCode != http.StatusBadRequest {
		t.Fatalf("unknown kind = %d", invalid.StatusCode)
	}
}

func post(t *testing.T, url, body string) *http.Response {
	t.Helper()
	response, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { response.Body.Close() })
	return response
}

func decodeResponse(t *testing.T, response *http.Response, into any) {
	t.Helper()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", response.StatusCode)
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		t.Fatal(err)
	}
}

func get(t *testing.T, url string) *http.Response {
	t.Helper()
	response, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { response.Body.Close() })
	return response
}

func decode(t *testing.T, url string, into any) {
	t.Helper()
	response := get(t, url)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("%s = %d", url, response.StatusCode)
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		t.Fatal(err)
	}
}
