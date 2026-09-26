package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserDatasetLibrary drives the Library tab over a hub serving
// datasets: a finished download offers register and holds validate until
// the entry is registered, registering names the dataset, validating
// previews its first row, and a dataset added from a search downloads,
// registers and validates in one chain.
func TestWebUIBrowserDatasetLibrary(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the dataset library leg uses Chromium through cmd/webui-lane")
	}
	notes := []byte("the sea is wide\nthe sky is high\n")
	digest := sha256.Sum256(notes)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/datasets", func(response http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(response).Encode([]map[string]any{{"id": "acme/added", "downloads": 3}})
	})
	mux.HandleFunc("GET /api/datasets/acme/{name}", func(response http.ResponseWriter, request *http.Request) {
		_ = json.NewEncoder(response).Encode(map[string]any{
			"id": "acme/" + request.PathValue("name"), "sha": "rev0",
			"siblings": []map[string]any{{"rfilename": "notes.txt", "lfs": map[string]any{
				"sha256": hex.EncodeToString(digest[:]), "size": len(notes),
			}}},
		})
	})
	mux.HandleFunc("GET /datasets/acme/{name}/resolve/rev0/notes.txt", func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write(notes)
	})
	hub := httptest.NewServer(mux)
	defer hub.Close()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	root := t.TempDir()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}), func(config *Config) {
		config.HubEndpoint, config.HubDownloadRoot = hub.URL, root
	})
	// A download another client started: its row offers the lifecycle, unchained.
	if started := serveTestRequest(handler, http.MethodPost, "/hub/downloads",
		`{"kind":"datasets","repository":"acme/fetched","revision":"","directory":""}`); started.Code != http.StatusAccepted {
		t.Fatalf("download status=%d body=%s", started.Code, started.Body.String())
	}
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => { location.hash = 'library'; return true; })()`)
	settle(`!!document.querySelector('#panel-library.active')`)
	check(`(() => {
  window.libraryPanel = () => document.querySelector('#panel-library');
  window.libraryJob = (repository) => [...libraryPanel().querySelectorAll('tr')].find(row => row.firstChild && row.firstChild.textContent === repository);
  window.libraryButton = (repository, label) => [...(libraryJob(repository) || document.createElement('tr')).querySelectorAll('button')].find(button => button.textContent === label);
  return true;
})()`)

	// A finished download offers register and holds validate until registered.
	settle(`!!libraryJob('acme/fetched') && libraryJob('acme/fetched').textContent.includes('done') && !!libraryButton('acme/fetched', 'register') && libraryButton('acme/fetched', 'validate').disabled`)
	check(`(() => { libraryButton('acme/fetched', 'register').click(); return true; })()`)
	settle(`!!libraryButton('acme/fetched', 'registered') && libraryButton('acme/fetched', 'registered').disabled && !libraryButton('acme/fetched', 'validate').disabled && libraryJob('acme/fetched').textContent.includes('registered dataset:')`)
	check(`(() => { libraryButton('acme/fetched', 'validate').click(); return true; })()`)
	settle(`libraryJob('acme/fetched').textContent.includes('validated: 1 example previewed')`)

	// A dataset added from a search downloads, registers and validates in one chain.
	check(`(() => {
  const kind = libraryPanel().querySelector('select'); kind.value = 'datasets'; kind.dispatchEvent(new Event('change', { bubbles: true }));
  [...libraryPanel().querySelectorAll('button')].find(button => button.textContent === 'Search').click();
  return true;
})()`)
	settle(`!!libraryButton('acme/added', 'Add')`)
	check(`(() => { libraryButton('acme/added', 'Add').click(); return true; })()`)
	settle(`[...libraryPanel().querySelectorAll('tr')].some(row => row.firstChild.textContent === 'acme/added' && row.textContent.includes('validated: 1 example previewed'))`)
	var listed struct {
		Datasets []struct {
			Name string `json:"name"`
		} `json:"datasets"`
	}
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/datasets", "").Body.Bytes(), &listed); err != nil {
		t.Fatal(err)
	}
	names := []string{}
	for _, dataset := range listed.Datasets {
		names = append(names, dataset.Name)
	}
	if joined := strings.Join(names, ","); !strings.Contains(joined, "fetched") || !strings.Contains(joined, "added") {
		t.Fatalf("registered datasets = %s", joined)
	}
	webuilane.Leg(t, "dataset library leg", "a finished download offered register and held validate until registered, validation previewed its first row, and a dataset added from a search downloaded, registered and validated in one chain")
}
