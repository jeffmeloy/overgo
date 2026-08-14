package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func writeManifest(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestBrowseDatasetsReadsManifest(t *testing.T) {
	dir := writeManifest(t, `{"datasets":[
		{"name":"alpha_ngrams","family":"synthetic","modality":"text","provenance":"synth"},
		{"name":"wikitext","family":"raw-text","modality":"text"}]}`)
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, DatasetsRoot: dir}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	response := serveTestRequest(handler, http.MethodGet, "/datasets", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result browseDatasetsResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Count != 2 || len(result.Datasets) != 2 {
		t.Fatalf("count=%d datasets=%d", result.Count, len(result.Datasets))
	}
	if result.Datasets[0].Name != "alpha_ngrams" || result.Datasets[0].Family != "synthetic" {
		t.Fatalf("dataset[0]=%+v", result.Datasets[0])
	}
}

func TestBrowseDatasetsUnconfigured(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{}) // no DatasetsRoot
	response := serveTestRequest(handler, http.MethodGet, "/datasets", "")
	if response.Code != http.StatusNotImplemented {
		t.Fatalf("status=%d, want 501", response.Code)
	}
}

func TestBrowseDatasetsRejectsNonGet(t *testing.T) {
	dir := writeManifest(t, `{"datasets":[]}`)
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, DatasetsRoot: dir}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/datasets", nil))
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status=%d, want 405", response.Code)
	}
}

func TestBrowseDatasetsRejectsOversizedManifest(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifest.json")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Truncate(browseDatasetManifestMaxBytes + 1); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, DatasetsRoot: dir}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	response := serveTestRequest(handler, http.MethodGet, "/datasets", "")
	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status=%d, want 413", response.Code)
	}
}
