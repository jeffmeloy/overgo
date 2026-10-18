package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestModelScopeCLI(t *testing.T) {
	t.Setenv("OVERGO_HF_TOKEN", "unrelated-hub-token")
	t.Setenv("OVERGO_MODELSCOPE_TOKEN", "")
	payload := []byte("modelscope command payload")
	revision := strings.Repeat("a", 40)
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Error("Hugging Face credential leaked to ModelScope")
		}
		if r.URL.Query().Get("Revision") != revision {
			t.Errorf("unpinned request: %s", r.URL)
		}
		if strings.HasSuffix(r.URL.Path, "/files") {
			_ = json.NewEncoder(w).Encode(map[string]any{"Code": 200, "Success": true, "Data": map[string]any{"Files": []any{map[string]any{"Type": "blob", "Path": "weights.bin", "Size": len(payload), "Sha256": fmt.Sprintf("%x", sha256.Sum256(payload))}}}})
			return
		}
		if r.URL.Query().Get("FilePath") != "weights.bin" {
			t.Errorf("wrong file: %s", r.URL)
		}
		_, _ = w.Write(payload)
	}))
	defer hub.Close()
	destination := t.TempDir()
	if err := runArguments(t.Context(), []string{"download", "-provider", "modelscope", "-endpoint", hub.URL, "-revision", revision, "-dest", destination, "openmoss/tiny"}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(destination, "weights.bin"))
	if err != nil || string(got) != string(payload) {
		t.Fatalf("payload %q: %v", got, err)
	}
	for _, args := range [][]string{
		{"resolve", "-provider", "unknown", "openmoss/tiny"},
		{"resolve", "-provider", "modelscope", "-revision", "master", "openmoss/tiny"},
		{"search", "-provider", "modelscope", "tiny"},
	} {
		if err := runArguments(t.Context(), args); err == nil {
			t.Fatalf("accepted unsupported command %v", args)
		}
	}
}
