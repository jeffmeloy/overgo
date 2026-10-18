package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/hfrepo"
)

func TestDownloadReceipt(t *testing.T) {
	payload := []byte("pinned source payload")
	const revision = "797fa8e678b4f0a766271bf5103e6cd7a2faa999"
	transfers := 0
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("Revision") != revision {
			t.Errorf("request lost immutable revision: %s", r.URL)
		}
		if strings.HasSuffix(r.URL.Path, "/repo/files") {
			_ = json.NewEncoder(w).Encode(map[string]any{"Code": 200, "Success": true, "Data": map[string]any{"Files": []any{map[string]any{"Type": "blob", "Path": "weights.bin", "Size": len(payload), "Sha256": fmt.Sprintf("%x", sha256.Sum256(payload))}}}})
			return
		}
		transfers++
		_, _ = w.Write(payload)
	}))
	defer hub.Close()
	root := t.TempDir()
	destination := filepath.Join(root, "model")
	output := filepath.Join(root, "source.json")
	args := []string{"download", "-provider", "modelscope", "-endpoint", hub.URL, "-revision", revision, "-dest", destination, "-receipt", output, "openmoss/tiny"}
	if err := runArguments(t.Context(), args); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	var receipt hfrepo.SourceReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Source != hub.URL+"/openmoss/tiny.git" || receipt.Revision != revision || len(receipt.Files) != 1 {
		t.Fatalf("wrong source receipt: %+v", receipt)
	}
	if _, err := receipt.Verify(t.Context(), destination); err != nil {
		t.Fatal(err)
	}
	if err := runArguments(t.Context(), args); err != nil {
		t.Fatal(err)
	}
	repeat, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, repeat) || transfers != 1 {
		t.Fatalf("repeat changed receipt or redownloaded verified bytes: transfers=%d", transfers)
	}
	args[len(args)-2] = filepath.Join(destination, "weights.bin")
	if err := runArguments(t.Context(), args); err == nil {
		t.Fatal("receipt could overwrite source payload")
	}
	unchanged, err := os.ReadFile(filepath.Join(destination, "weights.bin"))
	if err != nil || !bytes.Equal(unchanged, payload) {
		t.Fatal("source payload changed")
	}
	args[0] = "resolve"
	if err := runArguments(t.Context(), args); err == nil || !strings.Contains(err.Error(), "-receipt requires download") {
		t.Fatalf("receipt option silently ignored: %v", err)
	}
}
