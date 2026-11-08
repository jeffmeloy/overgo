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

	"overgo/internal/hfrepo"
	"overgo/internal/testcheck"
)

// TestADownloadTakesTheFilesItNames holds a download to what a decision named: on 2026-11-08 decision
// self-forcing-download named one 5.68 GB checkpoint of a repository of 45 GB. A download naming one file of two fetches
// that file alone and receipts it alone, and one naming a file the repository does not hold is refused.
func TestADownloadTakesTheFilesItNames(t *testing.T) {
	t.Parallel()
	const revision = "797fa8e678b4f0a766271bf5103e6cd7a2faa999"
	payloads := map[string][]byte{"checkpoints/student.pt": []byte("the student"), "checkpoints/teacher.pt": []byte("the teacher")}
	fetched := map[string]int{}
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/repo/files") {
			var listed []any
			for path, payload := range payloads {
				listed = append(listed, map[string]any{"Type": "blob", "Path": path, "Size": len(payload), "Sha256": fmt.Sprintf("%x", sha256.Sum256(payload))})
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"Code": 200, "Success": true, "Data": map[string]any{"Files": listed}})
			return
		}
		for path, payload := range payloads {
			if strings.Contains(r.URL.String(), filepath.Base(path)) {
				fetched[path]++
				_, _ = w.Write(payload)
			}
		}
	}))
	defer hub.Close()
	root := t.TempDir()
	destination, receiptPath := filepath.Join(root, "model"), filepath.Join(root, "source.json")
	download := func(files string) error {
		return runArguments(t.Context(), []string{"download", "-provider", "modelscope", "-endpoint", hub.URL, "-revision", revision,
			"-dest", destination, "-receipt", receiptPath, "-files", files, "openmoss/pair"})
	}
	testcheck.NoError(t, download("checkpoints/student.pt"))
	var receipt hfrepo.SourceReceipt
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(receiptPath))(t), &receipt))
	if len(receipt.Files) != 1 || receipt.Files[0].Path != "checkpoints/student.pt" || fetched["checkpoints/teacher.pt"] != 0 {
		t.Fatalf("naming the student fetched %v and receipted %+v, want the student alone", fetched, receipt.Files)
	}
	if _, err := os.Stat(filepath.Join(destination, "checkpoints", "teacher.pt")); err == nil {
		t.Fatal("a file no one named was written")
	}
	if err := download("checkpoints/absent.pt"); err == nil || !strings.Contains(err.Error(), "-files") {
		t.Fatalf("naming a file the repository does not hold ended as %v, want it refused", err)
	}
}
