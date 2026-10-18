package hfhub

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const modelScopeTestRevision = "797fa8e678b4f0a766271bf5103e6cd7a2faa999"

func TestModelScopeDownload(t *testing.T) {
	for _, mode := range []string{"complete", "resume", "interrupted", "corrupt"} {
		t.Run(mode, func(t *testing.T) {
			payload := []byte("verified ModelScope model payload")
			file := RepoFile{Path: "transformer/weights.safetensors", Size: int64(len(payload)), SHA256: fmt.Sprintf("%x", sha256.Sum256(payload))}
			transfers := 0
			hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Query().Get("Revision") != modelScopeTestRevision {
					t.Errorf("unpinned request: %s", r.URL)
				}
				if strings.HasSuffix(r.URL.Path, "/repo/files") {
					_ = json.NewEncoder(w).Encode(map[string]any{"Code": 200, "Success": true, "Data": map[string]any{"Files": []any{
						map[string]any{"Type": "tree", "Path": "transformer"},
						map[string]any{"Type": "blob", "Path": file.Path, "Size": file.Size, "Sha256": file.SHA256},
					}}})
					return
				}
				transfers++
				if r.URL.Path != "/api/v1/models/openmoss/tiny/repo" || r.URL.Query().Get("FilePath") != file.Path {
					t.Errorf("wrong file request: %s", r.URL)
				}
				if mode == "corrupt" {
					_, _ = w.Write(bytes.Repeat([]byte("x"), len(payload)))
					return
				}
				if mode == "interrupted" && transfers == 1 {
					w.Header().Set("Content-Length", fmt.Sprint(len(payload)))
					_, _ = w.Write(payload[:8])
					return
				}
				if (mode == "resume" || mode == "interrupted") && r.Header.Get("Range") != "bytes=8-" {
					t.Errorf("resume range=%q", r.Header.Get("Range"))
				}
				serveDownloadBytes(w, r, payload)
			}))
			defer hub.Close()
			client, err := NewModelScope(hub.URL, "")
			if err != nil {
				t.Fatal(err)
			}
			request := DownloadRequest{Kind: KindModel, Repository: "openmoss/tiny", Revision: modelScopeTestRevision, Destination: t.TempDir()}
			if mode == "resume" {
				partial := client.partialPath(bookkeepingFor(t, request, file), request, modelScopeTestRevision, file)
				if err := os.WriteFile(partial, payload[:8], 0600); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "interrupted" {
				if _, err := client.Download(t.Context(), request); err == nil {
					t.Fatal("interrupted download succeeded")
				}
				if _, err := os.Stat(filepath.Join(request.Destination, file.Path)); !os.IsNotExist(err) {
					t.Fatalf("partial payload published: %v", err)
				}
			}
			resolved, err := client.Download(t.Context(), request)
			if mode == "corrupt" {
				if err == nil {
					t.Fatal("corrupt payload accepted")
				}
				if _, err := os.Stat(filepath.Join(request.Destination, file.Path)); !os.IsNotExist(err) {
					t.Fatalf("corrupt payload published: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if resolved.Revision != modelScopeTestRevision || len(resolved.Files) != 1 {
				t.Fatalf("inventory=%+v", resolved)
			}
			got, err := os.ReadFile(filepath.Join(request.Destination, file.Path))
			if err != nil || !bytes.Equal(got, payload) {
				t.Fatalf("payload=%q err=%v", got, err)
			}
			if _, err := client.Download(t.Context(), request); err != nil {
				t.Fatal(err)
			}
			wantTransfers := 1
			if mode == "interrupted" {
				wantTransfers++
			}
			if transfers != wantTransfers {
				t.Fatalf("verified payload redownloaded %d times", transfers)
			}
		})
	}
}

func TestModelScopeInventoryRefusals(t *testing.T) {
	file := map[string]any{"Type": "blob", "Path": "weights.bin", "Size": 1, "Sha256": fmt.Sprintf("%x", sha256.Sum256([]byte("x")))}
	for _, failure := range []string{"mutable", "dataset", "envelope", "duplicate", "escape", "digest", "negative-size", "unknown-type", "empty"} {
		t.Run(failure, func(t *testing.T) {
			entry := map[string]any{}
			maps.Copy(entry, file)
			switch failure {
			case "escape":
				entry["Path"] = "../weights.bin"
			case "digest":
				entry["Sha256"] = ""
			case "negative-size":
				entry["Size"] = -1
			case "unknown-type":
				entry["Type"] = "symlink"
			}
			entries := []any{entry}
			if failure == "duplicate" {
				entries = append(entries, entry)
			}
			if failure == "empty" {
				entries = nil
			}
			hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{"Code": 200, "Success": failure != "envelope", "Data": map[string]any{"Files": entries}})
			}))
			defer hub.Close()
			client, err := NewModelScope(hub.URL, "")
			if err != nil {
				t.Fatal(err)
			}
			revision := modelScopeTestRevision
			kind := KindModel
			if failure == "mutable" {
				revision = "master"
			}
			if failure == "dataset" {
				kind = KindDataset
			}
			if _, err := client.Resolve(t.Context(), kind, "openmoss/tiny", revision); err == nil {
				t.Fatal("invalid inventory accepted")
			}
		})
	}
}
