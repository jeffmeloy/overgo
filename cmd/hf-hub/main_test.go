package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDownloadCLI(t *testing.T) {
	weights := []byte("verified CLI bytes")
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			_ = json.NewEncoder(w).Encode(map[string]any{"id": "acme/tiny", "sha": "commit1", "siblings": []any{
				map[string]any{"rfilename": "weights.bin", "lfs": map[string]any{"size": len(weights), "sha256": fmt.Sprintf("%x", sha256.Sum256(weights))}},
			}})
			return
		}
		if r.URL.Path != "/acme/tiny/resolve/commit1/weights.bin" {
			t.Errorf("mutable path %s", r.URL.Path)
		}
		_, _ = w.Write(weights)
	}))
	defer hub.Close()
	destination := t.TempDir()
	if err := runArguments(t.Context(), []string{"download", "-endpoint", hub.URL, "-revision", "main", "-dest", destination, "acme/tiny"}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(destination, "weights.bin"))
	if err != nil || string(got) != string(weights) {
		t.Fatalf("download=%q: %v", got, err)
	}
}

func TestDownloadCLICallerBudget(t *testing.T) {
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			_, _ = w.Write([]byte(`{"id":"acme/tiny","sha":"commit1","siblings":[{"rfilename":"weights.bin","size":8}]}`))
			return
		}
		w.Header().Set("Content-Length", "8")
		w.Header().Set("ETag", `"weights-v1"`)
		_, _ = w.Write([]byte("abcd"))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer hub.Close()
	destination := t.TempDir()
	err := runArguments(t.Context(), []string{"download", "-endpoint", hub.URL, "-timeout", "1s", "-dest", destination, "acme/tiny"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("caller deadline = %v", err)
	}
	partials, err := filepath.Glob(filepath.Join(destination, "weights.bin.partial-*"))
	if err != nil {
		t.Fatal(err)
	}
	retained := false
	for _, path := range partials {
		if strings.HasSuffix(path, ".etag") {
			continue
		}
		got, err := os.ReadFile(path)
		if err != nil || string(got) != "abcd" {
			t.Fatalf("retained=%q: %v", got, err)
		}
		retained = true
	}
	if !retained {
		t.Fatal("deadline lost received bytes")
	}
	if _, err := os.Stat(filepath.Join(destination, "weights.bin")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("partial published: %v", err)
	}
}

func TestDownloadCLIRefusalsAndParentCancellation(t *testing.T) {
	for _, args := range [][]string{nil, {"download"}, {"download", "-timeout", "-1s"}, {"download", "-timeout", "invalid"}} {
		if err := runArguments(t.Context(), args); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if err := runArguments(ctx, []string{"download", "-endpoint", "http://127.0.0.1", "-dest", t.TempDir(), "acme/tiny"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("parent cancellation = %v", err)
	}
}
