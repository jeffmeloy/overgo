package hfhub

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/processlock"
)

func downloadFixture(t *testing.T, weights []byte, transfer http.HandlerFunc) (*Client, DownloadRequest, RepoFile) {
	t.Helper()
	digest := sha256.Sum256(weights)
	file := RepoFile{Path: "weights.bin", Size: int64(len(weights)), SHA256: hex.EncodeToString(digest[:])}
	client, request := downloadFixtureForFile(t, file, transfer)
	return client, request, file
}

func downloadFixtureForFile(t *testing.T, file RepoFile, transfer http.HandlerFunc) (*Client, DownloadRequest) {
	t.Helper()
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "acme/tiny", "sha": "commit1", "siblings": []any{map[string]any{
					"rfilename": file.Path, "lfs": map[string]any{"size": file.Size, "sha256": file.SHA256},
				}},
			})
			return
		}
		if !strings.Contains(r.URL.Path, "/resolve/commit1/") {
			t.Errorf("download used a mutable revision: %s", r.URL.Path)
		}
		transfer(w, r)
	}))
	t.Cleanup(hub.Close)
	client, err := New(hub.URL, "")
	if err != nil {
		t.Fatal(err)
	}
	return client, DownloadRequest{Kind: KindModel, Repository: "acme/tiny", Revision: "main", Destination: t.TempDir()}
}

func seedDownloadPrefix(t *testing.T, client *Client, request DownloadRequest, file RepoFile, prefix []byte) string {
	t.Helper()
	path := client.partialPath(filepath.Join(request.Destination, file.Path), request, "commit1", file)
	if err := os.WriteFile(path, prefix, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path+".etag", []byte(`"weights-v1"`), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func serveDownloadBytes(w http.ResponseWriter, r *http.Request, weights []byte) {
	w.Header().Set("ETag", `"weights-v1"`)
	http.ServeContent(w, r, "weights.bin", time.Time{}, bytes.NewReader(weights))
}

type downloadCountingWriter struct {
	http.ResponseWriter
	served *atomic.Int64
}

func (w downloadCountingWriter) Write(data []byte) (int, error) {
	written, err := w.ResponseWriter.Write(data)
	w.served.Add(int64(written))
	return written, err
}

func (w downloadCountingWriter) Flush() { w.ResponseWriter.(http.Flusher).Flush() }

func TestDownloadResumeInterruptedServedBytes(t *testing.T) {
	weights := []byte("matching immutable transfer bytes")
	prefix := weights[:len(weights)/2]
	counts := map[string]int64{}
	for _, mode := range []string{"resume", "restart"} {
		t.Run(mode, func(t *testing.T) {
			var calls atomic.Int64
			var served atomic.Int64
			client, request, file := downloadFixture(t, weights, func(w http.ResponseWriter, r *http.Request) {
				w = downloadCountingWriter{w, &served}
				if calls.Add(1) == 1 {
					w.Header().Set("ETag", `"weights-v1"`)
					w.Header().Set("Content-Length", strconv.Itoa(len(weights)))
					_, _ = w.Write(prefix)
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				}
				if mode == "resume" {
					if r.Header.Get("Range") != fmt.Sprintf("bytes=%d-", len(prefix)) || r.Header.Get("If-Range") != `"weights-v1"` {
						t.Errorf("resume headers = %v", r.Header)
					}
				} else {
					if r.Header.Get("Range") != "" {
						t.Errorf("fresh destination sent Range: %v", r.Header)
					}
				}
				serveDownloadBytes(w, r, weights)
			})
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(context.Canceled)
			request.Observe = func(progress Progress) { cancel(context.Canceled) }
			if _, err := client.Download(ctx, request); !errors.Is(err, context.Canceled) {
				t.Fatalf("interruption = %v", err)
			}
			partialPath := client.partialPath(filepath.Join(request.Destination, file.Path), request, "commit1", file)
			if retained, err := os.ReadFile(partialPath); err != nil || !bytes.Equal(retained, prefix) {
				t.Fatalf("retained %q: %v", retained, err)
			}
			if _, err := os.Stat(filepath.Join(request.Destination, file.Path)); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("interrupted file published: %v", err)
			}
			if mode == "restart" {
				request.Destination = t.TempDir()
			}
			var progress []Progress
			request.Observe = func(value Progress) { progress = append(progress, value) }
			// A new client proves restart does not depend on in-memory state.
			client, _ = New(client.endpoint, "")
			resolved, err := client.Download(t.Context(), request)
			if err != nil || resolved.Revision != "commit1" {
				t.Fatalf("resume = %+v, %v", resolved, err)
			}
			got, err := os.ReadFile(filepath.Join(request.Destination, file.Path))
			if err != nil || !bytes.Equal(got, weights) || len(progress) == 0 || progress[len(progress)-1].Received != int64(len(weights)) {
				t.Fatalf("published=%q progress=%v error=%v", got, progress, err)
			}
			if mode == "resume" {
				for _, path := range []string{partialPath, partialPath + ".etag"} {
					if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
						t.Fatalf("published transfer retained partial %s: %v", path, err)
					}
				}
			}
			if _, err := client.Download(t.Context(), request); err != nil || calls.Load() != 2 {
				t.Fatalf("verified destination refetched: calls=%d err=%v", calls.Load(), err)
			}
			counts[mode] = served.Load()
		})
	}
	if counts["resume"] != int64(len(weights)) || counts["restart"] != int64(len(weights)+len(prefix)) {
		t.Fatalf("paired served bytes = %v", counts)
	}
	t.Logf("same %d-byte input, %d-byte interruption: resume=%d bytes, fresh restart=%d bytes", len(weights), len(prefix), counts["resume"], counts["restart"])
}

func TestDownloadRangeResponses(t *testing.T) {
	weights := []byte("abcdefgh")
	for _, test := range []struct {
		name, prefix, contentRange, etag, body, length string
		status                                         int
		wantError, discard, preserve                   bool
	}{
		{name: "valid", prefix: "abcd", status: 206, contentRange: "bytes 4-7/8", body: "efgh"},
		{name: "full_restart", prefix: "abcd", status: 200, body: "abcdefgh"},
		{name: "changed_validator_full_restart", prefix: "abcd", status: 200, etag: `"weights-v2"`, body: "abcdefgh"},
		{name: "wrong_full_length", prefix: "abcd", status: 200, length: "7", body: "abcdefg", wantError: true, preserve: true},
		{name: "wrong_partial_length", prefix: "abcd", status: 206, length: "3", contentRange: "bytes 4-7/8", body: "efg", wantError: true, preserve: true},
		{name: "wrong_offset", prefix: "abcd", status: 206, contentRange: "bytes 3-6/8", body: "defg", wantError: true},
		{name: "wrong_total", prefix: "abcd", status: 206, contentRange: "bytes 4-8/9", body: "efghi", wantError: true},
		{name: "missing_range", prefix: "abcd", status: 206, body: "efgh", wantError: true},
		{name: "unsolicited_partial", status: 206, contentRange: "bytes 0-7/8", body: "abcdefgh", wantError: true},
		{name: "changed_validator_partial", prefix: "abcd", status: 206, contentRange: "bytes 4-7/8", etag: `"weights-v2"`, body: "efgh", wantError: true},
		{name: "truncated_suffix", prefix: "abcd", status: 206, contentRange: "bytes 4-7/8", body: "ef", wantError: true},
		{name: "oversized_suffix", prefix: "abcd", status: 206, contentRange: "bytes 4-7/8", body: "efghi", wantError: true, discard: true},
		{name: "corrupt_prefix", prefix: "xxxx", status: 206, contentRange: "bytes 4-7/8", body: "efgh", wantError: true, discard: true},
		{name: "corrupt_suffix", prefix: "abcd", status: 206, contentRange: "bytes 4-7/8", body: "xxxx", wantError: true, discard: true},
		{name: "corrupt_complete_416", prefix: "xxxxxxxx", status: 416, contentRange: "bytes */8", wantError: true, discard: true},
		{name: "incomplete_416", prefix: "abcd", status: 416, contentRange: "bytes */8", wantError: true},
		{name: "malformed_416", prefix: "xxxxxxxx", status: 416, contentRange: "8", wantError: true, preserve: true},
		{name: "wrong_total_416", prefix: "xxxxxxxx", status: 416, contentRange: "bytes */9", wantError: true, preserve: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			var transfers atomic.Int64
			client, request, file := downloadFixture(t, weights, func(w http.ResponseWriter, r *http.Request) {
				transfers.Add(1)
				w.Header().Set("Content-Range", test.contentRange)
				w.Header().Set("ETag", test.etag)
				if test.length != "" {
					w.Header().Set("Content-Length", test.length)
				}
				w.WriteHeader(test.status)
				w.(http.Flusher).Flush() // exercise chunked, not just Content-Length validation
				_, _ = w.Write([]byte(test.body))
			})
			partial := seedDownloadPrefix(t, client, request, file, []byte(test.prefix))
			_, err := client.Download(t.Context(), request)
			if transfers.Load() != 1 {
				t.Fatalf("response case did not execute its transfer: calls=%d", transfers.Load())
			}
			if (err != nil) != test.wantError {
				t.Fatalf("error=%v wantError=%t", err, test.wantError)
			}
			got, readErr := os.ReadFile(filepath.Join(request.Destination, file.Path))
			if test.wantError {
				if !errors.Is(readErr, os.ErrNotExist) {
					t.Fatalf("invalid transfer published %q: %v", got, readErr)
				}
				if test.discard {
					if _, err := os.Stat(partial); !errors.Is(err, os.ErrNotExist) {
						t.Fatalf("corrupt prefix retained: %v", err)
					}
				}
				if test.preserve {
					got, err := os.ReadFile(partial)
					if err != nil || string(got) != test.prefix {
						t.Fatalf("invalid headers lost prefix=%q: %v", got, err)
					}
				}
			} else if readErr != nil || !bytes.Equal(got, weights) {
				t.Fatalf("published %q: %v", got, readErr)
			}
		})
	}
}

func TestDownloadWithoutDigestRestartsValidPrefix(t *testing.T) {
	weights := []byte("abcdefgh")
	for _, etag := range []string{"", `W/"weights-v1"`, `"weights-v1"`} {
		t.Run(etag, func(t *testing.T) {
			client, request, file := downloadFixture(t, weights, func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Range") != "" || r.Header.Get("If-Range") != "" {
					t.Errorf("unverified prefix reused: %v", r.Header)
				}
				serveDownloadBytes(w, r, weights)
			})
			file.SHA256 = ""
			partial := seedDownloadPrefix(t, client, request, file, weights[:len(weights)/2])
			if err := os.WriteFile(partial+".etag", []byte(etag), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := client.downloadFile(t.Context(), request, "commit1", file); err != nil {
				t.Fatal(err)
			}
			got, err := os.ReadFile(filepath.Join(request.Destination, file.Path))
			if err != nil || !bytes.Equal(got, weights) {
				t.Fatalf("bytes=%q: %v", got, err)
			}
		})
	}
}

func TestDownloadIdentitySeparatesSources(t *testing.T) {
	weights := []byte("abcdefgh")
	for _, field := range []string{"endpoint", "namespace", "repository", "revision", "path", "digest", "size"} {
		t.Run(field, func(t *testing.T) {
			client, request, file := downloadFixture(t, weights, func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Range") != "" {
					t.Errorf("foreign prefix reused: %v", r.Header)
				}
				serveDownloadBytes(w, r, weights)
			})
			priorClient, priorRequest, priorFile, revision := *client, request, file, "commit1"
			switch field {
			case "endpoint":
				priorClient.endpoint += "/mirror"
			case "namespace":
				priorRequest.Kind = KindDataset
			case "repository":
				priorRequest.Repository = "another/repo"
			case "revision":
				revision = "prior-commit"
			case "path":
				priorFile.Path = "other.bin"
			case "digest":
				priorFile.SHA256 = strings.Repeat("0", len(file.SHA256))
			case "size":
				priorFile.Size++
			}
			foreign := priorClient.partialPath(filepath.Join(request.Destination, file.Path), priorRequest, revision, priorFile)
			if err := os.WriteFile(foreign, weights[:len(weights)/2], 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := client.Download(t.Context(), request); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(foreign); err != nil {
				t.Fatalf("foreign progress lost: %v", err)
			}
		})
	}
}

func TestDownloadSameTargetSerializationAndWaitCancellation(t *testing.T) {
	weights := []byte("abcdefgh")
	var transfers atomic.Int64
	client, request, file := downloadFixture(t, weights, func(w http.ResponseWriter, r *http.Request) {
		transfers.Add(1)
		serveDownloadBytes(w, r, weights)
	})
	local := filepath.Join(request.Destination, file.Path)
	lock, err := processlock.Acquire(local+".download.lock", 0o600)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := client.Download(ctx, request); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled request: %v", err)
	}
	// Exercise the resolved transfer while another process owns the target.
	// Cancellation must retain the proven contention and start no content GET.
	waiting, stop := context.WithCancelCause(t.Context())
	blocked := make(chan error, 1)
	go func() { blocked <- client.downloadFile(waiting, request, "commit1", file) }()
	stop(context.Canceled)
	err = <-blocked
	if !errors.Is(err, context.Canceled) || !errors.Is(err, processlock.ErrBusy) || transfers.Load() != 0 {
		t.Fatalf("contended cancellation: transfers=%d err=%v", transfers.Load(), err)
	}
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	results := make(chan error, 2)
	for range cap(results) {
		go func() { _, err := client.Download(t.Context(), request); results <- err }()
	}
	for range cap(results) {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
	if transfers.Load() != 1 {
		t.Fatalf("duplicate content streams: %d", transfers.Load())
	}
}

func TestDownloadRequiresResolvedCommit(t *testing.T) {
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"acme/tiny","siblings":[]}`))
	}))
	defer hub.Close()
	client, _ := New(hub.URL, "")
	if _, err := client.Resolve(t.Context(), KindModel, "acme/tiny", "main"); err == nil {
		t.Fatal("mutable revision accepted without a resolved commit")
	}
}

func TestDownloadRecoveryAfterShortOrCorruptBody(t *testing.T) {
	weights := []byte("abcdefgh")
	for _, corrupt := range []bool{false, true} {
		t.Run(fmt.Sprintf("corrupt=%t", corrupt), func(t *testing.T) {
			var calls atomic.Int64
			client, request, file := downloadFixture(t, weights, func(w http.ResponseWriter, r *http.Request) {
				if calls.Add(1) == 1 {
					w.Header().Set("ETag", `"weights-v1"`)
					w.Header().Set("Content-Length", strconv.Itoa(len(weights)))
					if corrupt {
						_, _ = w.Write([]byte("xxxxxxxx"))
					} else {
						_, _ = w.Write(weights[:len(weights)/2])
					}
					return
				}
				wantRange := "bytes=4-"
				if corrupt {
					wantRange = ""
				}
				if r.Header.Get("Range") != wantRange {
					t.Errorf("retry Range=%q want=%q", r.Header.Get("Range"), wantRange)
				}
				serveDownloadBytes(w, r, weights)
			})
			if _, err := client.Download(t.Context(), request); err == nil {
				t.Fatal("invalid initial transfer succeeded")
			}
			if _, err := client.Download(t.Context(), request); err != nil {
				t.Fatal(err)
			}
			got, err := os.ReadFile(filepath.Join(request.Destination, file.Path))
			if err != nil || !bytes.Equal(got, weights) {
				t.Fatalf("retry=%q: %v", got, err)
			}
		})
	}
}
