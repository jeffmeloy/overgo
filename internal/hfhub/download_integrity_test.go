package hfhub

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

func TestDownloadWithoutDigestRejectsCorruptedPrefix(t *testing.T) {
	weights := []byte("abcdefgh")
	for _, etag := range []string{"", `W/"weights-v1"`, `"weights-v1"`} {
		t.Run(etag, func(t *testing.T) {
			var transfers, served atomic.Int64
			file := RepoFile{Path: "weights.bin", Size: int64(len(weights))}
			client, request := downloadFixtureForFile(t, file, func(w http.ResponseWriter, r *http.Request) {
				transfers.Add(1)
				if r.Header.Get("Range") != "" || r.Header.Get("If-Range") != "" {
					t.Errorf("unverified prefix reused: %v", r.Header)
				}
				serveDownloadBytes(downloadCountingWriter{w, &served}, r, weights)
			})
			partial := seedDownloadPrefix(t, client, request, file, make([]byte, len(weights)/2))
			if err := os.WriteFile(partial+".etag", []byte(etag), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := client.Download(t.Context(), request); err != nil {
				t.Fatal(err)
			}
			got, err := os.ReadFile(filepath.Join(request.Destination, file.Path))
			if err != nil || !bytes.Equal(got, weights) || transfers.Load() != 1 || served.Load() != int64(len(weights)) {
				t.Fatalf("published=%x transfers=%d served=%d error=%v", got, transfers.Load(), served.Load(), err)
			}
			for _, path := range []string{partial, partial + ".etag"} {
				if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("published transfer retained %s: %v", path, err)
				}
			}
		})
	}
}

func TestDownloadCompleteRetainedPrefix(t *testing.T) {
	for _, test := range []struct {
		name                    string
		weights                 []byte
		digest, unknown, cancel bool
		wrongSize               bool
	}{
		{name: "verified", weights: []byte("abcdefgh"), digest: true},
		{name: "verified_unknown_size", weights: []byte("abcdefgh"), digest: true, unknown: true},
		{name: "verified_empty", digest: true},
		{name: "etag_only", weights: []byte("abcdefgh")},
		{name: "etag_only_unknown_size", weights: []byte("abcdefgh"), unknown: true},
		{name: "cancelled_verified", weights: []byte("abcdefgh"), digest: true, cancel: true},
		{name: "verified_wrong_size", weights: []byte("abcdefgh"), digest: true, wrongSize: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			file := RepoFile{Path: "weights.bin", Size: int64(len(test.weights))}
			if test.unknown {
				file.Size = 0
			}
			if test.wrongSize {
				file.Size++
			}
			if test.digest {
				digest := sha256.Sum256(test.weights)
				file.SHA256 = hex.EncodeToString(digest[:])
			}
			var transfers, served atomic.Int64
			client, request := downloadFixtureForFile(t, file, func(w http.ResponseWriter, r *http.Request) {
				transfers.Add(1)
				serveDownloadBytes(downloadCountingWriter{w, &served}, r, test.weights)
			})
			partial := seedDownloadPrefix(t, client, request, file, test.weights)
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(context.Canceled)
			if test.cancel {
				cancel(context.Canceled)
			}
			_, err := client.Download(ctx, request)
			local := filepath.Join(request.Destination, file.Path)
			if test.cancel {
				if !errors.Is(err, context.Canceled) || transfers.Load() != 0 {
					t.Fatalf("cancelled transfer: calls=%d error=%v", transfers.Load(), err)
				}
				if _, err := os.Stat(local); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("cancelled prefix published: %v", err)
				}
				return
			}
			if test.wrongSize {
				if err == nil || transfers.Load() != 0 {
					t.Fatalf("inconsistent declaration: calls=%d error=%v", transfers.Load(), err)
				}
				if _, err := os.Stat(local); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("inconsistent size published: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			got, err := os.ReadFile(local)
			if err != nil || !bytes.Equal(got, test.weights) {
				t.Fatalf("published=%q error=%v", got, err)
			}
			var wantTransfers, wantServed int64
			if !test.digest {
				wantTransfers, wantServed = 1, int64(len(test.weights))
			}
			if transfers.Load() != wantTransfers || served.Load() != wantServed {
				t.Fatalf("transfers=%d served=%d want transfers=%d served=%d", transfers.Load(), served.Load(), wantTransfers, wantServed)
			}
			for _, path := range []string{partial, partial + ".etag"} {
				if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("published transfer retained %s: %v", path, err)
				}
			}
		})
	}
}
