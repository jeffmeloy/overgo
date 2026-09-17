package hfhub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/processlock"
)

// Progress reports one download's advancing byte count. Total is zero when
// the hub declared no size.
type Progress struct {
	Path     string
	Received int64
	Total    int64
}

// DownloadRequest names one repository revision and the destination
// directory its files land in.
type DownloadRequest struct {
	Kind        RepoKind
	Repository  string
	Revision    string
	Destination string
	// Files limits the download to exact paths; empty downloads everything.
	Files []string
	// Observe receives progress after each chunk; nil observes nothing.
	Observe func(Progress)
}

// Download fetches a repository revision into the destination directory.
// Every file downloads to an identity-bound partial sibling and is renamed into
// place only after its declared size and, for LFS files, its sha256 match;
// a partial or tampered download never lands under its final name. Files
// already present with matching size and digest are not fetched again.
func (c *Client) Download(ctx context.Context, request DownloadRequest) (RepoRevision, error) {
	if strings.TrimSpace(request.Destination) == "" {
		return RepoRevision{}, errors.New("hfhub: download requires a destination directory")
	}
	resolved, err := c.Resolve(ctx, request.Kind, request.Repository, request.Revision)
	if err != nil {
		return RepoRevision{}, err
	}
	wanted := map[string]bool{}
	for _, path := range request.Files {
		wanted[path] = true
	}
	if err := os.MkdirAll(request.Destination, 0o755); err != nil {
		return RepoRevision{}, err
	}
	for _, file := range resolved.Files {
		if len(wanted) > 0 && !wanted[file.Path] {
			continue
		}
		if err := c.downloadFile(ctx, request, resolved.Revision, file); err != nil {
			return RepoRevision{}, fmt.Errorf("hfhub: %s: %w", file.Path, err)
		}
	}
	return resolved, nil
}

func (c *Client) downloadFile(ctx context.Context, request DownloadRequest, revision string, file RepoFile) error {
	local, err := secureJoin(request.Destination, file.Path)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(local), 0o755); err != nil {
		return err
	}
	// Lock the destination, not the source identity: different revisions must
	// not race publication. The lock file survives close to keep one OS owner.
	lock, err := processlock.AcquireContext(ctx, local+".download.lock", 0o600)
	if err != nil {
		return err
	}
	defer lock.Close()
	if alreadyComplete(ctx, local, file) {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	partial, err := c.openPartial(ctx, local, request, revision, file)
	if err != nil {
		return err
	}
	defer partial.file.Close()
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodGet,
		c.fileURL(request.Kind, request.Repository, revision, file.Path), nil)
	if err != nil {
		return err
	}
	httpRequest.Header.Set("Accept-Encoding", "identity")
	if partial.received > 0 {
		httpRequest.Header.Set("Range", fmt.Sprintf("bytes=%d-", partial.received))
		if partial.etag != "" {
			httpRequest.Header.Set("If-Range", partial.etag)
		}
	}
	c.authorize(httpRequest)
	response, err := c.transfer.Do(httpRequest)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if err := partial.acceptResponse(response, &file); err != nil {
		return err
	}
	if response.StatusCode != http.StatusRequestedRangeNotSatisfiable {
		received, copyErr := observedCopy(ctx, io.MultiWriter(partial.file, partial.digest), response.Body, file, partial.received, request.Observe)
		partial.received = received
		if copyErr != nil {
			if errors.Is(copyErr, errDownloadOversize) {
				return errors.Join(copyErr, partial.discard())
			}
			return copyErr
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if file.Size > 0 && partial.received != file.Size {
		return fmt.Errorf("received %d bytes, hub declared %d: %w", partial.received, file.Size, io.ErrUnexpectedEOF)
	}
	if file.SHA256 != "" && !strings.EqualFold(hex.EncodeToString(partial.digest.Sum(nil)), file.SHA256) {
		return errors.Join(errors.New("sha256 differs from the hub's declared digest"), partial.discard())
	}
	if err := partial.file.Sync(); err != nil {
		return err
	}
	if err := partial.file.Close(); err != nil {
		return err
	}
	if err := os.Rename(partial.file.Name(), local); err != nil {
		return err
	}
	return removePartialMetadata(partial.file.Name())
}

var errDownloadOversize = errors.New("download exceeds the declared size")

func observedCopy(ctx context.Context, destination io.Writer, source io.Reader, file RepoFile, received int64, observe func(Progress)) (int64, error) {
	buffer := make([]byte, 1<<20)
	for {
		if err := ctx.Err(); err != nil {
			return received, err
		}
		read, err := source.Read(buffer)
		if read > 0 {
			if file.Size > 0 && int64(read) > file.Size-received {
				return received, errDownloadOversize
			}
			written, writeErr := destination.Write(buffer[:read])
			received += int64(written)
			if writeErr != nil {
				return received, writeErr
			}
			if written != read {
				return received, io.ErrShortWrite
			}
			if observe != nil {
				observe(Progress{Path: file.Path, Received: received, Total: file.Size})
			}
		}
		if errors.Is(err, io.EOF) {
			return received, nil
		}
		if err != nil {
			return received, err
		}
	}
}

// alreadyComplete reports whether the local file exactly matches the hub's
// declaration; without a declared digest, matching size is the best claim
// available and re-download is skipped on it.
func alreadyComplete(ctx context.Context, local string, file RepoFile) bool {
	info, err := os.Stat(local)
	if err != nil || info.IsDir() {
		return false
	}
	if file.Size > 0 && info.Size() != file.Size {
		return false
	}
	if file.SHA256 == "" {
		return file.Size > 0
	}
	handle, err := os.Open(local)
	if err != nil {
		return false
	}
	defer handle.Close()
	digest := sha256.New()
	if _, err := observedCopy(ctx, digest, handle, file, 0, nil); err != nil {
		return false
	}
	return strings.EqualFold(hex.EncodeToString(digest.Sum(nil)), file.SHA256)
}

// secureJoin refuses any repository path that would escape the destination.
func secureJoin(destination, path string) (string, error) {
	if path == "" || strings.Contains(path, "\x00") {
		return "", fmt.Errorf("invalid repository path %q", path)
	}
	joined := filepath.Join(destination, filepath.FromSlash(path))
	root, err := filepath.Abs(destination)
	if err != nil {
		return "", err
	}
	absolute, err := filepath.Abs(joined)
	if err != nil {
		return "", err
	}
	if absolute != root && !strings.HasPrefix(absolute, root+string(filepath.Separator)) {
		return "", fmt.Errorf("repository path %q escapes the destination", path)
	}
	return joined, nil
}
