package hfhub

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"overgo/internal/atomicfile"
)

// The name binds all source facts, excluding credentials. Interrupted prefixes
// survive process exit; every reuse rehashes the actual bytes under the target
// lock. Colibri f028d26's c/download_fp8.py (Apache-2.0) informed the behavior;
// this is an independent implementation in the existing Go download owner.
func (c *Client) partialPath(local string, request DownloadRequest, revision string, file RepoFile) string {
	identity, _ := json.Marshal([]string{c.endpoint, string(request.Kind), request.Repository, revision, file.Path, strings.ToLower(file.SHA256), strconv.FormatInt(file.Size, 10)})
	return fmt.Sprintf("%s.partial-%x", local, sha256.Sum256(identity))
}

type downloadPartial struct {
	file     *os.File
	digest   hash.Hash
	received int64
	etag     string
}

func (c *Client) openPartial(ctx context.Context, local string, request DownloadRequest, revision string, file RepoFile) (*downloadPartial, error) {
	path := c.partialPath(local, request, revision, file)
	handle, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	partial := &downloadPartial{file: handle, digest: sha256.New()}
	raw, err := os.ReadFile(path + ".etag")
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		_ = handle.Close()
		return nil, err
	}
	partial.etag = strongETag(string(raw))
	// An ETag validates the remote representation, not the retained local
	// bytes. Reuse requires an end-to-end digest to check before publication.
	if file.SHA256 == "" {
		err = partial.restart()
	} else {
		partial.received, err = observedCopy(ctx, partial.digest, handle, file, 0, nil)
		if errors.Is(err, errDownloadOversize) {
			err = partial.restart()
		}
	}
	if err != nil {
		_ = handle.Close()
		return nil, err
	}
	return partial, nil
}

func (partial *downloadPartial) restart() error {
	if err := partial.file.Truncate(0); err != nil {
		return err
	}
	if _, err := partial.file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	partial.received, partial.etag = 0, ""
	partial.digest.Reset()
	return removePartialMetadata(partial.file.Name())
}

func (partial *downloadPartial) discard() error {
	return errors.Join(partial.file.Close(), os.Remove(partial.file.Name()), removePartialMetadata(partial.file.Name()))
}

func removePartialMetadata(path string) error {
	err := os.Remove(path + ".etag")
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

func strongETag(raw string) string {
	tag, quoted := strings.CutPrefix(raw, `"`)
	if !quoted {
		return ""
	}
	tag, quoted = strings.CutSuffix(tag, `"`)
	if !quoted || strings.ContainsAny(tag, "\"\r\n") {
		return ""
	}
	return raw
}

func (partial *downloadPartial) acceptResponse(response *http.Response, file *RepoFile) error {
	if encoding := response.Header.Get("Content-Encoding"); encoding != "" && encoding != "identity" {
		return errors.New("hfhub: encoded transfer cannot be resumed by byte offset")
	}
	switch response.StatusCode {
	case http.StatusOK:
		// A server may ignore Range or reject If-Range with a full response.
		// Never append that response to the retained prefix.
		if file.Size > 0 && response.ContentLength >= 0 && response.ContentLength != file.Size {
			return errors.New("hfhub: response length differs from the declared size")
		}
		if err := partial.restart(); err != nil {
			return err
		}
	case http.StatusPartialContent:
		start, end, total, err := parseContentRange(response.Header.Get("Content-Range"))
		if err != nil || partial.received == 0 || start != partial.received || end != total-1 ||
			(file.Size > 0 && total != file.Size) || (response.ContentLength >= 0 && response.ContentLength != total-start) {
			return errors.New("hfhub: invalid resumed Content-Range or length")
		}
		if etag := response.Header.Get("ETag"); partial.etag != "" && etag != "" && etag != partial.etag {
			return errors.New("hfhub: resumed ETag differs from If-Range")
		}
		file.Size = total
	case http.StatusRequestedRangeNotSatisfiable:
		total, err := strconv.ParseInt(strings.TrimPrefix(response.Header.Get("Content-Range"), "bytes */"), 10, 64)
		if !strings.HasPrefix(response.Header.Get("Content-Range"), "bytes */") || err != nil || total <= 0 || total != partial.received || file.Size != total || file.SHA256 == "" {
			return errors.New("hfhub: unverified or incomplete prefix at range refusal")
		}
		// The caller still checks the rehashed digest before publishing.
		return nil
	default:
		return statusError(response)
	}
	if response.StatusCode == http.StatusOK {
		partial.etag = strongETag(response.Header.Get("ETag"))
	}
	if partial.etag != "" {
		return atomicfile.Write(partial.file.Name()+".etag", []byte(partial.etag), 0o600)
	}
	return nil
}

// Content-Range uses decimal inclusive byte offsets (RFC 9110 section 14.4).
func parseContentRange(raw string) (start, end, total int64, err error) {
	span, totalText, found := strings.Cut(strings.TrimPrefix(raw, "bytes "), "/")
	startText, endText, bounded := strings.Cut(span, "-")
	if !strings.HasPrefix(raw, "bytes ") || !found || !bounded {
		return 0, 0, 0, errors.New("invalid Content-Range")
	}
	start, startErr := strconv.ParseInt(startText, 10, 64)
	end, endErr := strconv.ParseInt(endText, 10, 64)
	total, totalErr := strconv.ParseInt(totalText, 10, 64)
	if startErr != nil || endErr != nil || totalErr != nil || start < 0 || end < start || total <= end {
		return 0, 0, 0, errors.New("invalid Content-Range bounds")
	}
	return start, end, total, nil
}
