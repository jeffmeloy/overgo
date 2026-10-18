package hfhub

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"slices"
	"strings"
)

// NewModelScope connects the verified downloader to ModelScope model repositories.
// The source must be an immutable Git commit rather than a branch or release label.
func NewModelScope(endpoint, token string) (*Client, error) {
	if strings.TrimSpace(endpoint) == "" {
		endpoint = "https://www.modelscope.cn"
	}
	client, err := New(endpoint, token)
	if err != nil {
		return nil, err
	}
	client.modelScope = true
	return client, nil
}

type modelScopeFile struct {
	Path   string
	Type   string
	Size   int64
	Sha256 string
}

func (c *Client) resolveModelScope(ctx context.Context, kind RepoKind, repository, revision string) (RepoRevision, error) {
	commit, err := hex.DecodeString(revision)
	if err != nil || len(commit) != sha1.Size {
		return RepoRevision{}, fmt.Errorf("hfhub: ModelScope revision %q must be a full Git SHA-1 commit", revision)
	}
	if kind != KindModel {
		return RepoRevision{}, fmt.Errorf("hfhub: ModelScope repository kind %q is unsupported; use models", kind)
	}
	query := url.Values{"Revision": {revision}, "Recursive": {"true"}}
	var response struct {
		Code    int
		Success bool
		Message string
		Data    struct{ Files []modelScopeFile }
	}
	if err := c.getJSON(ctx, "/api/v1/models/"+escapePath(repository)+"/repo/files?"+query.Encode(), &response); err != nil {
		return RepoRevision{}, err
	}
	if !response.Success || response.Code != http.StatusOK {
		return RepoRevision{}, fmt.Errorf("hfhub: ModelScope inventory refused with code %d: %s", response.Code, response.Message)
	}
	resolved := RepoRevision{ID: repository, Revision: revision}
	seen := map[string]bool{}
	for _, entry := range response.Data.Files {
		if entry.Type == "tree" {
			continue
		}
		if entry.Type != "blob" {
			return RepoRevision{}, fmt.Errorf("hfhub: ModelScope entry %q has unsupported type %q", entry.Path, entry.Type)
		}
		if entry.Path == "" || path.IsAbs(entry.Path) || path.Clean(entry.Path) != entry.Path || entry.Path == "." || strings.HasPrefix(entry.Path, "../") || strings.ContainsAny(entry.Path, "\\:\x00") || seen[entry.Path] {
			return RepoRevision{}, fmt.Errorf("hfhub: ModelScope file path %q must be unique and repository-relative", entry.Path)
		}
		digest, err := hex.DecodeString(entry.Sha256)
		if err != nil || len(digest) != sha256.Size || entry.Size < 0 {
			return RepoRevision{}, fmt.Errorf("hfhub: ModelScope file %q requires SHA-256 and nonnegative byte size", entry.Path)
		}
		seen[entry.Path] = true
		resolved.Files = append(resolved.Files, RepoFile{Path: entry.Path, Size: entry.Size, SHA256: strings.ToLower(entry.Sha256)})
	}
	if len(resolved.Files) == 0 {
		return RepoRevision{}, errors.New("hfhub: ModelScope inventory contains no files")
	}
	slices.SortFunc(resolved.Files, func(a, b RepoFile) int { return strings.Compare(a.Path, b.Path) })
	return resolved, nil
}

func (c *Client) modelScopeFileURL(repository, revision, file string) string {
	query := url.Values{"Revision": {revision}, "FilePath": {file}}
	return c.endpoint + "/api/v1/models/" + escapePath(repository) + "/repo?" + query.Encode()
}
