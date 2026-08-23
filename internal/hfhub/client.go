// Package hfhub speaks the Hugging Face Hub HTTP API: model and dataset
// discovery, file resolution, and verified download. It is the network half
// of model intake -- hfrepo reads checkpoint directories already on disk;
// this package fills those directories from the hub.
//
// Every outbound request goes to one configured endpoint (the public hub by
// default), and every downloaded file that declares a size or an LFS sha256
// is verified against it before the file is surrendered to its final name --
// a partial or tampered download never lands under the requested path.
package hfhub

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const (
	// PublicEndpoint is the hub every client reaches unless redirected to a
	// mirror; tests point the client at a local server instead.
	PublicEndpoint = "https://huggingface.co"
	// maxListingBytes bounds one discovery response.
	maxListingBytes = 32 << 20
	requestTimeout  = 60 * time.Second
)

// Client is one configured hub connection.
type Client struct {
	endpoint string
	token    string
	http     *http.Client
}

// New returns a client for the endpoint; an empty endpoint means the public
// hub. The token is optional and sent only as a bearer credential to that
// endpoint, never anywhere else.
func New(endpoint, token string) (*Client, error) {
	if strings.TrimSpace(endpoint) == "" {
		endpoint = PublicEndpoint
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("hfhub: invalid endpoint %q", endpoint)
	}
	return &Client{
		endpoint: strings.TrimRight(endpoint, "/"),
		token:    strings.TrimSpace(token),
		http:     &http.Client{Timeout: requestTimeout},
	}, nil
}

// RepoKind selects the hub namespace being addressed.
type RepoKind string

const (
	// KindModel addresses the hub's model repositories.
	KindModel RepoKind = "models"
	// KindDataset addresses the hub's dataset repositories.
	KindDataset RepoKind = "datasets"
)

func (kind RepoKind) valid() bool { return kind == KindModel || kind == KindDataset }

// Listing is one discovery result.
type Listing struct {
	ID        string   `json:"id"`
	Author    string   `json:"author,omitempty"`
	Downloads int64    `json:"downloads,omitempty"`
	Likes     int64    `json:"likes,omitempty"`
	Gated     bool     `json:"gated,omitempty"`
	Private   bool     `json:"private,omitempty"`
	Tags      []string `json:"tags,omitempty"`
}

// SearchQuery bounds one discovery request.
type SearchQuery struct {
	Kind   RepoKind
	Search string
	Filter string
	Limit  int
}

// Search lists repositories matching the query, most-downloaded first.
func (c *Client) Search(ctx context.Context, query SearchQuery) ([]Listing, error) {
	if !query.Kind.valid() {
		return nil, fmt.Errorf("hfhub: invalid repository kind %q", query.Kind)
	}
	if query.Limit <= 0 {
		return nil, errors.New("hfhub: search requires a positive result limit")
	}
	values := url.Values{}
	values.Set("sort", "downloads")
	values.Set("direction", "-1")
	values.Set("limit", strconv.Itoa(query.Limit))
	if strings.TrimSpace(query.Search) != "" {
		values.Set("search", query.Search)
	}
	if strings.TrimSpace(query.Filter) != "" {
		values.Set("filter", query.Filter)
	}
	var listings []Listing
	err := c.getJSON(ctx, "/api/"+string(query.Kind)+"?"+values.Encode(), &listings)
	return listings, err
}

// RepoFile is one downloadable file in a repository revision.
type RepoFile struct {
	Path   string `json:"rfilename"`
	Size   int64  `json:"size,omitempty"`
	SHA256 string `json:"-"`
}

type siblingLFS struct {
	SHA256 string `json:"sha256,omitempty"`
	Size   int64  `json:"size,omitempty"`
}

type sibling struct {
	Path string      `json:"rfilename"`
	Size int64       `json:"size,omitempty"`
	LFS  *siblingLFS `json:"lfs,omitempty"`
}

type repoInfo struct {
	ID       string    `json:"id"`
	SHA      string    `json:"sha,omitempty"`
	Siblings []sibling `json:"siblings,omitempty"`
}

// RepoRevision is a resolved repository file inventory at one revision.
type RepoRevision struct {
	ID       string
	Revision string
	Files    []RepoFile
}

// Resolve lists a repository revision's files with declared sizes and LFS
// digests. An empty revision resolves the default branch.
func (c *Client) Resolve(ctx context.Context, kind RepoKind, repository, revision string) (RepoRevision, error) {
	if !kind.valid() {
		return RepoRevision{}, fmt.Errorf("hfhub: invalid repository kind %q", kind)
	}
	if !validRepositoryID(repository) {
		return RepoRevision{}, fmt.Errorf("hfhub: invalid repository id %q", repository)
	}
	path := "/api/" + string(kind) + "/" + repository
	if strings.TrimSpace(revision) != "" {
		path += "/revision/" + url.PathEscape(revision)
	}
	var info repoInfo
	if err := c.getJSON(ctx, path+"?blobs=true", &info); err != nil {
		return RepoRevision{}, err
	}
	resolved := RepoRevision{ID: info.ID, Revision: revision}
	if resolved.Revision == "" {
		resolved.Revision = info.SHA
	}
	for _, entry := range info.Siblings {
		file := RepoFile{Path: entry.Path, Size: entry.Size}
		if entry.LFS != nil {
			file.SHA256 = entry.LFS.SHA256
			if entry.LFS.Size > 0 {
				file.Size = entry.LFS.Size
			}
		}
		resolved.Files = append(resolved.Files, file)
	}
	return resolved, nil
}

// fileURL is the raw-content location for one file at one revision.
func (c *Client) fileURL(kind RepoKind, repository, revision, path string) string {
	prefix := ""
	if kind == KindDataset {
		prefix = "/datasets"
	}
	if strings.TrimSpace(revision) == "" {
		revision = "main"
	}
	return c.endpoint + prefix + "/" + repository + "/resolve/" + url.PathEscape(revision) + "/" + escapePath(path)
}

func escapePath(path string) string {
	segments := strings.Split(path, "/")
	for index, segment := range segments {
		segments[index] = url.PathEscape(segment)
	}
	return strings.Join(segments, "/")
}

func validRepositoryID(id string) bool {
	if id == "" || len(id) > 512 || strings.Contains(id, "..") {
		return false
	}
	segments := strings.Split(id, "/")
	if len(segments) > 2 {
		return false
	}
	for _, segment := range segments {
		if segment == "" || strings.ContainsAny(segment, "\\?#%") {
			return false
		}
	}
	return true
}

func (c *Client) getJSON(ctx context.Context, path string, value any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint+path, nil)
	if err != nil {
		return err
	}
	c.authorize(request)
	response, err := c.http.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return statusError(response)
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxListingBytes+1))
	if err != nil {
		return err
	}
	if len(body) > maxListingBytes {
		return errors.New("hfhub: listing response exceeds size bound")
	}
	return json.Unmarshal(body, value)
}

func (c *Client) authorize(request *http.Request) {
	if c.token != "" {
		request.Header.Set("Authorization", "Bearer "+c.token)
	}
}

func statusError(response *http.Response) error {
	detail, _ := io.ReadAll(io.LimitReader(response.Body, 512))
	text := strings.TrimSpace(string(detail))
	if text == "" {
		text = response.Status
	}
	return fmt.Errorf("hfhub: %s: %s", response.Request.URL.Path, text)
}
