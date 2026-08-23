package server

import (
	"context"
	"errors"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/hfhub"
	"overgo/internal/strictjson"
)

// The hub workspace is the workbench's discovery surface: the local catalog
// with its evidence state, hub search over models and datasets, and verified
// download jobs that fill the data root. Every route is JSON over the same
// mux as the rest of the workbench.

// catalogModels lists the store's servable models exactly as the discovery
// query proves them: hashed on disk, active recipe, verified evidence.
func (h *Handler) catalogModels(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		methodNotAllowed(response)
		return
	}
	if h.config.Repository == nil {
		writeError(response, http.StatusServiceUnavailable, "hub_unavailable", "no artifact repository is configured")
		return
	}
	entries, err := discovery.Servable(request.Context(), h.config.Repository, maxCatalogEntries)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "catalog_error", err.Error())
		return
	}
	listed := make([]catalogModel, 0, len(entries))
	for _, entry := range entries {
		listed = append(listed, catalogModel{
			Model: idText(entry.Model), Recipe: idText(entry.Recipe), Tier: string(entry.Tier),
			Location: entry.Location, Present: entry.Present, Stale: entry.Stale,
		})
	}
	writeJSON(response, http.StatusOK, map[string]any{"models": listed})
}

// catalogModel is the catalog's wire shape. A stale entry can carry invalid
// artifact identities; the codec refuses to marshal those, and one broken
// activation must not blank the whole catalog, so identities render as text
// with absence rendered empty.
type catalogModel struct {
	Model    string `json:"model"`
	Recipe   string `json:"recipe"`
	Tier     string `json:"tier,omitempty"`
	Location string `json:"location"`
	Present  bool   `json:"present"`
	Stale    string `json:"stale,omitempty"`
}

func idText(id artifact.ID) string {
	if !id.Valid() {
		return ""
	}
	return id.String()
}

// hubSearch proxies one bounded discovery query to the configured hub.
func (h *Handler) hubSearch(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		methodNotAllowed(response)
		return
	}
	client, err := h.hubClient()
	if err != nil {
		writeError(response, http.StatusServiceUnavailable, "hub_unavailable", err.Error())
		return
	}
	query := hfhub.SearchQuery{
		Kind:   hfhub.KindModel,
		Search: request.URL.Query().Get("q"),
		Filter: request.URL.Query().Get("filter"),
		Limit:  defaultHubSearchLimit,
	}
	if request.URL.Query().Get("kind") == string(hfhub.KindDataset) {
		query.Kind = hfhub.KindDataset
	}
	if raw := request.URL.Query().Get("limit"); raw != "" {
		parsed, parseErr := strconv.Atoi(raw)
		if parseErr != nil || parsed <= 0 || parsed > maxHubSearchLimit {
			writeError(response, http.StatusBadRequest, "invalid_request", "limit must be a positive integer within the search bound")
			return
		}
		query.Limit = parsed
	}
	listings, err := client.Search(request.Context(), query)
	if err != nil {
		writeError(response, http.StatusBadGateway, "hub_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{"results": listings})
}

// DownloadJob is one hub download's observable state.
type DownloadJob struct {
	ID          uint64 `json:"id"`
	Kind        string `json:"kind"`
	Repository  string `json:"repository"`
	Revision    string `json:"revision,omitempty"`
	Destination string `json:"destination"`
	State       string `json:"state"`
	Error       string `json:"error,omitempty"`
	File        string `json:"file,omitempty"`
	Received    int64  `json:"received"`
	Total       int64  `json:"total"`
	Files       int    `json:"files"`
}

const (
	// maxCatalogEntries bounds one catalog response; it only has to exceed
	// any legitimate servable-model count.
	maxCatalogEntries = 512
	// defaultHubSearchLimit sizes an uninstructed hub search page; the limit
	// parameter overrides it per request up to maxHubSearchLimit.
	defaultHubSearchLimit = 20
	// maxHubSearchLimit bounds one proxied hub search.
	maxHubSearchLimit = 100

	downloadStateRunning   = "running"
	downloadStateSucceeded = "succeeded"
	downloadStateFailed    = "failed"
)

type downloadRegistry struct {
	mu   sync.Mutex
	next atomic.Uint64
	jobs map[uint64]*DownloadJob
}

func (r *downloadRegistry) snapshot() []DownloadJob {
	r.mu.Lock()
	defer r.mu.Unlock()
	listed := make([]DownloadJob, 0, len(r.jobs))
	for _, job := range r.jobs {
		listed = append(listed, *job)
	}
	sort.Slice(listed, func(left, right int) bool { return listed[left].ID > listed[right].ID })
	return listed
}

type downloadRequestBody struct {
	Kind       string `json:"kind"`
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	// Directory names the destination inside the configured download root;
	// empty derives it from the repository name.
	Directory string `json:"directory"`
}

// hubDownloads starts a download job (POST) or lists jobs (GET). Jobs write
// only under the configured download root, through the client's verified
// download path.
func (h *Handler) hubDownloads(response http.ResponseWriter, request *http.Request) {
	switch request.Method {
	case http.MethodGet:
		writeJSON(response, http.StatusOK, map[string]any{"downloads": h.downloads.snapshot()})
	case http.MethodPost:
		h.startHubDownload(response, request)
	default:
		methodNotAllowed(response)
	}
}

func (h *Handler) startHubDownload(response http.ResponseWriter, request *http.Request) {
	client, err := h.hubClient()
	if err != nil {
		writeError(response, http.StatusServiceUnavailable, "hub_unavailable", err.Error())
		return
	}
	root := strings.TrimSpace(h.config.HubDownloadRoot)
	if root == "" {
		writeError(response, http.StatusServiceUnavailable, "hub_unavailable", "no hub download root is configured")
		return
	}
	var body downloadRequestBody
	if err := strictjson.Decode(request.Body, &body); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	kind := hfhub.KindModel
	if body.Kind == string(hfhub.KindDataset) {
		kind = hfhub.KindDataset
	}
	directory := strings.TrimSpace(body.Directory)
	if directory == "" {
		segments := strings.Split(body.Repository, "/")
		directory = segments[len(segments)-1]
	}
	destination, err := hubDestination(root, directory)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	job := &DownloadJob{
		ID: h.downloads.next.Add(1), Kind: string(kind), Repository: body.Repository,
		Revision: body.Revision, Destination: destination, State: downloadStateRunning,
	}
	h.downloads.mu.Lock()
	if h.downloads.jobs == nil {
		h.downloads.jobs = map[uint64]*DownloadJob{}
	}
	h.downloads.jobs[job.ID] = job
	h.downloads.mu.Unlock()
	go h.runHubDownload(client, kind, body, destination, job)
	writeJSON(response, http.StatusAccepted, *job)
}

func (h *Handler) runHubDownload(client *hfhub.Client, kind hfhub.RepoKind, body downloadRequestBody, destination string, job *DownloadJob) {
	resolved, err := client.Download(context.Background(), hfhub.DownloadRequest{
		Kind: kind, Repository: body.Repository, Revision: body.Revision, Destination: destination,
		Observe: func(progress hfhub.Progress) {
			h.downloads.mu.Lock()
			job.File, job.Received, job.Total = progress.Path, progress.Received, progress.Total
			h.downloads.mu.Unlock()
		},
	})
	h.downloads.mu.Lock()
	defer h.downloads.mu.Unlock()
	if err != nil {
		job.State, job.Error = downloadStateFailed, err.Error()
		return
	}
	job.State, job.Files, job.Revision = downloadStateSucceeded, len(resolved.Files), resolved.Revision
}

// hubDestination confines job directories to the configured root.
func hubDestination(root, directory string) (string, error) {
	if directory == "" || strings.ContainsAny(directory, "/\\") || strings.Contains(directory, "..") {
		return "", errors.New("download directory must be one plain path segment")
	}
	return root + string('/') + directory, nil
}

func (h *Handler) hubClient() (*hfhub.Client, error) {
	return hfhub.New(h.config.HubEndpoint, h.config.HubToken)
}

func methodNotAllowed(response http.ResponseWriter) {
	writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "method not allowed")
}
