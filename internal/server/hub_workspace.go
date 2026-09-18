package server

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/hfhub"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

// The hub workspace is the workbench's discovery surface: the local catalog
// with its evidence state, hub search over models and datasets, and verified
// download jobs that fill the data root. Every route is JSON over the same
// mux as the rest of the workbench.

// catalogModels lists every model the store activates for any task, as
// the discovery capability catalog proves them: activation aliases drive
// enumeration, presence is a stat of recorded bytes, and each task
// carries its own tier or defect. The inference activation also fills
// the flat recipe/tier fields so existing consumers keep their shape.
func (h *Handler) catalogModels(response http.ResponseWriter, request *http.Request) {
	if h.config.Repository == nil {
		writeError(response, http.StatusServiceUnavailable, "hub_unavailable", "no artifact repository is configured")
		return
	}
	entries, truncated, err := discovery.CapabilityCatalog(
		request.Context(), h.config.Repository, h.config.MaxStoredResponses, h.catalogMemo,
	)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "catalog_error", err.Error())
		return
	}
	// Identities hashed during this pass persist so the next process
	// answers from stat checks instead of re-hashing the model bytes.
	if err := discovery.PublishMemo(request.Context(), h.config.Repository, h.catalogMemo); err != nil {
		log.Printf("catalog: persist identity memo: %v", err)
	}
	evidence := h.catalogEvidence(request.Context(), h.workspaceSuiteNames(request.Context()))
	writeJSON(response, http.StatusOK, map[string]any{
		"models": catalogListing(entries, evidence), "truncated": truncated, "coverage": h.catalogCoverage(request.Context()),
	})
}

// catalogListing: the wire rows for catalogued entries with their evidence; the idle shell lists the same rows without evidence.
func catalogListing(entries []discovery.CatalogEntry, evidence evaluation.EvidenceIndex) []catalogModel {
	listed := make([]catalogModel, 0, len(entries))
	for _, entry := range entries {
		model := catalogModel{
			Model: idText(entry.Model), Location: entry.Location, Present: entry.Present, KeyEnvironment: entry.KeyEnvironment,
		}
		if summary, measured := evidence.BenchmarksByLocation[entry.Location]; measured {
			model.Benchmark = &summary
		}
		for _, capability := range entry.Capabilities {
			model.Capabilities = append(model.Capabilities, catalogCapability{
				Task: string(capability.Task), Recipe: idText(capability.Recipe),
				Tier: string(capability.Tier), Stale: capability.Stale,
			})
			if capability.Task == recipe.TaskInference {
				model.Recipe, model.Tier, model.Stale = idText(capability.Recipe), string(capability.Tier), capability.Stale
				model.Evals = evidence.EvaluationsByRecipe[capability.Recipe]
			}
		}
		listed = append(listed, model)
	}
	return listed
}

// catalogCoverage reports the registered denominators beside the
// activation listing: how many architecture profiles and datasets the
// store SHOULD hold against how many it publishes. Coverage that cannot
// be derived reports its defect instead of vanishing.
func (h *Handler) catalogCoverage(ctx context.Context) map[string]any {
	coverage := map[string]any{}
	profiles, err := modelrecipe.InspectArchitectureProfileCatalog(ctx, h.config.Repository)
	if err != nil {
		coverage["profiles"] = map[string]any{"error": err.Error()}
	} else {
		coverage["profiles"] = map[string]any{
			"registered": profiles.Registered, "published": profiles.Published, "complete": profiles.Complete,
		}
	}
	datasets, err := dataset.InspectCatalog(ctx, h.config.Repository)
	if err != nil {
		coverage["datasets"] = map[string]any{"error": err.Error()}
	} else {
		coverage["datasets"] = map[string]any{
			"registered": datasets.Registered, "published": datasets.Published,
			"available": datasets.Available, "complete": datasets.Complete,
		}
	}
	return coverage
}

// catalogModel is the catalog's wire shape. A stale entry can carry invalid
// artifact identities; the codec refuses to marshal those, and one broken
// activation must not blank the whole catalog, so identities render as text
// with absence rendered empty. The flat recipe/tier/stale fields mirror the
// inference capability for consumers of the original inference-only shape.
type catalogModel struct {
	Model    string `json:"model"`
	Recipe   string `json:"recipe"`
	Tier     string `json:"tier,omitzero"`
	Location string `json:"location"`
	Present  bool   `json:"present"`
	Stale    string `json:"stale,omitzero"`
	// KeyEnvironment names the variable a hosted model's provider key
	// lives in, so the page can take the key when the entry is refused.
	KeyEnvironment string              `json:"key_environment,omitzero"`
	Capabilities   []catalogCapability `json:"capabilities,omitempty"`
	// Benchmark and Evals surface the model's committed evidence beside
	// its entry: perf from the latest benchmark claim, quality from the
	// latest evaluation per derived suite.
	Benchmark *evaluation.BenchmarkSummary `json:"benchmark,omitempty"`
	Evals     []evaluation.EvalSummary     `json:"evals,omitempty"`
}

// catalogCapability is one task activation on a catalogued model.
type catalogCapability struct {
	Task   string `json:"task"`
	Recipe string `json:"recipe"`
	Tier   string `json:"tier,omitzero"`
	Stale  string `json:"stale,omitzero"`
}

func idText(id artifact.ID) string {
	if !id.Valid() {
		return ""
	}
	return id.String()
}

// hubSearch proxies one bounded discovery query to the configured hub.
func (h *Handler) hubSearch(response http.ResponseWriter, request *http.Request) {
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
	Revision    string `json:"revision,omitzero"`
	Destination string `json:"destination"`
	State       string `json:"state"`
	Error       string `json:"error,omitzero"`
	File        string `json:"file,omitzero"`
	Received    int64  `json:"received"`
	Total       int64  `json:"total"`
	Files       int    `json:"files"`
}

const (
	// defaultHubSearchLimit sizes an uninstructed hub search page; the limit
	// parameter overrides it per request up to maxHubSearchLimit.
	defaultHubSearchLimit = 20
	// maxHubSearchLimit bounds one proxied hub search.
	maxHubSearchLimit = 100

	downloadStateRunning   = "running"
	downloadStateSucceeded = "succeeded"
	downloadStateFailed    = "failed"
	downloadStateCancelled = "cancelled"
)

// errDownloadsShutDown refuses admission once the registry owns shutdown.
var errDownloadsShutDown = errors.New("download workspace is shut down")

type downloadRegistry struct {
	mu                      sync.Mutex
	next                    atomic.Uint64
	jobs                    map[uint64]*DownloadJob
	cancels                 map[uint64]context.CancelFunc
	maxRunning, maxRetained int
	// transfers tracks every running transfer goroutine so shutdown
	// returns only after the last one unwound; closed refuses new work.
	transfers sync.WaitGroup
	closed    bool
}

func newDownloadRegistry(maxRunning, maxRetained int) downloadRegistry {
	return downloadRegistry{maxRunning: maxRunning, maxRetained: maxRetained}
}

// admit registers a new job under the concurrency and retention
// bounds; a full running set is a refusal, and the oldest finished
// jobs make room for history.
func (r *downloadRegistry) admit(job *DownloadJob, cancel context.CancelFunc) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return errDownloadsShutDown
	}
	running := 0
	for _, existing := range r.jobs {
		if existing.State == downloadStateRunning {
			running++
		}
	}
	if running >= r.maxRunning {
		return fmt.Errorf("download backlog is full: %d transfer(s) already running", running)
	}
	if r.jobs == nil {
		r.jobs, r.cancels = map[uint64]*DownloadJob{}, map[uint64]context.CancelFunc{}
	}
	for len(r.jobs) >= r.maxRetained {
		oldest := uint64(0)
		for id, existing := range r.jobs {
			if existing.State != downloadStateRunning && (oldest == 0 || id < oldest) {
				oldest = id
			}
		}
		if oldest == 0 {
			break
		}
		delete(r.jobs, oldest)
		delete(r.cancels, oldest)
	}
	r.jobs[job.ID] = job
	r.cancels[job.ID] = cancel
	r.transfers.Add(1)
	return nil
}

// shutdown is the one download lifecycle owner on the way out: it stops
// admission, cancels every running transfer with the interruption
// recorded on the job, and returns only after the last transfer
// goroutine has unwound -- so close never races a write to disk. The
// transfer path retains identity-bound progress when its context ends.
func (r *downloadRegistry) shutdown() {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}
	r.closed = true
	for id, job := range r.jobs {
		if job.State != downloadStateRunning {
			continue
		}
		if stop := r.cancels[id]; stop != nil {
			stop()
		}
		job.State, job.Error = downloadStateCancelled, "server shutdown interrupted the transfer"
	}
	r.mu.Unlock()
	r.transfers.Wait()
}

// cancel stops one running job; finished jobs report their state.
func (r *downloadRegistry) cancel(id uint64) (string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	job, found := r.jobs[id]
	if !found {
		return "", false
	}
	if job.State == downloadStateRunning {
		if stop := r.cancels[id]; stop != nil {
			stop()
		}
		job.State = downloadStateCancelled
	}
	return job.State, true
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
	// Timeout is an optional Go duration covering resolution and transfer.
	// Empty or zero keeps the existing policy: run until completion, DELETE,
	// or server shutdown. Browser disconnect alone does not cancel the job.
	Timeout string `json:"timeout"`
}

// hubDownloads starts a download job (POST), lists jobs (GET), or
// cancels one (DELETE with ?id=). Jobs write only under the configured
// download root, through the client's verified download path.
func (h *Handler) hubDownloads(response http.ResponseWriter, request *http.Request) {
	switch request.Method {
	case http.MethodGet:
		writeJSON(response, http.StatusOK, map[string]any{"downloads": h.downloads.snapshot()})
	case http.MethodPost:
		h.startHubDownload(response, request)
	case http.MethodDelete:
		h.cancelHubDownload(response, request)
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
	var timeout time.Duration
	if body.Timeout != "" {
		var err error
		timeout, err = time.ParseDuration(body.Timeout)
		if err != nil || timeout < 0 {
			writeError(response, http.StatusBadRequest, "invalid_request", "timeout must be a non-negative duration")
			return
		}
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
	// The transfer detaches from the request context deliberately --
	// closing the browser tab must not abort a multi-gigabyte pull --
	// but every job carries its own cancel, reachable over DELETE.
	var ctx context.Context
	var cancel context.CancelFunc
	if timeout > 0 {
		ctx, cancel = context.WithTimeoutCause(context.Background(), timeout, fmt.Errorf("hub download caller budget: %w", context.DeadlineExceeded))
	} else {
		ctx, cancel = context.WithCancel(context.Background())
	}
	admitted := *job
	if err := h.downloads.admit(job, cancel); err != nil {
		cancel()
		if errors.Is(err, errDownloadsShutDown) {
			writeError(response, http.StatusServiceUnavailable, "hub_unavailable", err.Error())
			return
		}
		writeError(response, http.StatusTooManyRequests, "download_backlog", err.Error())
		return
	}
	go func() {
		defer cancel()
		h.runHubDownload(ctx, client, kind, body, destination, job)
	}()
	writeJSON(response, http.StatusAccepted, admitted)
}

func (h *Handler) cancelHubDownload(response http.ResponseWriter, request *http.Request) {
	id, err := strconv.ParseUint(request.URL.Query().Get("id"), identifierRadix, identifierBits)
	if err != nil || id == 0 {
		writeError(response, http.StatusBadRequest, "invalid_request", "cancel requires a numeric job id")
		return
	}
	state, found := h.downloads.cancel(id)
	if !found {
		writeError(response, http.StatusNotFound, "unknown_download", "no such download job")
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{"id": id, "state": state})
}

func (h *Handler) runHubDownload(ctx context.Context, client *hfhub.Client, kind hfhub.RepoKind, body downloadRequestBody, destination string, job *DownloadJob) {
	defer h.downloads.transfers.Done()
	resolved, err := client.Download(ctx, hfhub.DownloadRequest{
		Kind: kind, Repository: body.Repository, Revision: body.Revision, Destination: destination,
		Observe: func(progress hfhub.Progress) {
			h.downloads.mu.Lock()
			job.File, job.Received, job.Total = progress.Path, progress.Received, progress.Total
			h.downloads.mu.Unlock()
		},
	})
	h.downloads.mu.Lock()
	defer h.downloads.mu.Unlock()
	if job.State == downloadStateCancelled {
		return
	}
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
