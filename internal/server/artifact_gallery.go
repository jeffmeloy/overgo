package server

import (
	"context"
	"errors"
	"io"
	"net/http"
	"slices"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

var errBrowseRepositoryUnavailable = errors.New("artifact browsing is not configured")

type artifactSummary struct {
	Descriptor artifact.Descriptor `json:"descriptor"`
	Producers  []artifact.ID       `json:"producers,omitempty"`
	Payload    bool                `json:"payload"`
}

type artifactGalleryResponse struct {
	Count     int               `json:"count"`
	Limit     int               `json:"limit"`
	Truncated bool              `json:"truncated"`
	Next      string            `json:"next,omitempty"`
	Artifacts []artifactSummary `json:"artifacts"`
}

func (h *Handler) artifactGallery(response http.ResponseWriter, request *http.Request) {
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	query, limit, err := h.artifactQuery(request)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	result, err := store.Query(request.Context(), query)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	next, err := encodeNextCursor(result.Next)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	items := make([]artifactSummary, 0, len(result.Artifacts))
	for _, descriptor := range result.Artifacts {
		_, payload := slices.BinarySearchFunc(result.Contents, descriptor.ID, func(content overgodb.ContentView, id artifact.ID) int {
			return artifact.CompareID(content.Artifact, id)
		})
		producers := make([]artifact.ID, 0)
		for _, edge := range result.Lineage {
			if edge.Child == descriptor.ID && edge.Relation == artifact.RelationProducedBy {
				producers = append(producers, edge.Parent)
			}
		}
		items = append(items, artifactSummary{Descriptor: descriptor, Producers: producers, Payload: payload})
	}
	writeJSON(response, http.StatusOK, artifactGalleryResponse{
		Count: result.Matched, Limit: limit, Truncated: result.Truncated, Next: next, Artifacts: items,
	})
}

func (h *Handler) artifactContent(response http.ResponseWriter, request *http.Request) {
	id, err := artifact.ParseID(request.URL.Query().Get("id"))
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	descriptor, reader, found, err := store.OpenContent(request.Context(), id)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	if !found {
		writeError(response, http.StatusNotFound, "payload_missing", "artifact has no inline payload")
		return
	}
	mediaType := descriptor.MediaType
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	response.Header().Set("Content-Type", mediaType)
	response.Header().Set("Content-Length", strconv.FormatUint(descriptor.Size, 10))
	response.WriteHeader(http.StatusOK)
	_, _ = io.Copy(response, reader)
}

func (h *Handler) artifactQuery(request *http.Request) (overgodb.Query, int, error) {
	values := request.URL.Query()
	limit := parseIntDefault(values.Get("limit"), h.config.MaxStoredResponses)
	var selected *artifact.ID
	if value := values.Get("id"); value != "" {
		id, err := artifact.ParseID(value)
		if err != nil {
			return overgodb.Query{}, 0, err
		}
		selected = &id
	}
	if limit <= 0 || limit > h.config.MaxStoredResponses {
		return overgodb.Query{}, 0, errors.New("artifact gallery: invalid limit")
	}
	query := overgodb.Query{
		Artifact: selected, MaxResults: limit,
		Projection: overgodb.ProjectArtifacts | overgodb.ProjectContentPresence | overgodb.ProjectParents,
	}
	if value := values.Get("cursor"); value != "" {
		cursor, err := overgodb.ParseQueryCursor(value)
		if err != nil {
			return overgodb.Query{}, 0, err
		}
		query.Cursor = &cursor
	}
	if selected == nil && values.Get("kind") != "" {
		value := values.Get("kind")
		kind, err := artifact.ParseKind(value)
		if err != nil {
			return overgodb.Query{}, 0, err
		}
		query.Kind = kind
	}
	return query, limit, nil
}

func encodeNextCursor(cursor *overgodb.QueryCursor) (string, error) {
	if cursor == nil {
		return "", nil
	}
	return overgodb.EncodeQueryCursor(*cursor)
}

func (h *Handler) browseStore(ctx context.Context) (*overgodb.Store, error) {
	if h.repository != nil {
		return h.repository, nil
	}
	if h.browseRepository == nil {
		return nil, errBrowseRepositoryUnavailable
	}
	if err := h.browseRepository.Refresh(ctx); err != nil {
		return nil, err
	}
	return h.browseRepository, nil
}

func (h *Handler) requireBrowseStore(response http.ResponseWriter, request *http.Request) (*overgodb.Store, bool) {
	store, err := h.browseStore(request.Context())
	if errors.Is(err, errBrowseRepositoryUnavailable) {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "artifact browsing is not configured")
		return nil, false
	}
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", "cannot refresh the artifact store")
		return nil, false
	}
	return store, true
}
