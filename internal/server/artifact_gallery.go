package server

import (
	"errors"
	"io"
	"net/http"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
)

type artifactSummary struct {
	Descriptor artifact.Descriptor `json:"descriptor"`
	Producers  []artifact.ID       `json:"producers,omitempty"`
	Payload    bool                `json:"payload"`
}

type artifactGalleryResponse struct {
	Offset    int               `json:"offset"`
	Limit     int               `json:"limit"`
	Truncated bool              `json:"truncated"`
	Artifacts []artifactSummary `json:"artifacts"`
}

func (h *Handler) artifactGallery(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	store, ok := h.openBrowseStore(response)
	if !ok {
		return
	}
	defer store.Close()
	query, offset, limit, err := h.artifactQuery(request)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	result, err := store.Query(request.Context(), query)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "repodb_error", err.Error())
		return
	}
	descriptors := result.Artifacts
	if offset < len(descriptors) {
		descriptors = descriptors[offset:]
	} else {
		descriptors = nil
	}
	truncated := result.Truncated || len(descriptors) > limit
	if len(descriptors) > limit {
		descriptors = descriptors[:limit]
	}
	items := make([]artifactSummary, 0, len(descriptors))
	for _, descriptor := range descriptors {
		payload, openErr := store.HasContent(request.Context(), descriptor.ID)
		if openErr != nil {
			writeError(response, http.StatusInternalServerError, "repodb_error", openErr.Error())
			return
		}
		parents, parentErr := store.Parents(request.Context(), descriptor.ID)
		if parentErr != nil {
			writeError(response, http.StatusInternalServerError, "repodb_error", parentErr.Error())
			return
		}
		producers := make([]artifact.ID, 0, len(parents))
		for _, edge := range parents {
			if edge.Relation == artifact.RelationProducedBy {
				producers = append(producers, edge.Parent)
			}
		}
		items = append(items, artifactSummary{Descriptor: descriptor, Producers: producers, Payload: payload})
	}
	writeJSON(response, http.StatusOK, artifactGalleryResponse{
		Offset: offset, Limit: limit, Truncated: truncated, Artifacts: items,
	})
}

func (h *Handler) artifactContent(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	id, err := artifact.ParseID(request.URL.Query().Get("id"))
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	store, ok := h.openBrowseStore(response)
	if !ok {
		return
	}
	defer store.Close()
	descriptor, reader, found, err := store.OpenContent(request.Context(), id)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "repodb_error", err.Error())
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

func (h *Handler) artifactQuery(request *http.Request) (repodb.Query, int, int, error) {
	values := request.URL.Query()
	offset := parseIntDefault(values.Get("offset"), 0)
	limit := parseIntDefault(values.Get("limit"), h.config.MaxStoredResponses)
	var selected *artifact.ID
	if value := values.Get("id"); value != "" {
		id, err := artifact.ParseID(value)
		if err != nil {
			return repodb.Query{}, 0, 0, err
		}
		selected, offset = &id, 0
	}
	if offset < 0 || limit <= 0 || limit > h.config.MaxStoredResponses || offset > repodb.MaxQueryResults-limit {
		return repodb.Query{}, 0, 0, errors.New("artifact gallery: invalid offset or limit")
	}
	query := repodb.Query{Artifact: selected, MaxResults: offset + limit}
	if query.MaxResults < repodb.MaxQueryResults {
		query.MaxResults++
	}
	if selected == nil && values.Get("kind") != "" {
		value := values.Get("kind")
		kind, err := artifact.ParseKind(value)
		if err != nil {
			return repodb.Query{}, 0, 0, err
		}
		query.Kind = kind
	}
	return query, offset, limit, nil
}

func (h *Handler) openBrowseStore(response http.ResponseWriter) (*repodb.Store, bool) {
	if h.config.RepoDBPath == "" {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "artifact browsing is not configured")
		return nil, false
	}
	store, err := repodb.OpenReadOnly(h.config.RepoDBPath)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "repodb_error", "cannot open the artifact store")
		return nil, false
	}
	return store, true
}
