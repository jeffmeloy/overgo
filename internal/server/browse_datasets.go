package server

import (
	"context"
	"encoding/binary"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/trainingdata"
)

type datasetEntry struct {
	Name        string   `json:"name"`
	StorageKind string   `json:"storage_kind"`
	Source      string   `json:"source"`
	Modality    string   `json:"modality"`
	Modalities  []string `json:"modalities,omitempty"`
	Formats     []string `json:"formats,omitempty"`
	Files       uint64   `json:"files"`
	Bytes       uint64   `json:"bytes"`
	Available   bool     `json:"available"`
}

type browseDatasetsResponse struct {
	Catalog  string         `json:"catalog"`
	Count    int            `json:"count"`
	Datasets []datasetEntry `json:"datasets"`
}

type DatasetPreviewAPI interface {
	OpenDataset(context.Context, string) (*trainingdata.Dataset, error)
}

type datasetPreviewRequest struct {
	Name     string `json:"name"`
	Position uint64 `json:"position"`
	Limit    int    `json:"limit"`
}

// storeDatasetPreview opens registered datasets from the store the page browses.
type storeDatasetPreview struct {
	reader artifact.Reader
	// maxSamples bounds one decoded audio record to the preview's bytes.
	maxSamples uint64
}

// OpenDataset opens the dataset registered under name, every record.
func (preview storeDatasetPreview) OpenDataset(ctx context.Context, name string) (*trainingdata.Dataset, error) {
	return trainingdata.OpenRegistered(ctx, preview.reader, name, preview.maxSamples)
}

// datasetPreview: a launcher's resolver wins; otherwise the store the page browses answers.
func (h *Handler) datasetPreview(response http.ResponseWriter, request *http.Request) (DatasetPreviewAPI, bool) {
	if h.config.DatasetPreview != nil {
		return h.config.DatasetPreview, true
	}
	store, ok := h.requireBrowseStore(response, request)
	return storeDatasetPreview{reader: store, maxSamples: uint64(h.config.ResponseStoreBytes) / uint64(binary.Size(float32(0)))}, ok
}

func (h *Handler) previewDataset(response http.ResponseWriter, request *http.Request) {
	opener, ok := h.datasetPreview(response, request)
	if !ok {
		return
	}
	var body datasetPreviewRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if body.Name == "" || body.Limit <= 0 || body.Limit > h.config.MaxStoredResponses {
		writeInvalidRequestMessage(response, "invalid dataset preview request")
		return
	}
	value, err := opener.OpenDataset(request.Context(), body.Name)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	defer value.Close()
	preview, err := trainingdata.Preview(
		request.Context(), value, body.Position, body.Limit, uint64(h.config.ResponseStoreBytes),
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	writeJSON(response, http.StatusOK, preview)
}

// browseDatasets serves the active OvergoDB dataset catalog.
func (h *Handler) browseDatasets(response http.ResponseWriter, request *http.Request) {
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	coverage, err := dataset.InspectCatalog(request.Context(), store)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	if coverage.Catalog == nil {
		writeError(response, http.StatusNotFound, "not_found", "dataset catalog is unavailable")
		return
	}
	entries := make([]datasetEntry, len(coverage.Entries))
	for index, row := range coverage.Entries {
		entry := row.Entry
		entries[index] = datasetEntry{
			Name: entry.Name, StorageKind: entry.StorageKind, Source: entry.Source, Modality: entry.Modality,
			Modalities: entry.Modalities,
			Formats:    entry.Formats, Files: entry.Files, Bytes: entry.Bytes, Available: row.Available,
		}
	}
	writeJSON(response, http.StatusOK, browseDatasetsResponse{
		Catalog: coverage.Catalog.String(), Count: len(entries), Datasets: entries,
	})
}
