package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"overgo/internal/trainingdata"
)

const browseDatasetManifestMaxBytes = 8 << 20

// datasetsManifest: the subset of datasets/manifest.json the browse surface
// reads. Unknown fields are ignored, so the manifest can carry more than this.
type datasetsManifest struct {
	Datasets []datasetEntry `json:"datasets"`
}

type datasetEntry struct {
	Name       string `json:"name"`
	Family     string `json:"family"`
	Modality   string `json:"modality"`
	Loader     string `json:"loader"`
	Provenance string `json:"provenance"`
	Language   string `json:"language"`
}

type browseDatasetsResponse struct {
	Root     string         `json:"root"`
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

func (h *Handler) previewDataset(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	if h.config.DatasetPreview == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "dataset preview is not configured")
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
	dataset, err := h.config.DatasetPreview.OpenDataset(request.Context(), body.Name)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	defer dataset.Close()
	preview, err := trainingdata.Preview(
		request.Context(), dataset, body.Position, body.Limit, uint64(h.config.ResponseStoreBytes),
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	writeJSON(response, http.StatusOK, preview)
}

// browseDatasets: read-only dataset registry from <DatasetsRoot>/manifest.json —
// name, family, modality, loader, provenance, language per dataset. No file
// access beyond the manifest; sampling is a separate, bounded endpoint.
func (h *Handler) browseDatasets(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	if h.config.DatasetsRoot == "" {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "dataset browsing is not configured")
		return
	}
	file, err := os.Open(filepath.Join(h.config.DatasetsRoot, "manifest.json"))
	if err != nil {
		writeError(response, http.StatusNotFound, "not_found", "dataset manifest is unavailable")
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || info.Size() > browseDatasetManifestMaxBytes {
		writeError(response, http.StatusRequestEntityTooLarge, "manifest_too_large", "dataset manifest exceeds the browse limit")
		return
	}
	var manifest datasetsManifest
	decoder := json.NewDecoder(io.LimitReader(file, browseDatasetManifestMaxBytes+1))
	if err := decoder.Decode(&manifest); err != nil {
		writeError(response, http.StatusInternalServerError, "invalid_manifest", "dataset manifest is not valid JSON")
		return
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		writeError(response, http.StatusInternalServerError, "invalid_manifest", "dataset manifest has trailing content")
		return
	}
	writeJSON(response, http.StatusOK, browseDatasetsResponse{
		Root:     h.config.DatasetsRoot,
		Count:    len(manifest.Datasets),
		Datasets: manifest.Datasets,
	})
}
