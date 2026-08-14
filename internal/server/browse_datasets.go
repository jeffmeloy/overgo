package server

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
)

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

// browseDatasets: read-only dataset registry from <DatasetsRoot>/manifest.json —
// name, family, modality, loader, provenance, language per dataset. No file
// access beyond the manifest; sampling is a separate, bounded endpoint.
func (h *Handler) browseDatasets(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	if h.config.DatasetsRoot == "" {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "dataset browsing is not configured")
		return
	}
	data, err := os.ReadFile(filepath.Join(h.config.DatasetsRoot, "manifest.json"))
	if err != nil {
		writeError(response, http.StatusNotFound, "not_found", "dataset manifest is unavailable")
		return
	}
	var manifest datasetsManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		writeError(response, http.StatusInternalServerError, "invalid_manifest", "dataset manifest is not valid JSON")
		return
	}
	writeJSON(response, http.StatusOK, browseDatasetsResponse{
		Root:     h.config.DatasetsRoot,
		Count:    len(manifest.Datasets),
		Datasets: manifest.Datasets,
	})
}
