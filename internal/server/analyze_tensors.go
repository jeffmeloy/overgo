package server

import (
	"net/http"
	"strconv"

	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/tensorstats"
)

// Bounds for the read-only tensor-statistics endpoints. Sampling is per-tensor
// evenly spaced and the total read is capped, so the calls are cheap and
// constant regardless of model size.
const (
	analyzeTensorMaxSamplesPerTensor = 4096
	analyzeTensorMaxReadBytes        = 64 << 20
	// analyzeTensorSpectralMaxDim bounds inline effective-rank computation: only
	// small 2-D matrices are analyzed per request (larger ones report deferred);
	// the offline producer handles full coverage. A compute budget, not a metric
	// threshold.
	analyzeTensorSpectralMaxDim = 512

	analyzeTensorSimilarDefaultK = 8
	analyzeTensorSimilarMaxK     = 64
)

// analyzeTensor is one tensor's storage identity plus its persisted measurement
// (name, characterization, and effective rank flatten into the JSON object).
type analyzeTensor struct {
	Storage string   `json:"storage"`
	Shape   []uint64 `json:"shape"`
	modelartifact.TensorMeasurement
}

// analyzeTensorsResponse returns a distribution-free value profile for every
// tensor in the loaded GGUF model. Each profile reports robust L-moments and
// energy descriptors of a bounded sample — measured functionals of the
// empirical distribution, never a fitted shape (see the workbench's
// distribution-free analysis principle).
type analyzeTensorsResponse struct {
	Model   string                          `json:"model"`
	Policy  modelartifact.MeasurementPolicy `json:"policy"`
	Count   int                             `json:"count"`
	Tensors []analyzeTensor                 `json:"tensors"`
}

// analyzeTensorMeasurementPolicy is the single measurement policy for the
// analysis endpoints: bounded sampling plus small-matrix effective rank.
var analyzeTensorMeasurementPolicy = modelartifact.MeasurementPolicy{
	MaxSamplesPerTensor: analyzeTensorMaxSamplesPerTensor,
	MaxReadBytes:        analyzeTensorMaxReadBytes,
	SpectralMaxDim:      analyzeTensorSpectralMaxDim,
}

func (h *Handler) analyzeTensors(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	profiles, ok := h.characterizeLoadedModel(response)
	if !ok {
		return
	}
	writeJSON(response, http.StatusOK, analyzeTensorsResponse{
		Model:   h.config.ModelID,
		Policy:  analyzeTensorMeasurementPolicy,
		Count:   len(profiles),
		Tensors: profiles,
	})
}

// tensorSimilarNeighbor is one nearby tensor and its shape-feature distance.
type tensorSimilarNeighbor struct {
	Distance float64 `json:"distance"`
	analyzeTensor
}

// analyzeTensorsSimilarResponse returns the tensors whose value distributions
// are closest in shape to a named tensor, by exact distribution-free
// nearest-neighbor over scale-free descriptors.
type analyzeTensorsSimilarResponse struct {
	Target    analyzeTensor           `json:"target"`
	Metric    string                  `json:"metric"`
	Neighbors []tensorSimilarNeighbor `json:"neighbors"`
}

func (h *Handler) analyzeTensorsSimilar(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	name := request.URL.Query().Get("name")
	if name == "" {
		writeError(response, http.StatusBadRequest, "invalid_request", "name query parameter is required")
		return
	}
	k := analyzeTensorSimilarDefaultK
	if raw := request.URL.Query().Get("k"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 {
			writeError(response, http.StatusBadRequest, "invalid_request", "k must be a positive integer")
			return
		}
		k = min(parsed, analyzeTensorSimilarMaxK)
	}
	profiles, ok := h.characterizeLoadedModel(response)
	if !ok {
		return
	}
	targetIndex := -1
	pool := make([]tensorstats.Characterization, len(profiles))
	for i, profile := range profiles {
		pool[i] = profile.Characterization
		if profile.Name == name {
			targetIndex = i
		}
	}
	if targetIndex < 0 {
		writeError(response, http.StatusNotFound, "not_found", "tensor "+name+" is not in the model")
		return
	}
	nearest := tensorstats.Nearest(pool, targetIndex, k)
	neighbors := make([]tensorSimilarNeighbor, len(nearest))
	for i, n := range nearest {
		neighbors[i] = tensorSimilarNeighbor{Distance: n.Distance, analyzeTensor: profiles[n.Index]}
	}
	writeJSON(response, http.StatusOK, analyzeTensorsSimilarResponse{
		Target:    profiles[targetIndex],
		Metric:    "rank-footrule",
		Neighbors: neighbors,
	})
}

// characterizeLoadedModel opens the served model and profiles every tensor via
// the shared measurement pipeline, or writes an HTTP error and returns false.
func (h *Handler) characterizeLoadedModel(response http.ResponseWriter) ([]analyzeTensor, bool) {
	api, ok := h.generator.(ModelPropertiesAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "model properties are unavailable")
		return nil, false
	}
	path := api.ModelProperties().Path
	if path == "" {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "model path is unavailable for tensor analysis")
		return nil, false
	}
	file, err := gguf.Open(path)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "model_read_failed", "open model: "+err.Error())
		return nil, false
	}
	defer file.Close()
	profiles, err := characterizeGGUF(file)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "tensor_characterization_failed", err.Error())
		return nil, false
	}
	return profiles, true
}

// characterizeGGUF builds the tensor inventory and measures every tensor through
// the shared, identity-keyed measurement pipeline, pairing each characterization
// with its storage and shape facts.
func characterizeGGUF(file *gguf.File) ([]analyzeTensor, error) {
	inventory, err := modelartifact.FromGGUF(file)
	if err != nil {
		return nil, err
	}
	document, err := modelartifact.MeasureGGUF(inventory.TensorInventory, file, analyzeTensorMeasurementPolicy)
	if err != nil {
		return nil, err
	}
	profiles := make([]analyzeTensor, len(document.Measurements))
	for i, measurement := range document.Measurements {
		fact, _ := inventory.TensorInventory.Tensor(measurement.Name)
		profiles[i] = analyzeTensor{
			Storage:           fact.Storage,
			Shape:             fact.Shape,
			TensorMeasurement: measurement,
		}
	}
	return profiles, nil
}
