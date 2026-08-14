package server

import (
	"net/http"
	"strconv"

	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/tensorstats"
)

// Neighbor-count bounds for /analyze/tensors/similar.
const (
	analyzeTensorSimilarDefaultK = 8
	analyzeTensorSimilarMaxK     = 64
)

// Bounds for the read-only tensor-statistics endpoint. Sampling is per-tensor
// evenly spaced and the total read is capped, so the call is cheap and constant
// regardless of model size.
const (
	analyzeTensorMaxSamplesPerTensor = 4096
	analyzeTensorMaxReadBytes        = 64 << 20
)

// analyzeTensorsResponse returns a distribution-free value profile for every
// tensor in the loaded GGUF model. Each profile reports robust L-moments and
// energy descriptors of a bounded sample — measured functionals of the
// empirical distribution, never a fitted shape (see the workbench's
// distribution-free analysis principle).
type analyzeTensorsResponse struct {
	Model   string                                 `json:"model"`
	Policy  analyzeTensorPolicy                    `json:"policy"`
	Count   int                                    `json:"count"`
	Tensors []modelartifact.TensorCharacterization `json:"tensors"`
}

type analyzeTensorPolicy struct {
	MaxSamplesPerTensor uint64 `json:"max_samples_per_tensor"`
	MaxReadBytes        uint64 `json:"max_read_bytes"`
}

func (h *Handler) analyzeTensors(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	api, ok := h.generator.(ModelPropertiesAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "model properties are unavailable")
		return
	}
	path := api.ModelProperties().Path
	if path == "" {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "model path is unavailable for tensor analysis")
		return
	}
	file, err := gguf.Open(path)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "model_read_failed", "open model: "+err.Error())
		return
	}
	defer file.Close()
	policy := modelartifact.MeasurementPolicy{
		MaxSamplesPerTensor: analyzeTensorMaxSamplesPerTensor,
		MaxReadBytes:        analyzeTensorMaxReadBytes,
	}
	profiles, err := modelartifact.CharacterizeGGUFTensors(file, policy)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "tensor_characterization_failed", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, analyzeTensorsResponse{
		Model:   h.config.ModelID,
		Policy:  analyzeTensorPolicy{policy.MaxSamplesPerTensor, policy.MaxReadBytes},
		Count:   len(profiles),
		Tensors: profiles,
	})
}

// tensorSimilarNeighbor is one nearby tensor and its shape-feature distance.
type tensorSimilarNeighbor struct {
	Distance float64 `json:"distance"`
	modelartifact.TensorCharacterization
}

// analyzeTensorsSimilarResponse returns the tensors whose value distributions
// are closest in shape to a named tensor, by exact distribution-free
// nearest-neighbor over scale-free descriptors.
type analyzeTensorsSimilarResponse struct {
	Target    modelartifact.TensorCharacterization `json:"target"`
	Neighbors []tensorSimilarNeighbor              `json:"neighbors"`
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
	api, ok := h.generator.(ModelPropertiesAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "model properties are unavailable")
		return
	}
	path := api.ModelProperties().Path
	if path == "" {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "model path is unavailable for tensor analysis")
		return
	}
	file, err := gguf.Open(path)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "model_read_failed", "open model: "+err.Error())
		return
	}
	defer file.Close()
	profiles, err := modelartifact.CharacterizeGGUFTensors(file, modelartifact.MeasurementPolicy{
		MaxSamplesPerTensor: analyzeTensorMaxSamplesPerTensor,
		MaxReadBytes:        analyzeTensorMaxReadBytes,
	})
	if err != nil {
		writeError(response, http.StatusInternalServerError, "tensor_characterization_failed", err.Error())
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
	nearest := tensorstats.Nearest(pool[targetIndex], pool, k, targetIndex)
	neighbors := make([]tensorSimilarNeighbor, len(nearest))
	for i, n := range nearest {
		neighbors[i] = tensorSimilarNeighbor{Distance: n.Distance, TensorCharacterization: profiles[n.Index]}
	}
	writeJSON(response, http.StatusOK, analyzeTensorsSimilarResponse{
		Target:    profiles[targetIndex],
		Neighbors: neighbors,
	})
}
