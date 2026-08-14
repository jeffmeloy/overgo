package server

import (
	"net/http"

	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
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
