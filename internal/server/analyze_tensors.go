package server

import (
	"errors"
	"math"
	"net/http"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/repodb"
	"overgo/internal/tensorstats"
)

type AnalysisPolicy struct {
	TensorSamples   uint64  `json:"tensor_samples"`
	TensorReadBytes uint64  `json:"tensor_read_bytes"`
	StatePositions  int     `json:"state_positions"`
	MDSIterations   int     `json:"mds_iterations"`
	MDSTolerance    float64 `json:"mds_tolerance"`
}

func (policy AnalysisPolicy) validate() error {
	if policy.TensorSamples == 0 || policy.TensorReadBytes == 0 || policy.StatePositions <= 0 ||
		policy.MDSIterations <= 0 || policy.MDSTolerance <= 0 || math.IsNaN(policy.MDSTolerance) || math.IsInf(policy.MDSTolerance, 0) {
		return errors.New("server: invalid analysis policy")
	}
	return nil
}

func (policy AnalysisPolicy) tensorMeasurementPolicy() modelartifact.MeasurementPolicy {
	return modelartifact.MeasurementPolicy{
		MaxSamplesPerTensor: policy.TensorSamples,
		MaxReadBytes:        policy.TensorReadBytes,
		SpectralMaxDim:      uint64(math.Sqrt(float64(policy.TensorReadBytes / binaryschema.Uint64Bytes))),
	}
}

// analyzeTensor is one tensor's storage identity plus its persisted measurement
// (name, characterization, and effective rank flatten into the JSON object).
// Model labels the owning model when the entry comes from the cross-model
// store catalog.
type analyzeTensor struct {
	Model   string   `json:"model,omitempty"`
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
		Policy:  h.config.Analysis.tensorMeasurementPolicy(),
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
// nearest-neighbor over scale-free descriptors. Pool names the retrieval
// source: "store-catalog" spans every measured model in RepoDB;
// "loaded-model" is the single-model fallback when no catalog is committed.
type analyzeTensorsSimilarResponse struct {
	Target    analyzeTensor           `json:"target"`
	Metric    string                  `json:"metric"`
	Pool      string                  `json:"pool"`
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
	k := 0
	if raw := request.URL.Query().Get("k"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 {
			writeError(response, http.StatusBadRequest, "invalid_request", "k must be a positive integer")
			return
		}
		k = parsed
	}
	profiles, source := h.storeComponentPool(request)
	if len(profiles) == 0 {
		var ok bool
		profiles, ok = h.characterizeLoadedModel(response)
		if !ok {
			return
		}
		source = "loaded-model"
	}
	targetIndex := -1
	targetModel := request.URL.Query().Get("model")
	pool := make([]tensorstats.Characterization, len(profiles))
	for i, profile := range profiles {
		pool[i] = profile.Characterization
		if profile.Name == name && (targetModel == "" || profile.Model == targetModel) &&
			(targetIndex < 0 || profile.Model == h.config.ModelID) {
			targetIndex = i
		}
	}
	if targetIndex < 0 {
		writeError(response, http.StatusNotFound, "not_found", "tensor "+name+" is not in the "+source+" pool")
		return
	}
	if k == 0 {
		k = defaultNeighborCount(len(profiles))
	}
	k = min(k, len(profiles)-1)
	nearest := tensorstats.Nearest(pool, targetIndex, k)
	neighbors := make([]tensorSimilarNeighbor, len(nearest))
	for i, n := range nearest {
		neighbors[i] = tensorSimilarNeighbor{Distance: n.Distance, analyzeTensor: profiles[n.Index]}
	}
	writeJSON(response, http.StatusOK, analyzeTensorsSimilarResponse{
		Target:    profiles[targetIndex],
		Metric:    "rank-footrule",
		Pool:      source,
		Neighbors: neighbors,
	})
}

// storeComponentPool reads the retained cross-model tensor catalog.
func (h *Handler) storeComponentPool(request *http.Request) ([]analyzeTensor, string) {
	store, err := h.browseStore(request.Context())
	if err != nil {
		return nil, ""
	}
	ctx := request.Context()
	var profiles []analyzeTensor
	_, err = repodb.VisitDecodedDocuments(ctx, store, repodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindTensorInventory, MediaType: modelartifact.TensorMeasurementMediaType,
			Schema: modelartifact.TensorMeasurementSchema,
		}}, Order: repodb.DocumentOldestFirst,
	}, modelartifact.ParseTensorMeasurementDocument, func(_ repodb.DocumentView, document modelartifact.TensorMeasurementDocument) error {
		inventory, ok, err := modelartifact.ReadTensorInventoryDocument(ctx, store, document.Inventory)
		if err != nil || !ok {
			return nil
		}
		model := inventory.Model.String()
		for _, measurement := range document.Measurements {
			entry := analyzeTensor{Model: model, TensorMeasurement: measurement}
			if fact, ok := inventory.Tensor(measurement.Name); ok {
				entry.Storage, entry.Shape = fact.Storage, fact.Shape
			}
			profiles = append(profiles, entry)
		}
		return nil
	})
	if err != nil {
		return nil, ""
	}
	return profiles, "store-catalog"
}

// characterizeLoadedModel opens the served model and profiles every tensor via
// the shared measurement pipeline, or writes an HTTP error and returns false.
func (h *Handler) characterizeLoadedModel(response http.ResponseWriter) ([]analyzeTensor, bool) {
	if h.config.Analysis == (AnalysisPolicy{}) {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "tensor analysis policy is unavailable")
		return nil, false
	}
	api, ok := h.generator.(ModelPropertiesAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "model properties are unavailable")
		return nil, false
	}
	path := api.ModelProperties().Path
	if path == "" {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "model path is unavailable for tensor analysis")
		return nil, false
	}
	file, err := gguf.Open(path)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "model_read_failed", "open model: "+err.Error())
		return nil, false
	}
	defer file.Close()
	profiles, err := characterizeGGUF(file, h.config.Analysis.tensorMeasurementPolicy())
	if err != nil {
		writeError(response, http.StatusInternalServerError, "tensor_characterization_failed", err.Error())
		return nil, false
	}
	return profiles, true
}

// characterizeGGUF builds the tensor inventory and measures every tensor through
// the shared, identity-keyed measurement pipeline, pairing each characterization
// with its storage and shape facts.
func characterizeGGUF(file *gguf.File, policy modelartifact.MeasurementPolicy) ([]analyzeTensor, error) {
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		return nil, err
	}
	document, err := modelartifact.MeasureGGUF(inventory.TensorInventory, file, policy)
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
