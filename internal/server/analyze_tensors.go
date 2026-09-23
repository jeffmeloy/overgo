package server

import (
	"context"
	"errors"
	"math"
	"net/http"
	"strconv"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
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
	Model   string   `json:"model,omitzero"`
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
	profiles, ok := h.characterizeLoadedModel(request.Context(), response)
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
// source: "store-catalog" spans every measured model in OvergoDB;
// "loaded-model" is the single-model fallback when no catalog is committed.
type analyzeTensorsSimilarResponse struct {
	Target    analyzeTensor           `json:"target"`
	Metric    string                  `json:"metric"`
	Pool      string                  `json:"pool"`
	Neighbors []tensorSimilarNeighbor `json:"neighbors"`
}

func (h *Handler) analyzeTensorsSimilar(response http.ResponseWriter, request *http.Request) {
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
		profiles, ok = h.characterizeLoadedModel(request.Context(), response)
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
	err = visitTensorMeasurements(ctx, store, overgodb.DocumentOldestFirst, func(document modelartifact.TensorMeasurementDocument) error {
		inventory, ok, err := modelartifact.ReadTensorInventoryDocument(ctx, store, document.Inventory)
		if err != nil || !ok {
			return nil
		}
		model := inventory.Owner.String()
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
func (h *Handler) characterizeLoadedModel(ctx context.Context, response http.ResponseWriter) ([]analyzeTensor, bool) {
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
	profiles, err := h.characterizeGGUF(ctx, file, h.config.Analysis.tensorMeasurementPolicy())
	if err != nil {
		writeError(response, http.StatusInternalServerError, "tensor_characterization_failed", err.Error())
		return nil, false
	}
	return profiles, true
}

// characterizeGGUF builds the tensor inventory and pairs each measurement
// with its storage and shape facts. A full measuring pass reads every tensor
// (minutes on a served model), so it runs once per inventory and policy: the
// handler keeps the document, and a store keeps it across restarts.
func (h *Handler) characterizeGGUF(ctx context.Context, file *gguf.File, policy modelartifact.MeasurementPolicy) ([]analyzeTensor, error) {
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		return nil, err
	}
	document, err := h.tensorMeasurement(ctx, inventory.TensorInventory, file, policy)
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

// measureGGUF is the full measuring pass; a test counts the passes through it.
var measureGGUF = modelartifact.MeasureGGUF

// tensorMeasurement answers an inventory's measurement under policy from the
// handler, then from a committed document, and only then measures the file,
// committing the result when the server holds a writable repository.
func (h *Handler) tensorMeasurement(ctx context.Context, inventory modelartifact.TensorInventoryDocument, file *gguf.File, policy modelartifact.MeasurementPolicy) (modelartifact.TensorMeasurementDocument, error) {
	if held, ok := h.tensorMeasurements.Load(inventory.ID); ok {
		return held.(modelartifact.TensorMeasurementDocument), nil
	}
	store, storeErr := h.browseStore(ctx)
	var document modelartifact.TensorMeasurementDocument
	found := false
	if storeErr == nil {
		err := visitTensorMeasurements(ctx, store, overgodb.DocumentNewestFirst, func(stored modelartifact.TensorMeasurementDocument) error {
			if !found && stored.Inventory == inventory.ID && stored.Policy == policy {
				document, found = stored, true
			}
			return nil
		})
		if err != nil {
			return modelartifact.TensorMeasurementDocument{}, err
		}
	}
	if !found {
		measured, err := measureGGUF(inventory, file, policy)
		if err != nil {
			return modelartifact.TensorMeasurementDocument{}, err
		}
		document = measured
		// Only the serving repository is writable; a browse-only server keeps the measurement in memory.
		if h.repository != nil {
			if err := publishTensorMeasurement(ctx, h.repository, inventory, document); err != nil {
				return modelartifact.TensorMeasurementDocument{}, err
			}
		}
	}
	h.tensorMeasurements.Store(inventory.ID, document)
	return document, nil
}

// visitTensorMeasurements visits every committed tensor measurement in order.
func visitTensorMeasurements(ctx context.Context, store *overgodb.Store, order overgodb.DocumentOrder, visit func(modelartifact.TensorMeasurementDocument) error) error {
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindTensorInventory, MediaType: modelartifact.TensorMeasurementMediaType,
			Schema: modelartifact.TensorMeasurementSchema,
		}}, Order: order,
	}, modelartifact.ParseTensorMeasurementDocument, func(_ overgodb.DocumentView, document modelartifact.TensorMeasurementDocument) error {
		return visit(document)
	})
	return err
}

// publishTensorMeasurement commits the measurement with the inventory it names.
func publishTensorMeasurement(ctx context.Context, store *overgodb.Store, inventory modelartifact.TensorInventoryDocument, document modelartifact.TensorMeasurementDocument) error {
	batch, err := document.Batch("tensor-measurement/" + document.ID.String())
	if err != nil {
		return err
	}
	inventoryContent, err := inventory.Content()
	if err != nil {
		return err
	}
	batch.Artifacts = append(batch.Artifacts, inventoryContent.Descriptor)
	batch.Contents = append(batch.Contents, inventoryContent)
	_, err = artifact.Publish(ctx, store, batch)
	return err
}
