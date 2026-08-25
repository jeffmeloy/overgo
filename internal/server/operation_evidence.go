package server

import (
	"context"
	"errors"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

var errOperationEvidenceNotFound = errors.New("server: operation evidence not found")

// OperationEvidenceDocument preserves immutable evidence identity in a projection.
type OperationEvidenceDocument[T any] struct {
	ID    artifact.ID `json:"id"`
	Value T           `json:"value"`
}

// OperationStageSummary counts the current durable lifecycle state of every stage.
type OperationStageSummary struct {
	Admitted  int `json:"admitted"`
	Running   int `json:"running"`
	Waiting   int `json:"waiting"`
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
}

// OperationEvidenceSummary is a payload-free aggregate of one operation's evidence.
type OperationEvidenceSummary struct {
	State             operation.State            `json:"state,omitempty"`
	Stages            OperationStageSummary      `json:"stages"`
	StageAttempts     uint64                     `json:"stage_attempts"`
	OperationAttempts int                        `json:"operation_attempts"`
	ServingAttempts   int                        `json:"serving_attempts"`
	Interactions      int                        `json:"interactions"`
	Decisions         int                        `json:"decisions"`
	Usage             runrecord.ServingUsage     `json:"usage"`
	Resources         runrecord.ServingResources `json:"resources"`
	TotalMeasuredNS   uint64                     `json:"total_measured_ns"`
}

// OperationEvidenceProjection joins bounded immutable facts without transcript payloads.
type OperationEvidenceProjection struct {
	ID           artifact.ID                                               `json:"id"`
	Operation    *operation.Status                                         `json:"operation,omitempty"`
	Summary      OperationEvidenceSummary                                  `json:"summary"`
	Stages       []OperationEvidenceDocument[runrecord.StageReceipt]       `json:"stages"`
	Serving      []OperationEvidenceDocument[runrecord.ServingObservation] `json:"serving"`
	Interactions []OperationEvidenceDocument[runrecord.Interaction]        `json:"interactions"`
	Decisions    []OperationEvidenceDocument[runrecord.HumanDecision]      `json:"decisions"`
	Runtime      runtimeSessionsResponse                                   `json:"runtime"`
	Truncated    bool                                                      `json:"truncated"`
}

func (h *Handler) operationEvidence(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	id, err := artifact.ParseID(request.URL.Query().Get("id"))
	if err != nil || id.Kind() != artifact.KindEvidence {
		writeError(response, http.StatusBadRequest, "invalid_operation", "operation evidence identity is required")
		return
	}
	limit := parseIntDefault(request.URL.Query().Get("limit"), h.config.MaxStoredResponses)
	if limit <= 0 || limit > h.config.MaxStoredResponses {
		limit = h.config.MaxStoredResponses
	}
	projection, err := h.operationEvidenceSnapshot(request.Context(), id, limit)
	if errors.Is(err, errBrowseRepositoryUnavailable) {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "operation evidence repository is not configured")
		return
	}
	if errors.Is(err, errOperationEvidenceNotFound) {
		writeError(response, http.StatusNotFound, "operation_not_found", err.Error())
		return
	}
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, projection)
}

func (h *Handler) operationEvidenceSnapshot(
	ctx context.Context,
	id artifact.ID,
	limit int,
) (OperationEvidenceProjection, error) {
	store, err := h.browseStore(ctx)
	if err != nil {
		return OperationEvidenceProjection{}, err
	}
	stages, stagePage, err := collectOperationEvidence(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.StageReceiptMediaType, Schema: runrecord.StageReceiptSchema,
	}, runrecord.StageReceiptAliasRoot+id.String()+"/", limit, runrecord.ParseStageReceipt)
	if err != nil {
		return OperationEvidenceProjection{}, err
	}
	serving, servingPage, err := collectOperationEvidence(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
	}, runrecord.ServingAttemptAliasRoot+id.String()+"/", limit, runrecord.ParseServingObservation)
	if err != nil {
		return OperationEvidenceProjection{}, err
	}
	interactions, interactionPage, err := collectOperationEvidence(ctx, store, artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: runrecord.InteractionMediaType, Schema: runrecord.InteractionSchema,
	}, runrecord.InteractionOperationAliasRoot+id.String()+"/", limit, runrecord.ParseInteraction)
	if err != nil {
		return OperationEvidenceProjection{}, err
	}
	decisions, decisionsTruncated, err := operationDecisions(ctx, store, id, limit)
	if err != nil {
		return OperationEvidenceProjection{}, err
	}
	projection := OperationEvidenceProjection{
		ID: id, Stages: stages, Serving: serving, Interactions: interactions, Decisions: decisions,
		Runtime:   h.runtimeSessionsSnapshot(),
		Truncated: stagePage.Truncated || servingPage.Truncated || interactionPage.Truncated || decisionsTruncated,
	}
	if status, found := h.operations.Status(id); found {
		projection.Operation = &status
	}
	if projection.Operation == nil && len(stages) == 0 && len(serving) == 0 && len(interactions) == 0 && len(decisions) == 0 {
		return OperationEvidenceProjection{}, errOperationEvidenceNotFound
	}
	projection.Summary, err = summarizeOperationEvidence(projection)
	return projection, err
}

func collectOperationEvidence[T any](
	ctx context.Context,
	store overgodb.DocumentReader,
	contract artifact.DocumentContract,
	aliasPrefix string,
	limit int,
	decode func([]byte) (T, error),
) ([]OperationEvidenceDocument[T], overgodb.DocumentPage, error) {
	documents := make([]OperationEvidenceDocument[T], 0, limit)
	page, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{contract}, AliasPrefixes: []string{aliasPrefix},
		Order: overgodb.DocumentNewestFirst, MaxResults: limit,
	}, decode, func(view overgodb.DocumentView, value T) error {
		documents = append(documents, OperationEvidenceDocument[T]{ID: view.Content.Descriptor.ID, Value: value})
		return nil
	})
	return documents, page, err
}

func operationDecisions(
	ctx context.Context,
	reader artifact.Reader,
	operationID artifact.ID,
	limit int,
) ([]OperationEvidenceDocument[runrecord.HumanDecision], bool, error) {
	latest, found, err := runrecord.ResolveHumanDecision(ctx, reader, operationID)
	if err != nil || !found {
		return nil, false, err
	}
	decisions := make([]OperationEvidenceDocument[runrecord.HumanDecision], 0, limit)
	current := latest
	for len(decisions) < limit {
		decisions = append(decisions, OperationEvidenceDocument[runrecord.HumanDecision]{ID: current.ID, Value: current})
		if !current.Prior.Valid() {
			return decisions, false, nil
		}
		current, err = runrecord.RequireHumanDecision(ctx, reader, current.Prior)
		if err != nil {
			return nil, false, err
		}
	}
	return decisions, current.Prior.Valid(), nil
}

func summarizeOperationEvidence(projection OperationEvidenceProjection) (OperationEvidenceSummary, error) {
	var summary OperationEvidenceSummary
	if projection.Operation != nil {
		summary.State = projection.Operation.State
		summary.OperationAttempts = len(projection.Operation.Attempts)
	}
	for _, document := range projection.Stages {
		if err := addOperationEvidenceMetric(&summary.StageAttempts, uint64(document.Value.Attempt)); err != nil {
			return OperationEvidenceSummary{}, errors.New("server: operation stage summary overflow")
		}
		switch document.Value.State {
		case runrecord.StageAdmitted:
			summary.Stages.Admitted++
		case runrecord.StageRunning:
			summary.Stages.Running++
		case runrecord.StageWaiting:
			summary.Stages.Waiting++
		case runrecord.StageCompleted:
			summary.Stages.Completed++
		case runrecord.StageFailed:
			summary.Stages.Failed++
		}
	}
	summary.ServingAttempts = len(projection.Serving)
	summary.Interactions = len(projection.Interactions)
	summary.Decisions = len(projection.Decisions)
	for _, document := range projection.Serving {
		observation := document.Value
		if err := addOperationEvidenceMetric(&summary.Usage.InputTokens, observation.Usage.InputTokens); err != nil ||
			addOperationEvidenceMetric(&summary.Usage.OutputTokens, observation.Usage.OutputTokens) != nil ||
			addOperationEvidenceMetric(&summary.Usage.InputBytes, observation.Usage.InputBytes) != nil ||
			addOperationEvidenceMetric(&summary.Usage.OutputBytes, observation.Usage.OutputBytes) != nil ||
			addOperationEvidenceMetric(&summary.Resources.HostToDeviceBytes, observation.Resources.HostToDeviceBytes) != nil ||
			addOperationEvidenceMetric(&summary.Resources.DeviceToHostBytes, observation.Resources.DeviceToHostBytes) != nil ||
			addOperationEvidenceMetric(&summary.TotalMeasuredNS, observation.MeasuredNS) != nil {
			return OperationEvidenceSummary{}, errors.New("server: operation evidence summary overflow")
		}
		summary.Resources.PeakHostBytes = max(summary.Resources.PeakHostBytes, observation.Resources.PeakHostBytes)
		summary.Resources.PeakDeviceBytes = max(summary.Resources.PeakDeviceBytes, observation.Resources.PeakDeviceBytes)
	}
	return summary, nil
}

func addOperationEvidenceMetric(total *uint64, value uint64) error {
	next, ok := checked.Add64(*total, value)
	if !ok {
		return errors.New("server: operation evidence metric overflow")
	}
	*total = next
	return nil
}
