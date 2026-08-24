package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

type runtimeSessionsResponse struct {
	Session   capabilityruntime.SessionSnapshot `json:"session"`
	Authority *runtimeAuthority                 `json:"authority,omitempty"`
	Slots     []slotStatusItem                  `json:"slots"`
}

type runtimeAuthority struct {
	Model     artifact.ID            `json:"model"`
	Recipe    artifact.ID            `json:"recipe"`
	Task      recipe.Task            `json:"task"`
	Runtime   modelrecipe.Runtime    `json:"runtime,omitempty"`
	Placement recipe.Placement       `json:"placement,omitempty"`
	Residency recipe.ResidencyPolicy `json:"residency,omitempty"`
	Stages    int                    `json:"stages"`
	Evidence  []artifact.ID          `json:"evidence,omitempty"`
}

type servingActivity struct {
	ID artifact.ID `json:"id"`
	runrecord.ServingObservation
}

type runtimeActivityResponse struct {
	Count        int                `json:"count"`
	Truncated    bool               `json:"truncated"`
	PublishFail  uint64             `json:"publish_failures"`
	Activity     []servingActivity  `json:"activity"`
	Operations   []operation.Status `json:"operations"`
	Stages       []json.RawMessage  `json:"stages,omitempty"`
	Decisions    []json.RawMessage  `json:"decisions,omitempty"`
	Interactions []json.RawMessage  `json:"interactions,omitempty"`
}

func (h *Handler) runtimeSessions(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	writeJSON(response, http.StatusOK, h.runtimeSessionsSnapshot())
}

func (h *Handler) runtimeSessionsSnapshot() runtimeSessionsResponse {
	var authority *runtimeAuthority
	if inspector, ok := h.generator.(interface {
		RecipeRuntimeDescription(recipe.Task) (modelrecipe.RuntimeDescription, error)
	}); ok {
		if description, err := inspector.RecipeRuntimeDescription(recipe.TaskInference); err == nil &&
			description.Identity.Model.Kind() == artifact.KindModel && description.Identity.Recipe.Kind() == artifact.KindRecipe {
			authority = &runtimeAuthority{
				Model: description.Identity.Model, Recipe: description.Identity.Recipe, Task: description.Task,
				Runtime: description.Identity.Runtime, Placement: description.Identity.Placement,
				Residency: description.Identity.Residency, Stages: len(description.Stages), Evidence: description.Evidence,
			}
		}
	}
	return runtimeSessionsResponse{
		Session: h.sessions.Snapshot(), Authority: authority, Slots: h.sessionStatus(false),
	}
}

func (h *Handler) runtimeActivity(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	activity, err := h.runtimeActivitySnapshot(request.Context())
	if errors.Is(err, errBrowseRepositoryUnavailable) {
		writeJSON(response, http.StatusOK, runtimeActivityResponse{Operations: h.operations.List()})
		return
	}
	if err != nil {
		writeError(response, http.StatusInternalServerError, "repodb_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, activity)
}

func (h *Handler) runtimeActivitySnapshot(ctx context.Context) (runtimeActivityResponse, error) {
	store, err := h.browseStore(ctx)
	if err != nil {
		return runtimeActivityResponse{}, err
	}
	activity := make([]servingActivity, 0, h.config.MaxStoredResponses)
	page, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
		}},
		AliasPrefixes: []string{runrecord.ServingAttemptAliasRoot},
		Order:         overgodb.DocumentNewestFirst, MaxResults: h.config.MaxStoredResponses,
	}, runrecord.ParseServingObservation, func(view overgodb.DocumentView, observation runrecord.ServingObservation) error {
		activity = append(activity, servingActivity{ID: view.Content.Descriptor.ID, ServingObservation: observation})
		return nil
	})
	if err != nil {
		return runtimeActivityResponse{}, err
	}
	stages, stagesTruncated, err := projectedDocuments(ctx, store,
		runrecord.StageReceiptMediaType, runrecord.StageReceiptSchema, runrecord.StageReceiptAliasRoot, h.config.MaxStoredResponses)
	if err != nil {
		return runtimeActivityResponse{}, err
	}
	decisions, decisionsTruncated, err := projectedDocuments(ctx, store,
		runrecord.HumanDecisionMediaType, runrecord.HumanDecisionSchema, runrecord.HumanDecisionAliasRoot, h.config.MaxStoredResponses)
	if err != nil {
		return runtimeActivityResponse{}, err
	}
	interactions, interactionsTruncated, err := projectedDocuments(ctx, store,
		runrecord.InteractionMediaType, runrecord.InteractionSchema, runrecord.InteractionResponseAliasRoot, h.config.MaxStoredResponses)
	return runtimeActivityResponse{
		Count: len(activity), PublishFail: h.observationErrors.Load(), Activity: activity, Operations: h.operations.List(),
		Stages: stages, Decisions: decisions, Interactions: interactions,
		Truncated: page.Truncated || stagesTruncated || decisionsTruncated || interactionsTruncated,
	}, err
}

func projectedDocuments(
	ctx context.Context,
	store *overgodb.Store,
	mediaType, schema, aliasPrefix string,
	limit int,
) ([]json.RawMessage, bool, error) {
	documents := make([]json.RawMessage, 0, limit)
	page, err := store.VisitDocuments(ctx, overgodb.DocumentQuery{
		Contracts:     []artifact.DocumentContract{{Kind: artifact.KindEvidence, MediaType: mediaType, Schema: schema}},
		AliasPrefixes: []string{aliasPrefix}, Order: overgodb.DocumentNewestFirst, MaxResults: limit,
	}, func(view overgodb.DocumentView) error {
		documents = append(documents, json.RawMessage(view.Content.Data))
		return nil
	})
	return documents, page.Truncated, err
}

func (h *Handler) runtimeActivityStream(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	events, unsubscribe, err := h.operations.Subscribe()
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	defer unsubscribe()
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSSEEmitter(request.Context(), response, flusher)
	if stream.named("runtime.sessions", h.runtimeSessionsSnapshot()) != nil {
		return
	}
	if activity, snapshotErr := h.runtimeActivitySnapshot(request.Context()); snapshotErr == nil {
		if stream.named("runtime.activity", activity) != nil {
			return
		}
	} else if errors.Is(snapshotErr, errBrowseRepositoryUnavailable) {
		if stream.named("runtime.activity", runtimeActivityResponse{Operations: h.operations.List()}) != nil {
			return
		}
	} else {
		return
	}
	if stream.named("operation.snapshot", h.operations.List()) != nil {
		return
	}
	for {
		select {
		case <-request.Context().Done():
			return
		case event, open := <-events:
			if !open || stream.named("operation", event) != nil ||
				stream.named("runtime.sessions", h.runtimeSessionsSnapshot()) != nil {
				return
			}
		}
	}
}
