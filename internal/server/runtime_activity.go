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
	Cursor       uint64             `json:"cursor,string"`
	Limit        int                `json:"limit"`
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
	activity, err := h.runtimeActivitySnapshot(request.Context())
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, activity)
}

func (h *Handler) runtimeActivitySnapshot(ctx context.Context) (runtimeActivityResponse, error) {
	h.servingEvents.mu.Lock()
	defer h.servingEvents.mu.Unlock()
	store, err := h.browseStore(ctx)
	if errors.Is(err, errBrowseRepositoryUnavailable) {
		return runtimeActivityResponse{
			Cursor: h.servingEvents.cursor, Limit: h.config.MaxStoredResponses,
			Activity: []servingActivity{}, Operations: h.operations.List(),
		}, nil
	}
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
		Cursor: h.servingEvents.cursor, Limit: h.config.MaxStoredResponses,
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
	events, unsubscribe, err := h.operations.Subscribe()
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	defer unsubscribe()
	serving, unsubscribeServing := h.servingEvents.subscribe(h.config.MaxStoredResponses)
	defer unsubscribeServing()
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSSEEmitter(request.Context(), response, flusher)
	if stream.named("runtime.sessions", h.runtimeSessionsSnapshot()) != nil {
		return
	}
	activity, err := h.runtimeActivitySnapshot(request.Context())
	if err != nil || stream.named("runtime.activity", activity) != nil {
		return
	}
	cursor := activity.Cursor
	if stream.named("operation.snapshot", h.operations.List()) != nil {
		return
	}
	for {
		select {
		case <-request.Context().Done():
			return
		case event, open := <-serving:
			if !open {
				return
			}
			if event.Cursor <= cursor {
				continue
			}
			if event.Activity == nil {
				activity, err := h.runtimeActivitySnapshot(request.Context())
				if err != nil || stream.named("runtime.activity", activity) != nil {
					return
				}
				cursor = activity.Cursor
			} else {
				if stream.named("runtime.serving", event) != nil {
					return
				}
				cursor = event.Cursor
			}
		case event, open := <-events:
			if !open || stream.named("operation", event) != nil ||
				stream.named("runtime.sessions", h.runtimeSessionsSnapshot()) != nil {
				return
			}
		}
	}
}
