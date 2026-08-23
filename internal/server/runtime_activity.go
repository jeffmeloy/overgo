package server

import (
	"net/http"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
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
	Count       int                `json:"count"`
	Truncated   bool               `json:"truncated"`
	PublishFail uint64             `json:"publish_failures"`
	Activity    []servingActivity  `json:"activity"`
	Operations  []operation.Status `json:"operations"`
}

func (h *Handler) runtimeSessions(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
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
	writeJSON(response, http.StatusOK, runtimeSessionsResponse{
		Session: h.sessions.Snapshot(), Authority: authority, Slots: h.sessionStatus(false),
	})
}

func (h *Handler) runtimeActivity(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	if h.repository == nil && h.config.RepoDBPath == "" {
		writeJSON(response, http.StatusOK, runtimeActivityResponse{Operations: h.operations.List()})
		return
	}
	store, ok := h.requireBrowseStore(response, request)
	if !ok {
		return
	}
	result, err := store.Query(request.Context(), repodb.Query{
		Kind:       artifact.KindEvidence,
		MediaType:  runrecord.ServingObservationMediaType,
		Schema:     runrecord.ServingObservationSchema,
		MaxResults: store.QueryExtent(),
		Projection: repodb.ProjectContentData,
	})
	if err != nil {
		writeError(response, http.StatusInternalServerError, "repodb_error", err.Error())
		return
	}
	activity := make([]servingActivity, 0, len(result.Contents))
	for _, content := range result.Contents {
		observation, parseErr := runrecord.ParseServingObservation(content.Data)
		if parseErr == nil {
			activity = append(activity, servingActivity{ID: content.Artifact, ServingObservation: observation})
		}
	}
	sort.Slice(activity, func(left, right int) bool {
		return activity[left].StartedUnixNS > activity[right].StartedUnixNS
	})
	writeJSON(response, http.StatusOK, runtimeActivityResponse{
		Count: len(activity), Truncated: result.Truncated,
		PublishFail: h.observationErrors.Load(), Activity: activity, Operations: h.operations.List(),
	})
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
	if stream.named("operation.snapshot", h.operations.List()) != nil {
		return
	}
	for {
		select {
		case <-request.Context().Done():
			return
		case event, open := <-events:
			if !open || stream.named("operation", event) != nil {
				return
			}
		}
	}
}
