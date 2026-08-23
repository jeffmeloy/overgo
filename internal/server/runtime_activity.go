package server

import (
	"net/http"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

type runtimeSessionsResponse struct {
	Session capabilityruntime.SessionSnapshot `json:"session"`
	Slots   []slotStatusItem                  `json:"slots"`
}

type servingActivity struct {
	ID artifact.ID `json:"id"`
	runrecord.ServingObservation
}

type runtimeActivityResponse struct {
	Count       int               `json:"count"`
	Truncated   bool              `json:"truncated"`
	PublishFail uint64            `json:"publish_failures"`
	Activity    []servingActivity `json:"activity"`
}

func (h *Handler) runtimeSessions(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	writeJSON(response, http.StatusOK, runtimeSessionsResponse{
		Session: h.sessions.Snapshot(), Slots: h.sessionStatus(false),
	})
}

func (h *Handler) runtimeActivity(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	store, release, ok := h.openBrowseStore(response)
	if !ok {
		return
	}
	defer release()
	result, err := store.Query(request.Context(), repodb.Query{
		Kind:       artifact.KindEvidence,
		MediaType:  runrecord.ServingObservationMediaType,
		Schema:     runrecord.ServingObservationSchema,
		MaxResults: repodb.MaxQueryResults,
	})
	if err != nil {
		writeError(response, http.StatusInternalServerError, "repodb_error", err.Error())
		return
	}
	activity := make([]servingActivity, 0, len(result.Artifacts))
	for _, descriptor := range result.Artifacts {
		content, found, readErr := store.Content(request.Context(), descriptor.ID)
		if readErr != nil || !found {
			continue
		}
		observation, parseErr := runrecord.ParseServingObservation(content.Data)
		if parseErr == nil {
			activity = append(activity, servingActivity{ID: descriptor.ID, ServingObservation: observation})
		}
	}
	sort.Slice(activity, func(left, right int) bool {
		return activity[left].StartedUnixNS > activity[right].StartedUnixNS
	})
	writeJSON(response, http.StatusOK, runtimeActivityResponse{
		Count: len(activity), Truncated: result.Truncated,
		PublishFail: h.observationErrors.Load(), Activity: activity,
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
