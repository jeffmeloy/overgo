package server

import (
	"context"
	"net/http"
	"strings"
)

// LoopControl reads and stops the unattended loop of the checkout the server
// runs from.
type LoopControl interface {
	LoopStatus(context.Context) (LoopStatus, error)
	StopLoop(ctx context.Context, detail string) error
}

// LoopStatus is the loop as the workbench shows it: whether a live
// supervisor runs it, whether a stop holds it and why, the row it works
// next, and its last landing.
type LoopStatus struct {
	Running    bool        `json:"running"`
	Stopped    bool        `json:"stopped"`
	StopReason string      `json:"stop_reason,omitzero"`
	Complete   bool        `json:"complete"`
	Row        string      `json:"row,omitzero"`
	RowTitle   string      `json:"row_title,omitzero"`
	Landing    LoopLanding `json:"landing"`
}

// LoopLanding is the loop's last landing.
type LoopLanding struct {
	Commit  string `json:"commit,omitzero"`
	Row     string `json:"row,omitzero"`
	Subject string `json:"subject,omitzero"`
	At      string `json:"at,omitzero"`
}

// loopStopDetail is the reason a stop from the workbench records.
const loopStopDetail = "stopped from the workbench"

// loopStatus answers GET /loop/status with the loop's status.
func (h *Handler) loopStatus(response http.ResponseWriter, request *http.Request) {
	if h.config.Loop == nil {
		writeError(response, http.StatusNotFound, "not_found", "the server runs from no checkout with a loop")
		return
	}
	status, err := h.config.Loop.LoopStatus(request.Context())
	if err != nil {
		writeError(response, http.StatusInternalServerError, "loop_unreadable", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, status)
}

// loopStop answers POST /loop/stop by recording a user stop the loop
// honours, then the loop's status.
func (h *Handler) loopStop(response http.ResponseWriter, request *http.Request) {
	if h.config.Loop == nil {
		writeError(response, http.StatusNotFound, "not_found", "the server runs from no checkout with a loop")
		return
	}
	if err := h.config.Loop.StopLoop(request.Context(), loopStopDetail); err != nil {
		writeError(response, http.StatusConflict, "loop_not_stopped", strings.TrimSpace(err.Error()))
		return
	}
	h.loopStatus(response, request)
}
