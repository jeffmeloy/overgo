package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"

	"overgo/internal/agentloop"
	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

// The agent workspace is protocol projection only: it decodes requests,
// calls the coordinator that owns admission, and encodes results. It
// holds no tool runtime and no session store of its own -- sessions are
// coordinator state keyed by id, and every durable fact lives in the
// interaction ledger the coordinator writes.

type agentSessions struct {
	mu       sync.Mutex
	sessions map[string]*agentloop.Session
}

// get returns the live session, restoring it from the durable
// interaction ledger on first sight so a restarted server neither
// forgets a session's step count and inspection state nor forges them.
// The session itself serializes its steps; this map only names it.
func (s *agentSessions) get(ctx context.Context, coordinator *agentloop.Coordinator, id string) (*agentloop.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sessions == nil {
		s.sessions = map[string]*agentloop.Session{}
	}
	if session, found := s.sessions[id]; found {
		return session, nil
	}
	session, err := coordinator.RestoreSession(ctx, id)
	if err != nil {
		return nil, err
	}
	s.sessions[id] = session
	return session, nil
}

// agentTools projects the registered tool catalog with its effect
// classes and publication coverage; the GUI reads effect to badge
// mutation tools and drives the approval control from it.
func (h *Handler) agentTools(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		methodNotAllowed(response)
		return
	}
	if h.repository == nil || h.agentCoordinator == nil {
		writeError(response, http.StatusServiceUnavailable, "agent_unavailable", "no agent runtime is configured")
		return
	}
	result, err := h.repository.Query(request.Context(), overgodb.Query{
		Kind: artifact.KindRecipe, MaxResults: h.config.MaxStoredResponses, Projection: overgodb.ProjectAliases,
	})
	if err != nil {
		writeError(response, http.StatusInternalServerError, "agent_error", err.Error())
		return
	}
	tools := make([]agentToolView, 0, len(result.Aliases))
	registered := 0
	for _, alias := range result.Aliases {
		name, ok := strings.CutPrefix(alias.Name, agenttool.RegisteredAliasPrefix)
		if !ok {
			continue
		}
		registered++
		manual, err := agenttool.ResolveRegisteredManual(request.Context(), h.repository, name)
		if err != nil {
			tools = append(tools, agentToolView{Name: name, Stale: err.Error()})
			continue
		}
		tools = append(tools, agentToolView{
			Name: manual.Name, Description: manual.Description, Effect: string(manual.Effect), Manual: manual.ID,
		})
	}
	writeJSON(response, http.StatusOK, map[string]any{
		"tools": tools, "coverage": map[string]any{"registered": registered, "published": len(tools)},
	})
}

// agentStep projects one coordinator step: decode the proposed tool,
// admit it under the coordinator's rules, and encode the durable
// result or the typed refusal.
func (h *Handler) agentStep(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		methodNotAllowed(response)
		return
	}
	if h.agentCoordinator == nil {
		writeError(response, http.StatusServiceUnavailable, "agent_unavailable", "no agent runtime is configured")
		return
	}
	var body agentStepRequest
	if err := strictjson.Decode(request.Body, &body); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if body.Session == "" || body.Tool == "" || request.URL.Path == "/agents/step" && body.Agent == "" {
		writeError(response, http.StatusBadRequest, "invalid_request", "session and tool are required")
		return
	}
	arguments := body.Arguments
	if len(arguments) == 0 {
		arguments = json.RawMessage(`{}`)
	}
	sessionID := body.Session
	var active runrecord.ActiveAgent
	var err error
	if body.Agent != "" {
		active, err = (runrecord.AgentAuthority{Repository: h.repository}).RequireActive(request.Context(), body.Agent)
		description, described := h.interactionDescription()
		if err != nil || !described || active.Definition.ModelRecipe != description.Identity.Recipe {
			writeError(response, http.StatusUnprocessableEntity, "agent_step_refused", "active agent model recipe differs from served runtime")
			return
		}
		sessionID = body.Agent + ":" + body.Session
	}
	session, err := h.agentSessions.get(request.Context(), h.agentCoordinator, sessionID)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "agent_error", err.Error())
		return
	}
	// An approved step first RECORDS the operator's grant as a durable
	// decision bound to the exact tool identity and argument bytes; the
	// proposal gate then verifies that committed decision -- the request
	// flag alone authorizes nothing.
	if body.Approve {
		if _, err := h.agentCoordinator.ApproveMutation(request.Context(), session, body.Tool, arguments); err != nil {
			writeError(response, http.StatusUnprocessableEntity, "agent_step_refused", err.Error())
			return
		}
	}
	var result json.RawMessage
	if body.Agent == "" {
		result, err = h.agentCoordinator.Propose(request.Context(), session, body.Tool, arguments, body.Approve)
	} else {
		result, err = h.agentCoordinator.ProposeWithManuals(
			request.Context(), session, body.Tool, arguments, body.Approve, active.Definition.ToolManuals,
		)
	}
	if err != nil {
		writeError(response, http.StatusUnprocessableEntity, "agent_step_refused", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{
		"session": session.ID, "steps": session.Steps, "inspected": session.Inspected,
		"interaction": idText(session.Interaction), "result": result,
	})
}

type agentToolView struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Effect      string      `json:"effect,omitempty"`
	Stale       string      `json:"stale,omitempty"`
	Manual      artifact.ID `json:"manual,omitzero"`
}

type agentStepRequest struct {
	Agent     string          `json:"agent,omitempty"`
	Session   string          `json:"session"`
	Tool      string          `json:"tool"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	Approve   bool            `json:"approve,omitempty"`
}

// buildAgentRuntime binds a coordinator over the served store when a
// serving identity is available; without one the agent routes report
// unavailability rather than serve a runtime with no durable identity.
func (h *Handler) buildAgentRuntime() {
	if h.repository == nil {
		return
	}
	description, ok := h.interactionDescription()
	if !ok {
		return
	}
	executor := agenttool.NewExecutor()
	if err := agenttool.RegisterStandardBuiltins(executor, h.repository); err != nil {
		return
	}
	coordinator, err := agentloop.New(h.repository, executor, agentloop.Identity{
		Recipe: description.Identity.Recipe, Model: description.Identity.Model,
		Node: description.Interaction.Node,
	}, h.config.MaxStoredResponses)
	if err != nil {
		return
	}
	h.agentCoordinator = coordinator
}
