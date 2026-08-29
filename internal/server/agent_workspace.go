package server

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strconv"
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
	Description string      `json:"description,omitzero"`
	Effect      string      `json:"effect,omitzero"`
	Stale       string      `json:"stale,omitzero"`
	Manual      artifact.ID `json:"manual,omitzero"`
}

type agentStepRequest struct {
	Agent     string          `json:"agent,omitzero"`
	Session   string          `json:"session"`
	Tool      string          `json:"tool"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	Approve   bool            `json:"approve,omitzero"`
}

// agentApprovalPreview projects the decision facts an operator grants
// on: the next step's operation, the manual's identity and effect, the
// exact argument bytes a grant would bind, any committed decision, and
// whether it binds these facts. It publishes and executes nothing.
func (h *Handler) agentApprovalPreview(response http.ResponseWriter, request *http.Request) {
	if h.agentCoordinator == nil {
		writeError(response, http.StatusServiceUnavailable, "agent_unavailable", "no agent runtime is configured")
		return
	}
	var body agentStepRequest
	if err := strictjson.Decode(request.Body, &body); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	if body.Session == "" || body.Tool == "" {
		writeError(response, http.StatusBadRequest, "invalid_request", "session and tool are required")
		return
	}
	arguments := body.Arguments
	if len(arguments) == 0 {
		arguments = json.RawMessage(`{}`)
	}
	sessionID := body.Session
	if body.Agent != "" {
		sessionID = body.Agent + ":" + body.Session
	}
	session, err := h.agentSessions.get(request.Context(), h.agentCoordinator, sessionID)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "agent_error", err.Error())
		return
	}
	preview, err := h.agentCoordinator.PreviewMutationDecision(request.Context(), session, body.Tool, arguments)
	if err != nil {
		writeError(response, http.StatusUnprocessableEntity, "agent_preview_refused", err.Error())
		return
	}
	payload := map[string]any{
		"call_id": preview.CallID, "operation": idText(preview.Operation),
		"manual": idText(preview.Manual), "tool": preview.Tool,
		"effect": string(preview.Effect), "arguments": preview.Arguments,
		"binds": preview.Binds,
	}
	if preview.Decision != nil {
		payload["decision"] = map[string]any{
			"id": idText(preview.Decision.ID), "answer": string(preview.Decision.Answer),
			"tool": preview.Decision.Tool, "arguments": preview.Decision.Arguments,
		}
	}
	writeJSON(response, http.StatusOK, payload)
}

// agentSessionList projects every durable agent session from the
// interaction alias namespace: the session identity, its recorded
// step count, the interaction tip, and -- restored through the same
// path a restart uses -- its inspection state against the step bound.
// A restarted server lists exactly what the ledger remembers.
func (h *Handler) agentSessionList(response http.ResponseWriter, request *http.Request) {
	if h.repository == nil || h.agentCoordinator == nil {
		writeError(response, http.StatusServiceUnavailable, "agent_unavailable", "no agent runtime is configured")
		return
	}
	result, err := h.repository.Query(request.Context(), overgodb.Query{
		Kind: artifact.KindEvidence, MaxResults: h.config.MaxStoredResponses, Projection: overgodb.ProjectAliases,
	})
	if err != nil {
		writeError(response, http.StatusInternalServerError, "agent_error", err.Error())
		return
	}
	steps := map[string]int{}
	for _, alias := range result.Aliases {
		name, ok := strings.CutPrefix(alias.Name, runrecord.InteractionResponseAliasRoot)
		if !ok {
			continue
		}
		base, stepText, ok := strings.Cut(name, "-step-")
		if !ok {
			continue
		}
		step, err := strconv.Atoi(stepText)
		if err != nil {
			continue
		}
		if step > steps[base] {
			steps[base] = step
		}
	}
	sessions := make([]map[string]any, 0, len(steps))
	for id := range steps {
		session, err := h.agentSessions.get(request.Context(), h.agentCoordinator, id)
		if err != nil {
			continue
		}
		sessions = append(sessions, map[string]any{
			"id": id, "steps": session.Steps, "inspected": session.Inspected,
			"interaction": idText(session.Interaction),
		})
	}
	slices.SortFunc(sessions, func(left, right map[string]any) int {
		return strings.Compare(left["id"].(string), right["id"].(string))
	})
	writeJSON(response, http.StatusOK, map[string]any{"sessions": sessions})
}

// agentProvenance walks one recorded step's full evidence: the
// interaction and its transcript facts (tool name, exact manual
// identity, arguments, result), the mutation receipt chain from tip
// back to admission, and the committed decision -- every link an
// artifact identity the operator can open. The walk reads what the
// step durably wrote; it derives nothing.
func (h *Handler) agentProvenance(response http.ResponseWriter, request *http.Request) {
	if h.repository == nil || h.agentCoordinator == nil {
		writeError(response, http.StatusServiceUnavailable, "agent_unavailable", "no evidence repository is configured")
		return
	}
	session := request.URL.Query().Get("session")
	step := request.URL.Query().Get("step")
	if session == "" || step == "" {
		writeError(response, http.StatusBadRequest, "invalid_request", "session and step are required")
		return
	}
	callID := session + "-step-" + step
	interaction, found, err := runrecord.ResolveInteraction(request.Context(), h.repository, callID)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	if !found {
		writeError(response, http.StatusNotFound, "not_found", "no interaction is recorded for this step")
		return
	}
	payload := map[string]any{
		"call_id": callID, "interaction": idText(interaction.ID),
		"transcript": idText(interaction.Message), "trace": idText(interaction.Trace),
	}
	if interaction.Parent.Valid() {
		payload["parent"] = idText(interaction.Parent)
	}
	transcript, err := runrecord.RequireInteractionTranscript(request.Context(), h.repository, interaction.Message)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	for _, message := range transcript.Messages {
		for _, call := range message.ToolCalls {
			payload["tool"] = call.Name
			payload["arguments"] = call.Arguments
			if call.Manual.Valid() {
				payload["manual"] = idText(call.Manual)
			}
		}
		if message.ToolCallID != "" {
			payload["result"] = map[string]any{"content": message.Content, "error": message.ToolResultError}
		}
	}
	operation, err := agentloop.MutationReceiptOperation(callID)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	if tip, found, err := runrecord.ResolveStageReceipt(request.Context(), h.repository, operation, h.agentCoordinator.ServingIdentity().Node); err == nil && found {
		receipts := []map[string]any{}
		for current, ok := tip, true; ok; {
			entry := map[string]any{"id": idText(current.ID), "state": string(current.State)}
			if current.Failure != "" {
				entry["failure"] = current.Failure
			}
			receipts = append(receipts, entry)
			if !current.Previous.Valid() {
				break
			}
			content, present, err := artifact.ReadContent(request.Context(), h.repository, current.Previous)
			if err != nil || !present {
				break
			}
			current, err = runrecord.ParseStageReceipt(content.Data)
			ok = err == nil
		}
		payload["operation"] = idText(operation)
		payload["receipts"] = receipts
	}
	if decision, found, err := runrecord.ResolveHumanDecision(request.Context(), h.repository, operation); err == nil && found {
		payload["decision"] = map[string]any{
			"id": idText(decision.ID), "answer": string(decision.Answer),
			"tool": decision.Tool, "arguments": decision.Arguments,
		}
	}
	writeJSON(response, http.StatusOK, payload)
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
	// The standard manuals publish idempotently at boot: a fresh store
	// serves a working agent workspace out of the box instead of
	// refusing every tool until an operator runs the CLI.
	if manuals, err := agenttool.StandardManuals(); err == nil {
		if _, err := agenttool.PublishManualCatalog(context.Background(), h.repository, manuals); err != nil {
			return
		}
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
