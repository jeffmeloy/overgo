package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
)

// AgentDefinitionInput is the typed editing boundary shared by API and GUI.
type AgentDefinitionInput struct {
	Name              string        `json:"name"`
	Prompt            artifact.ID   `json:"prompt"`
	ModelRecipe       artifact.ID   `json:"model_recipe"`
	ToolManuals       []artifact.ID `json:"tool_manuals,omitempty"`
	CapabilityBundles []artifact.ID `json:"capability_bundles,omitempty"`
	Datasets          []artifact.ID `json:"datasets,omitempty"`
	Automations       []artifact.ID `json:"automations,omitempty"`
	Policies          []artifact.ID `json:"policies"`
}

// AgentAutomationAttachment exposes one exact attached automation definition.
type AgentAutomationAttachment struct {
	ID   artifact.ID `json:"id"`
	Name string      `json:"name"`
}

// AgentInventoryEntry projects current immutable lifecycle authority.
type AgentInventoryEntry struct {
	Name        string                      `json:"name"`
	Definition  artifact.ID                 `json:"definition"`
	Activation  artifact.ID                 `json:"activation"`
	State       runrecord.AgentState        `json:"state"`
	Prompt      artifact.ID                 `json:"prompt"`
	ModelRecipe artifact.ID                 `json:"model_recipe"`
	Tools       []artifact.ID               `json:"tools,omitempty"`
	Bundles     []artifact.ID               `json:"capability_bundles,omitempty"`
	Datasets    []artifact.ID               `json:"datasets,omitempty"`
	Automations []AgentAutomationAttachment `json:"automations,omitempty"`
	Policies    []artifact.ID               `json:"policies"`
	Refusal     string                      `json:"refusal,omitempty"`
}

// AgentObservable is a structured tool event; model hidden reasoning is never projected.
type AgentObservable struct {
	Interaction artifact.ID                 `json:"interaction"`
	Transcript  artifact.ID                 `json:"transcript"`
	Step        int                         `json:"step"`
	Calls       []AgentToolCallObservable   `json:"calls,omitempty"`
	Results     []AgentToolResultObservable `json:"results,omitempty"`
}

// AgentToolCallObservable identifies admitted tool authority without copying arguments.
type AgentToolCallObservable struct {
	Name   string      `json:"name"`
	Manual artifact.ID `json:"manual"`
}

// AgentToolResultObservable projects one durable tool result without model reasoning.
type AgentToolResultObservable struct {
	ToolCallID string `json:"tool_call_id"`
	Error      bool   `json:"error,omitempty"`
}

type agentDefinitionRequest struct {
	Definition artifact.ID `json:"definition"`
}

type agentStateRequest struct {
	Name  string               `json:"name"`
	State runrecord.AgentState `json:"state"`
}

type agentRetrievalRequest struct {
	Agent        string      `json:"agent"`
	Projection   artifact.ID `json:"projection"`
	Query        string      `json:"query"`
	Limit        uint64      `json:"limit"`
	RerankPolicy artifact.ID `json:"rerank_policy"`
}

type agentAutomationRequest struct {
	Agent       string                     `json:"agent"`
	Automation  artifact.ID                `json:"automation"`
	Key         string                     `json:"key,omitempty"`
	Destination string                     `json:"destination,omitempty"`
	Inputs      map[string]json.RawMessage `json:"inputs"`
}

type agentChatRequest struct {
	Agent    string                  `json:"agent"`
	Messages []inference.ChatMessage `json:"messages"`
}

func (h *Handler) agentControl(response http.ResponseWriter, request *http.Request) {
	if h.repository == nil || h.agentCoordinator == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "agent control workspace is unavailable")
		return
	}
	switch request.URL.Path {
	case "/agents":
		if !requireMethod(response, request, http.MethodGet) {
			return
		}
		inventory, err := h.agentInventory(request.Context())
		writeAgentResult(response, http.StatusOK, inventory, err)
	case "/agents/definitions":
		var body AgentDefinitionInput
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		definition, err := h.publishAgentDefinition(request.Context(), body)
		writeAgentResult(response, http.StatusCreated, struct {
			ID         artifact.ID            `json:"id"`
			Definition recipe.AgentDefinition `json:"definition"`
		}{definition.ID, definition}, err)
	case "/agents/activate":
		var body agentDefinitionRequest
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		active, err := h.activateAgent(request.Context(), body.Definition)
		writeAgentResult(response, http.StatusOK, active, err)
	case "/agents/state":
		var body agentStateRequest
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		active, err := h.transitionAgent(request.Context(), body.Name, body.State)
		writeAgentResult(response, http.StatusOK, active, err)
	case "/agents/step":
		h.agentStep(response, request)
	case "/agents/approval":
		h.agentApprovalPreview(response, request)
	case "/agents/chat":
		h.agentChat(response, request)
	case "/agents/retrieval":
		var body agentRetrievalRequest
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		results, err := h.agentRetrieval(request.Context(), body)
		writeAgentResult(response, http.StatusOK, results, err)
	case "/agents/automation":
		var body agentAutomationRequest
		if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
			return
		}
		execution, err := h.runAgentAutomation(context.WithoutCancel(request.Context()), body)
		writeAgentResult(response, http.StatusAccepted, execution, err)
	case "/agents/evidence":
		if !requireMethod(response, request, http.MethodGet) {
			return
		}
		value, err := h.agentEvidence(request.Context(), request.URL.Query().Get("session"))
		writeAgentResult(response, http.StatusOK, value, err)
	case "/agents/stream":
		h.agentStream(response, request)
	default:
		writeError(response, http.StatusNotFound, "not_found", "agent route is absent")
	}
}

func (h *Handler) agentChat(response http.ResponseWriter, request *http.Request) {
	var body agentChatRequest
	if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	active, err := (runrecord.AgentAuthority{Repository: h.repository}).RequireActive(request.Context(), body.Agent)
	description, described := h.interactionDescription()
	if err != nil || !described || active.Definition.ModelRecipe != description.Identity.Recipe || len(body.Messages) == 0 {
		writeAgentResult(response, http.StatusOK, nil, errors.Join(errors.New("server: active agent chat authority differs"), err))
		return
	}
	for _, message := range body.Messages {
		if message.Role == inference.ChatRoleSystem || message.ReasoningContent != "" || len(message.ToolCalls) != 0 ||
			message.ToolCallID != "" || message.ToolResultError || len(message.Media) != 0 {
			writeAgentResult(response, http.StatusOK, nil, errors.New("server: agent chat message is outside the supported contract"))
			return
		}
	}
	prompt, found, err := artifact.ReadContent(request.Context(), h.repository, active.Definition.Prompt)
	if err != nil || !found {
		writeAgentResult(response, http.StatusOK, nil, errors.Join(errors.New("server: agent prompt content is unavailable"), err))
		return
	}
	messages := make([]inference.ChatMessage, 0, len(body.Messages)+1)
	messages = append(messages, inference.ChatMessage{Role: inference.ChatRoleSystem, Content: string(prompt.Data)})
	messages = append(messages, body.Messages...)
	maxTokens := h.config.MaxTokens
	encoded, err := json.Marshal(chatCompletionRequest{Messages: messages, MaxTokens: &maxTokens})
	if err != nil {
		writeAgentResult(response, http.StatusOK, nil, err)
		return
	}
	request.Body = http.NoBody
	forwarded := request.Clone(request.Context())
	forwarded.Body = io.NopCloser(bytes.NewReader(encoded))
	forwarded.ContentLength = int64(len(encoded))
	h.chatCompletions(response, forwarded)
}

func (h *Handler) agentInventory(ctx context.Context) ([]AgentInventoryEntry, error) {
	result, err := h.repository.Query(ctx, overgodb.Query{
		MaxResults: h.config.MaxStoredResponses, Projection: overgodb.ProjectAliases,
	})
	if err != nil {
		return nil, err
	}
	authority := runrecord.AgentAuthority{Repository: h.repository}
	entries := make([]AgentInventoryEntry, 0)
	for _, alias := range result.Aliases {
		name, found := strings.CutPrefix(alias.Name, runrecord.AgentActiveAliasRoot)
		if !found {
			continue
		}
		active, resolved, resolveErr := authority.Resolve(ctx, name)
		if resolveErr != nil || !resolved {
			return nil, errors.Join(errors.New("server: agent active alias differs"), resolveErr)
		}
		definition := active.Definition
		entry := AgentInventoryEntry{
			Name: name, Definition: definition.ID, Activation: active.Activation.ID, State: active.Activation.State,
			Prompt: definition.Prompt, ModelRecipe: definition.ModelRecipe, Tools: slices.Clone(definition.ToolManuals),
			Bundles: slices.Clone(definition.CapabilityBundles), Datasets: slices.Clone(definition.Datasets),
			Policies: slices.Clone(definition.Policies),
		}
		for _, id := range definition.Automations {
			automation, automationErr := recipe.RequireAutomationDefinition(ctx, h.repository, id)
			if automationErr != nil {
				entry.Refusal = automationErr.Error()
				continue
			}
			entry.Automations = append(entry.Automations, AgentAutomationAttachment{ID: id, Name: automation.Name})
		}
		if active.Activation.State != runrecord.AgentActive {
			entry.Refusal = "agent is paused"
		}
		entries = append(entries, entry)
	}
	slices.SortFunc(entries, func(left, right AgentInventoryEntry) int { return strings.Compare(left.Name, right.Name) })
	return entries, nil
}

func (h *Handler) publishAgentDefinition(ctx context.Context, input AgentDefinitionInput) (recipe.AgentDefinition, error) {
	definition, err := recipe.NewAgentDefinition(recipe.AgentDefinition{
		Name: input.Name, Prompt: input.Prompt, ModelRecipe: input.ModelRecipe,
		ToolManuals: input.ToolManuals, CapabilityBundles: input.CapabilityBundles,
		Datasets: input.Datasets, Automations: input.Automations, Policies: input.Policies,
	})
	if err != nil {
		return recipe.AgentDefinition{}, err
	}
	err = (runrecord.AgentAuthority{Repository: h.repository}).PublishDefinition(
		ctx, "agent/workspace/definition/"+definition.ID.String(), definition,
	)
	return definition, err
}

func (h *Handler) activateAgent(ctx context.Context, id artifact.ID) (runrecord.ActiveAgent, error) {
	definition, err := recipe.RequireAgentDefinition(ctx, h.repository, id)
	if err != nil {
		return runrecord.ActiveAgent{}, err
	}
	current, found, err := (runrecord.AgentAuthority{Repository: h.repository}).Resolve(ctx, definition.Name)
	if err != nil {
		return runrecord.ActiveAgent{}, err
	}
	prior := artifact.ID{}
	if found {
		prior = current.Activation.ID
	}
	decision, err := h.publishAgentDecision(ctx, definition.Name, definition.ID, runrecord.AgentActive, prior)
	if err != nil {
		return runrecord.ActiveAgent{}, err
	}
	return (runrecord.AgentAuthority{Repository: h.repository}).Activate(
		ctx, "agent/workspace/activate/"+definition.ID.String(), definition, decision,
	)
}

func (h *Handler) transitionAgent(ctx context.Context, name string, state runrecord.AgentState) (runrecord.ActiveAgent, error) {
	authority := runrecord.AgentAuthority{Repository: h.repository}
	current, found, err := authority.Resolve(ctx, name)
	if err != nil || !found {
		return runrecord.ActiveAgent{}, errors.Join(errors.New("server: agent authority is absent"), err)
	}
	decision, err := h.publishAgentDecision(ctx, name, current.Definition.ID, state, current.Activation.ID)
	if err != nil {
		return runrecord.ActiveAgent{}, err
	}
	switch state {
	case runrecord.AgentPaused:
		return authority.Pause(ctx, "agent/workspace/pause/"+decision.String(), name, decision)
	case runrecord.AgentActive:
		return authority.Resume(ctx, "agent/workspace/resume/"+decision.String(), name, decision)
	default:
		return runrecord.ActiveAgent{}, errors.New("server: invalid agent state")
	}
}

func (h *Handler) publishAgentDecision(
	ctx context.Context,
	name string,
	definition artifact.ID,
	state runrecord.AgentState,
	prior artifact.ID,
) (artifact.ID, error) {
	content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo.agent-workspace-decision/v1"), struct {
		Name       string               `json:"name"`
		Definition artifact.ID          `json:"definition"`
		State      runrecord.AgentState `json:"state"`
		Prior      artifact.ID          `json:"prior,omitzero"`
	}{Name: name, Definition: definition, State: state, Prior: prior})
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"agent/workspace/decision/"+content.Descriptor.ID.String(), []artifact.Content{content},
		artifact.DependencyLineage(content.Descriptor.ID, definition), nil,
	)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, h.repository, batch)
	}
	if err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return artifact.ID{}, err
	}
	return content.Descriptor.ID, nil
}

func (h *Handler) agentRetrieval(ctx context.Context, request agentRetrievalRequest) ([]dataset.AgentRetrievalResult, error) {
	active, err := (runrecord.AgentAuthority{Repository: h.repository}).RequireActive(ctx, request.Agent)
	if err != nil || h.config.AgentEmbedder == nil || h.config.AgentReranker == nil ||
		!slices.Contains(active.Definition.Policies, request.RerankPolicy) {
		return nil, errors.Join(errors.New("server: agent retrieval authority is unavailable"), err)
	}
	return (dataset.AgentRetrievalBuilder{Repository: h.repository}).Search(ctx, dataset.AgentRetrievalQuery{
		Projection: request.Projection, Text: request.Query, Limit: request.Limit,
		AllowedDatasets: active.Definition.Datasets, Embedder: h.config.AgentEmbedder, Reranker: h.config.AgentReranker,
	})
}

func (h *Handler) runAgentAutomation(ctx context.Context, request agentAutomationRequest) (workflowruntime.AutomationExecution, error) {
	active, err := (runrecord.AgentAuthority{Repository: h.repository}).RequireActive(ctx, request.Agent)
	if err != nil || !slices.Contains(active.Definition.Automations, request.Automation) {
		return workflowruntime.AutomationExecution{}, errors.Join(errors.New("server: automation is outside active agent authority"), err)
	}
	automation, err := recipe.RequireAutomationDefinition(ctx, h.repository, request.Automation)
	if err != nil {
		return workflowruntime.AutomationExecution{}, err
	}
	workspace, ok := h.generator.(AutomationWorkspaceAPI)
	if !ok {
		return workflowruntime.AutomationExecution{}, errors.New("server: automation workspace is unavailable")
	}
	return workspace.RunAutomation(ctx, h.operations, AutomationExecutionInput{
		Name: automation.Name, Key: request.Key, Destination: request.Destination, Inputs: request.Inputs,
	})
}

func (h *Handler) agentEvidence(ctx context.Context, session string) ([]AgentObservable, error) {
	if session == "" {
		return nil, errors.New("server: agent session is required")
	}
	result := make([]AgentObservable, 0)
	for index := range h.config.MaxStoredResponses {
		step := index
		step++
		interaction, found, err := runrecord.ResolveInteraction(ctx, h.repository, fmt.Sprintf("%s-step-%d", session, step))
		if err != nil {
			return nil, err
		}
		if !found {
			break
		}
		transcript, err := runrecord.RequireInteractionTranscript(ctx, h.repository, interaction.Message)
		if err != nil {
			return nil, err
		}
		observable := AgentObservable{Interaction: interaction.ID, Transcript: interaction.Message, Step: step}
		for _, message := range transcript.Messages {
			for _, call := range message.ToolCalls {
				observable.Calls = append(observable.Calls, AgentToolCallObservable{Name: call.Name, Manual: call.Manual})
			}
			if message.ToolCallID != "" {
				observable.Results = append(observable.Results, AgentToolResultObservable{
					ToolCallID: message.ToolCallID, Error: message.ToolResultError,
				})
			}
		}
		result = append(result, observable)
	}
	return result, nil
}

func (h *Handler) agentStream(response http.ResponseWriter, request *http.Request) {
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
	inventory, err := h.agentInventory(request.Context())
	if err != nil || stream.named("agent.inventory", inventory) != nil ||
		stream.named("operation.snapshot", h.operations.List()) != nil {
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

func writeAgentResult(response http.ResponseWriter, status int, value any, err error) {
	if err != nil {
		writeError(response, http.StatusUnprocessableEntity, "agent_refused", err.Error())
		return
	}
	writeJSON(response, status, value)
}
