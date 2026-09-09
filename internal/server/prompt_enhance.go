package server

import (
	"errors"
	"net/http"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inference"
)

// PromptEnhanceInstruction is the one instruction the served chat model
// rewrites a generation prompt under: the page never types it, and the
// enhancement record carries it so the rewrite is reproducible.
const PromptEnhanceInstruction = "Rewrite the following prompt for an image, video or speech generation model. " +
	"Keep every subject, action, setting and constraint it states, add concrete visual or acoustic detail " +
	"(composition, lighting, material, mood, camera or voice), and answer with the rewritten prompt alone: " +
	"one paragraph, no preamble, no quotation marks, no explanation."

// promptEnhancementSchema names the stored record of one enhancement: the
// instruction, the original, the rewrite and the model that wrote it.
const promptEnhancementSchema = "overgo/prompt-enhancement/v1"

// promptEnhancement is the stored record and the route's answer.
type promptEnhancement struct {
	Version     uint16 `json:"version"`
	Instruction string `json:"instruction"`
	Original    string `json:"original"`
	Enhanced    string `json:"enhanced"`
	Model       string `json:"model"`
}

type promptEnhanceRequest struct {
	Prompt string `json:"prompt"`
}

type promptEnhanceResponse struct {
	promptEnhancement
	// Record is the stored enhancement a generation run cites as a source
	// once the page accepts the rewrite; empty without a repository.
	Record string `json:"record,omitzero"`
}

// promptEnhance answers /generation/enhance: the prompt rewritten by the
// served chat model under the fixed instruction, recorded with its
// original so a run that uses the rewrite keeps the original as its
// source.
func (h *Handler) promptEnhance(response http.ResponseWriter, request *http.Request) {
	formatter, ok := h.requireChatFormatter(response)
	if !ok {
		return
	}
	var body promptEnhanceRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	original := strings.TrimSpace(body.Prompt)
	if original == "" {
		writeInvalidRequestMessage(response, "prompt is required")
		return
	}
	chat := chatCompletionRequest{Messages: []inference.ChatMessage{
		{Role: inference.ChatRoleSystem, Content: PromptEnhanceInstruction},
		{Role: inference.ChatRoleUser, Content: original},
	}}
	prompt, err := h.normalizeChatPrompt(request.Context(), formatter, chat, nil, false)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	// The rewrite takes the default completion budget, held to the server's ceiling.
	maxTokens, err := boundedProtocolTokens(nil, min(h.defaultOutputTokens, h.config.MaxTokens), h.config.MaxTokens, "max_tokens", false)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	plan, ok := h.prepareProtocolGenerationPlan(response, request, prompt, samplingParameters{}, maxTokens, nil)
	if !ok {
		return
	}
	defer plan.release()
	result, err := plan.run(nil)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	enhanced := strings.TrimSpace(result.pump.text())
	if enhanced == "" {
		writeError(response, http.StatusInternalServerError, "generation_error", "the served model answered no rewrite")
		return
	}
	answer := promptEnhanceResponse{promptEnhancement: promptEnhancement{
		Version: artifact.InitialDocumentVersion, Instruction: PromptEnhanceInstruction, Original: original, Enhanced: enhanced, Model: h.config.ModelID,
	}}
	if h.repository != nil {
		content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, promptEnhancementSchema), answer.promptEnhancement)
		if err != nil {
			writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
			return
		}
		if _, err := artifact.CommitBatch(request.Context(), h.repository, artifact.Batch{
			Key: "prompt-enhancement/" + content.Descriptor.ID.String(), Contents: []artifact.Content{content},
		}); err != nil && !errors.Is(err, artifact.ErrNoChange) {
			writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
			return
		}
		answer.Record = content.Descriptor.ID.String()
	}
	writeJSON(response, http.StatusOK, answer)
}
