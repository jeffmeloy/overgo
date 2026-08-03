package server

import (
	"fmt"
	"net/http"
	"strings"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

func (h *Handler) prepareProtocolGeneration(
	response http.ResponseWriter,
	request *http.Request,
	prompt nativePrompt,
) (preparedPrompt, int, bool) {
	slotID, acquired := h.acquireRequestSlot(response, -1)
	if !acquired {
		return preparedPrompt{}, 0, false
	}
	prepared, err := h.preparePrompt(request.Context(), prompt, true)
	if err != nil {
		h.releaseSlot(slotID)
		writeGenerationError(response, err)
		return preparedPrompt{}, 0, false
	}
	return prepared, slotID, true
}

func (h *Handler) writeProtocolInputTokenCount(
	response http.ResponseWriter,
	request *http.Request,
	prompt nativePrompt,
	object bool,
) {
	prepared, err := h.preparePrompt(request.Context(), prompt, true)
	if err != nil {
		if nativePromptHasMedia(prompt) {
			writeGenerationError(response, err)
		} else {
			writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		}
		return
	}
	payload := map[string]any{"input_tokens": len(prepared.TokenIDs)}
	if object {
		payload["object"] = "response.input_tokens"
	}
	writeJSON(response, http.StatusOK, payload)
}

type toolDeltaStream struct {
	output    inference.ChatOutputStream
	buffered  strings.Builder
	names     []string
	arguments []strings.Builder
	started   bool
}

func (h *Handler) requireChatFormatter(
	response http.ResponseWriter,
) (ChatFormatter, bool) {
	formatter, ok := h.generator.(ChatFormatter)
	if !ok {
		writeError(
			response,
			http.StatusNotImplemented,
			"unsupported_operation",
			"chat formatting is unavailable",
		)
	}
	return formatter, ok
}

func (h *Handler) requireTokenCounting(response http.ResponseWriter) bool {
	if _, ok := h.generator.(TokenizationAPI); ok {
		return true
	}
	writeError(
		response,
		http.StatusNotImplemented,
		"unsupported_operation",
		"token counting is unavailable",
	)
	return false
}

func newToolDeltaStream(
	generator Generator,
	tools []inference.ChatTool,
) (*toolDeltaStream, error) {
	result := &toolDeltaStream{}
	if len(tools) == 0 {
		return result, nil
	}
	provider, ok := generator.(ChatOutputStreamProvider)
	if !ok {
		return result, nil
	}
	output, err := provider.NewChatOutputStream(tools)
	if err != nil {
		return nil, err
	}
	result.output = output
	return result, nil
}

func (s *toolDeltaStream) accept(piece string) ([]inference.ChatToolCallDelta, error) {
	s.buffered.WriteString(piece)
	if s.output == nil || piece == "" {
		return nil, nil
	}
	deltas, err := s.output.Accept(piece)
	if err != nil {
		return nil, err
	}
	for _, delta := range deltas {
		if delta.Index < 0 {
			return nil, fmt.Errorf("tool stream returned negative index %d", delta.Index)
		}
		for len(s.names) <= delta.Index {
			s.names = append(s.names, "")
			s.arguments = append(s.arguments, strings.Builder{})
		}
		if delta.Started {
			s.names[delta.Index] = delta.Name
			s.started = true
		}
		if delta.Arguments != "" {
			s.arguments[delta.Index].WriteString(delta.Arguments)
		}
	}
	return deltas, nil
}

func (s *toolDeltaStream) text() string {
	return s.buffered.String()
}

func (s *toolDeltaStream) streamed(index int) bool {
	return index >= 0 && index < len(s.names) && s.names[index] != ""
}

func (s *toolDeltaStream) argumentText(index int) string {
	if index < 0 || index >= len(s.arguments) {
		return ""
	}
	return s.arguments[index].String()
}

func boundedProtocolTokens(
	configured *int,
	fallback, limit int,
	field string,
	required bool,
) (int, error) {
	value := fallback
	if configured == nil {
		if required {
			return 0, fmt.Errorf("%s is required", field)
		}
	} else {
		value = *configured
	}
	if value < 0 || value > limit {
		return 0, fmt.Errorf("%s must be in [0,%d]", field, limit)
	}
	return value, nil
}

func (h *Handler) protocolGenerationOptions(
	maxTokens int,
	sampler *sampling.Sampler,
	promptIDs []tokenizer.TokenID,
	projected *inference.ProjectedInputs,
) inference.GenerateOptions {
	return inference.GenerateOptions{
		MaxNewTokens:    maxTokens,
		Sampler:         sampler,
		ParseSpecial:    true,
		ContextShift:    h.config.ContextShift,
		PromptTokenIDs:  promptIDs,
		CachePrompt:     projected != nil,
		ProjectedInputs: projected,
	}
}

func (h *Handler) configureChatToolGrammar(
	response http.ResponseWriter,
	parameters *samplingParameters,
	tools []inference.ChatTool,
	required, thinking, parallel bool,
) bool {
	if len(tools) == 0 {
		return true
	}
	provider, ok := h.generator.(ChatToolGrammarProvider)
	if !ok {
		writeError(
			response,
			http.StatusNotImplemented,
			"unsupported_operation",
			"tool-call grammar generation is unavailable",
		)
		return false
	}
	source, root, patterns, err := provider.ChatToolGrammar(
		tools, required, thinking, parallel,
	)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return false
	}
	parameters.Grammar = source
	parameters.GrammarRoot = root
	parameters.GrammarLazy = len(patterns) != 0
	parameters.GrammarTriggerPatterns = patterns
	return true
}
