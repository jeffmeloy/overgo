package server

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"overgo/internal/inference"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

func writeInvalidRequest(response http.ResponseWriter, err error) {
	writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
}

func writeInvalidRequestMessage(response http.ResponseWriter, message string) {
	writeError(response, http.StatusBadRequest, "invalid_request_error", message)
}

func emitGenerationError(write func(any) error, err error) error {
	return write(errorEnvelope("generation_error", err.Error()))
}

func emitNamedGenerationError(
	write func(string, any) error,
	event string,
	err error,
) error {
	return write(event, map[string]any{
		"type": event, "error": errorEnvelope("generation_error", err.Error()).Error,
	})
}

func (h *Handler) decodeProtocolJSON(
	response http.ResponseWriter,
	request *http.Request,
	target any,
	model func() string,
) bool {
	return h.decodeMultimodalJSON(response, request, target) &&
		h.requireModel(response, model())
}

type protocolGenerationPlan struct {
	handler   *Handler
	request   *http.Request
	session   *requestSession
	prompt    preparedPrompt
	sampler   *sampling.Sampler
	maxTokens int
	stops     []string
}

type protocolGenerationResult struct {
	ids  []tokenizer.TokenID
	pump *generationPump
}

type protocolBatchGenerationPlan struct {
	handler   *Handler
	request   *http.Request
	session   *requestSession
	prompts   []nativePrompt
	sampler   *sampling.Sampler
	maxTokens int
	stops     []string
}

func (h *Handler) prepareProtocolBatchGenerationPlan(
	response http.ResponseWriter,
	request *http.Request,
	prompts []nativePrompt,
	parameters samplingParameters,
	maxTokens int,
	stops []string,
) (*protocolBatchGenerationPlan, bool) {
	sampler, err := h.newSampler(parameters)
	if err != nil {
		writeInvalidRequest(response, err)
		return nil, false
	}
	lease, acquired := h.acquireRequestSession(request.Context(), response, -1)
	if !acquired {
		return nil, false
	}
	return &protocolBatchGenerationPlan{
		handler: h, request: request, session: lease, prompts: prompts,
		sampler: sampler, maxTokens: maxTokens, stops: stops,
	}, true
}

func (plan *protocolBatchGenerationPlan) release() {
	plan.handler.releaseSession(plan.session)
}

func (plan *protocolBatchGenerationPlan) run(
	choices int,
	emit func(int, string) error,
	accept func(int, int, protocolGenerationResult) error,
) error {
	choiceIndex := 0
	for _, prompt := range plan.prompts {
		for promptChoice := range choices {
			sampler, err := samplerForChoice(plan.sampler, choiceIndex)
			if err != nil {
				return err
			}
			var emitChoice func(string) error
			if emit != nil {
				index := choiceIndex
				emitChoice = func(piece string) error { return emit(index, piece) }
			}
			ids, pump, err := plan.handler.generateWithPump(
				plan.request.Context(), plan.session, prompt.Text,
				inference.GenerateOptions{
					MaxNewTokens: plan.maxTokens, Sampler: sampler,
					PromptTokenIDs: prompt.TokenIDs,
					ContextShift:   plan.handler.config.ContextShift,
				},
				plan.stops, emitChoice,
			)
			if err != nil {
				return err
			}
			if err := accept(
				promptChoice, choiceIndex, protocolGenerationResult{ids: ids, pump: pump},
			); err != nil {
				return err
			}
			choiceIndex++
		}
	}
	return nil
}

func (result protocolGenerationResult) promptTokens() int {
	return len(result.ids) - result.pump.generated
}

func (h *Handler) prepareProtocolGenerationPlan(
	response http.ResponseWriter,
	request *http.Request,
	prompt nativePrompt,
	parameters samplingParameters,
	maxTokens int,
	stops []string,
) (*protocolGenerationPlan, bool) {
	sampler, err := h.newSampler(parameters)
	if err != nil {
		writeInvalidRequest(response, err)
		return nil, false
	}
	lease, acquired := h.acquireRequestSession(request.Context(), response, -1)
	if !acquired {
		return nil, false
	}
	prepared, err := h.preparePrompt(request.Context(), prompt, true)
	if err != nil {
		h.releaseSession(lease)
		writeGenerationError(response, err)
		return nil, false
	}
	return &protocolGenerationPlan{
		handler: h, request: request, session: lease, prompt: prepared,
		sampler: sampler, maxTokens: maxTokens, stops: stops,
	}, true
}

func (plan *protocolGenerationPlan) release() {
	plan.handler.releaseSession(plan.session)
}

func (plan *protocolGenerationPlan) context() context.Context {
	return plan.request.Context()
}

func (plan *protocolGenerationPlan) run(
	emit func(string) error,
) (protocolGenerationResult, error) {
	return plan.runWithSampler(plan.sampler, emit)
}

func (plan *protocolGenerationPlan) runChoice(
	index int,
	emit func(string) error,
) (protocolGenerationResult, error) {
	sampler, err := samplerForChoice(plan.sampler, index)
	if err != nil {
		return protocolGenerationResult{}, err
	}
	return plan.runWithSampler(sampler, emit)
}

func (plan *protocolGenerationPlan) runWithSampler(
	sampler *sampling.Sampler,
	emit func(string) error,
) (protocolGenerationResult, error) {
	ids, pump, err := plan.handler.generateWithPump(
		plan.context(),
		plan.session,
		plan.prompt.Text,
		plan.handler.protocolGenerationOptions(
			plan.maxTokens, sampler, plan.prompt.TokenIDs, plan.prompt.ProjectedInputs,
		),
		plan.stops,
		emit,
	)
	return protocolGenerationResult{ids: ids, pump: pump}, err
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
			writeInvalidRequest(response, err)
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

type toolDeltaSink struct {
	Reasoning func(string) error
	Content   func(string) error
	Tool      func(inference.ChatToolCallDelta) error
}

func (h *Handler) requireChatFormatter(
	response http.ResponseWriter,
) (ChatFormatter, bool) {
	formatter, ok := h.generator.(ChatFormatter)
	if !ok {
		writeError(
			response,
			http.StatusNotImplemented,
			errorCodeUnsupportedOperation,
			"chat formatting is unavailable",
		)
	}
	return formatter, ok
}

func (h *Handler) requireProtocolTokenCounting(
	response http.ResponseWriter,
	request *http.Request,
) (ChatFormatter, bool) {
	formatter, ok := h.requireChatFormatter(response)
	if !ok || !h.requireTokenCounting(response) {
		return nil, false
	}
	return formatter, true
}

func (h *Handler) requireChatOutputParser(
	response http.ResponseWriter,
	feature string,
) (ChatOutputParser, bool) {
	parser, ok := h.generator.(ChatOutputParser)
	if !ok {
		writeError(
			response, http.StatusNotImplemented, errorCodeUnsupportedOperation,
			feature+" output parsing is unavailable",
		)
	}
	return parser, ok
}

func (h *Handler) requireTokenCounting(response http.ResponseWriter) bool {
	if _, ok := h.generator.(TokenizationAPI); ok {
		return true
	}
	writeError(
		response,
		http.StatusNotImplemented,
		errorCodeUnsupportedOperation,
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

func (s *toolDeltaStream) route(piece string, sink toolDeltaSink) error {
	deltas, err := s.accept(piece)
	if err != nil {
		return err
	}
	for _, delta := range deltas {
		if delta.ReasoningContent != "" && sink.Reasoning != nil {
			if err := sink.Reasoning(delta.ReasoningContent); err != nil {
				return err
			}
		}
		if delta.Content != "" && sink.Content != nil {
			if err := sink.Content(delta.Content); err != nil {
				return err
			}
		}
		if sink.Tool != nil {
			if err := sink.Tool(delta); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *toolDeltaStream) parse(parser ChatOutputParser, tools []inference.ChatTool) (inference.ChatMessage, error) {
	return parser.ParseChatOutput(s.text(), tools)
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
			errorCodeUnsupportedOperation,
			"tool-call grammar generation is unavailable",
		)
		return false
	}
	source, root, patterns, err := provider.ChatToolGrammar(
		tools, required, thinking, parallel,
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return false
	}
	parameters.Grammar = source
	parameters.GrammarRoot = root
	parameters.GrammarLazy = len(patterns) != 0
	parameters.GrammarTriggerPatterns = patterns
	return true
}

// writeRefusableResult reports one route-family outcome: a refusal
// error maps to unprocessable-entity with the family refusal code,
// success writes the value at the given status.
func writeRefusableResult(response http.ResponseWriter, status int, value any, err error, refusal string) {
	if err != nil {
		writeError(response, http.StatusUnprocessableEntity, refusal, err.Error())
		return
	}
	writeJSON(response, status, value)
}
