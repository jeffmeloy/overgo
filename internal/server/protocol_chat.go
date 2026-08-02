package server

import (
	"bytes"

	"encoding/json"

	"errors"

	"fmt"

	"io"

	"llamacpp2go/internal/inference"

	"llamacpp2go/internal/sampling"

	"llamacpp2go/internal/tokenizer"

	"net/http"

	"strconv"

	"strings"

	"time"
)

type chatCompletionRequest struct {
	Model          string                  `json:"model"`
	Messages       []inference.ChatMessage `json:"messages"`
	Tools          []inference.ChatTool    `json:"tools"`
	ToolChoice     json.RawMessage         `json:"tool_choice"`
	ParallelTools  *bool                   `json:"parallel_tool_calls"`
	AddPrompt      *bool                   `json:"add_generation_prompt"`
	TemplateKwargs map[string]any          `json:"chat_template_kwargs"`
	MaxTokens      *int                    `json:"max_tokens"`
	Stop           json.RawMessage         `json:"stop"`
	N              int                     `json:"n"`
	JSONSchema     json.RawMessage         `json:"json_schema"`
	ResponseFormat json.RawMessage         `json:"response_format"`
	samplingParameters
	Stream bool `json:"stream"`
}

func formatChatRequest(
	formatter ChatFormatter,
	messages []inference.ChatMessage,
	tools []inference.ChatTool,
	addGenerationPrompt *bool,
	kwargs map[string]any,
) (string, error) {
	addPrompt := true
	if addGenerationPrompt != nil {
		addPrompt = *addGenerationPrompt
	}
	enableThinking, err := chatThinkingEnabled(kwargs)
	if err != nil {
		return "", err
	}
	if len(tools) == 0 && addPrompt && enableThinking {
		return formatter.FormatChat(messages)
	}
	extended, ok := formatter.(ToolAwareChatFormatter)
	if !ok {
		return "", errors.New("tool-aware chat formatting is unavailable")
	}
	return extended.FormatChatWithOptions(messages, inference.ChatFormatOptions{
		Tools:               tools,
		AddGenerationPrompt: addPrompt,
		EnableThinking:      enableThinking,
	})
}

func chatMediaCount(messages []inference.ChatMessage) int {
	count := 0
	for index := range messages {
		count += len(messages[index].Media)
	}
	return count
}

func (h *Handler) parseChatSingleMultimodalPrompt(
	body chatCompletionRequest,
) (nativePrompt, error) {
	if body.N != 1 {
		return nativePrompt{}, errors.New("multimodal chat requires n=1")
	}
	if len(body.Messages) != 1 ||
		body.Messages[0].Role != "user" ||
		len(body.Messages[0].Media) == 0 || len(body.Messages[0].Media) > 8 {
		return nativePrompt{}, errors.New("multimodal chat requires one user message with one to eight media items")
	}
	if len(body.Tools) != 0 || rawJSONConfigured(body.ToolChoice) || body.ParallelTools != nil {
		return nativePrompt{}, errors.New("multimodal chat cannot use tools")
	}
	if body.AddPrompt != nil && !*body.AddPrompt {
		return nativePrompt{}, errors.New("multimodal chat requires add_generation_prompt")
	}
	if len(body.TemplateKwargs) != 0 {
		return nativePrompt{}, errors.New("multimodal chat cannot use chat_template_kwargs")
	}
	message := body.Messages[0]
	if message.Name != "" || message.ReasoningContent != "" ||
		message.ToolCallID != "" || message.ToolResultError ||
		len(message.ToolCalls) != 0 {
		return nativePrompt{}, errors.New("multimodal chat user message contains unsupported fields")
	}
	segments := make([]string, len(message.Media)+1)
	cursor := 0
	for index, media := range message.Media {
		if media.TextOffset < cursor || media.TextOffset > len(message.Content) {
			return nativePrompt{}, errors.New("multimodal chat media offsets are invalid")
		}
		segments[index] = message.Content[cursor:media.TextOffset]
		cursor = media.TextOffset
	}
	segments[len(segments)-1] = message.Content[cursor:]
	prompt := nativePrompt{
		Text:        message.Content,
		Response:    message.Content,
		BeforeMedia: segments[0],
		AfterMedia:  segments[1],
		MediaText:   segments,
	}
	if len(message.Media) > 1 {
		prompt.Images = make([][]byte, len(message.Media))
		for index, media := range message.Media {
			if media.Type != "image" {
				return nativePrompt{}, errors.New("multiple media items must all be images")
			}
			if h.config.ImageProjector == nil {
				return nativePrompt{}, errors.New("multiple images require a multi-image projector")
			}
			if !strings.HasPrefix(media.Data, "data:image/") {
				return nativePrompt{}, errors.New("image_url.url must use a base64 image data URI")
			}
			decoded, err := decodeNativeImageData(media.Data)
			if err != nil {
				return nativePrompt{}, err
			}
			prompt.Images[index] = decoded
		}
		if err := validateMultimodalImages(prompt.Images); err != nil {
			return nativePrompt{}, err
		}
		prompt.Image = prompt.Images[0]
		return prompt, nil
	}
	media := message.Media[0]
	switch media.Type {
	case "image":
		if h.config.ImageProjector == nil && h.config.Qwen3VLProjector == nil {
			return nativePrompt{}, errors.New("image data provided, but the server has no image projector")
		}
		if !strings.HasPrefix(media.Data, "data:image/") {
			return nativePrompt{}, errors.New("image_url.url must use a base64 image data URI")
		}
		decoded, err := decodeNativeImageData(media.Data)
		if err != nil {
			return nativePrompt{}, err
		}
		prompt.Image = decoded
		prompt.Images = [][]byte{decoded}
		if err := validateMultimodalImages(prompt.Images); err != nil {
			return nativePrompt{}, err
		}
	case "audio":
		if h.config.AudioProjector == nil {
			return nativePrompt{}, errors.New("audio data provided, but the server has no audio projector")
		}
		if !strings.EqualFold(media.Format, "wav") {
			return nativePrompt{}, errors.New("input_audio.format must be wav")
		}
		decoded, err := decodeNativeAudioData("data:audio/wav;base64," + media.Data)
		if err != nil {
			return nativePrompt{}, err
		}
		prompt.Audio = decoded
	default:
		return nativePrompt{}, fmt.Errorf("unsupported multimodal chat media type %q", media.Type)
	}
	return prompt, nil
}

const chatMediaMarkerPrefix = "<__llamacpp2go_media_"

func (h *Handler) parseChatMultimodalPrompt(
	formatter ChatFormatter,
	body chatCompletionRequest,
) (nativePrompt, error) {
	if len(body.Messages) == 1 {
		return h.parseChatSingleMultimodalPrompt(body)
	}
	if body.N != 1 {
		return nativePrompt{}, errors.New("multimodal chat requires n=1")
	}
	mediaCount := chatMediaCount(body.Messages)
	if len(body.Messages) == 0 || mediaCount == 0 || mediaCount > 8 {
		return nativePrompt{}, errors.New("multimodal chat requires one to eight media items")
	}
	if len(body.Tools) != 0 || rawJSONConfigured(body.ToolChoice) || body.ParallelTools != nil {
		return nativePrompt{}, errors.New("multimodal chat cannot use tools")
	}
	if body.AddPrompt != nil && !*body.AddPrompt {
		return nativePrompt{}, errors.New("multimodal chat requires add_generation_prompt")
	}
	if len(body.TemplateKwargs) != 0 {
		return nativePrompt{}, errors.New("multimodal chat cannot use chat_template_kwargs")
	}
	messages := append([]inference.ChatMessage(nil), body.Messages...)
	markers := make([]string, 0, mediaCount)
	images := make([][]byte, 0, mediaCount)
	for messageIndex := range messages {
		message := &messages[messageIndex]
		if strings.Contains(message.Content, chatMediaMarkerPrefix) {
			return nativePrompt{}, errors.New("multimodal chat content contains a reserved media marker")
		}
		if len(message.Media) == 0 {
			continue
		}
		if message.Role != "user" {
			return nativePrompt{}, errors.New("multimodal chat media is only supported in user messages")
		}
		if message.Name != "" || message.ReasoningContent != "" ||
			message.ToolCallID != "" || message.ToolResultError || len(message.ToolCalls) != 0 {
			return nativePrompt{}, errors.New("multimodal chat user message contains unsupported fields")
		}
		var content strings.Builder
		cursor := 0
		for _, media := range message.Media {
			if media.TextOffset < cursor || media.TextOffset > len(message.Content) {
				return nativePrompt{}, errors.New("multimodal chat media offsets are invalid")
			}
			if media.Type != "image" {
				return nativePrompt{}, errors.New("multimodal history currently requires image media")
			}
			if h.config.ImageProjector == nil && h.config.Qwen3VLProjector == nil {
				return nativePrompt{}, errors.New("image data provided, but the server has no image projector")
			}
			if !strings.HasPrefix(media.Data, "data:image/") {
				return nativePrompt{}, errors.New("image_url.url must use a base64 image data URI")
			}
			decoded, err := decodeNativeImageData(media.Data)
			if err != nil {
				return nativePrompt{}, err
			}
			content.WriteString(message.Content[cursor:media.TextOffset])
			marker := fmt.Sprintf("%s%08d__>", chatMediaMarkerPrefix, len(markers))
			content.WriteString(marker)
			markers = append(markers, marker)
			images = append(images, decoded)
			cursor = media.TextOffset
		}
		content.WriteString(message.Content[cursor:])
		message.Content = content.String()
		message.Media = nil
	}
	if err := validateMultimodalImages(images); err != nil {
		return nativePrompt{}, err
	}
	formatted, err := formatChatRequest(formatter, messages, nil, body.AddPrompt, nil)
	if err != nil {
		return nativePrompt{}, err
	}
	segments := make([]string, len(markers)+1)
	remainder := formatted
	for index, marker := range markers {
		before, after, found := strings.Cut(remainder, marker)
		if !found {
			return nativePrompt{}, fmt.Errorf("multimodal chat template dropped media marker %d", index)
		}
		segments[index] = before
		remainder = after
	}
	segments[len(segments)-1] = remainder
	return nativePrompt{
		Text: formatted, Response: formatted, Image: images[0], Images: images,
		MediaText: segments, BeforeMedia: segments[0], AfterMedia: segments[1],
		MediaHistory: true,
	}, nil
}

func (h *Handler) normalizeChatPrompt(
	formatter ChatFormatter,
	body chatCompletionRequest,
	promptTools []inference.ChatTool,
) (nativePrompt, error) {
	if chatMediaCount(body.Messages) != 0 {
		return h.parseChatMultimodalPrompt(formatter, body)
	}
	prompt, err := formatChatRequest(
		formatter,
		body.Messages,
		promptTools,
		body.AddPrompt,
		body.TemplateKwargs,
	)
	if err != nil {
		return nativePrompt{}, err
	}
	return nativePrompt{Text: prompt, Response: prompt}, nil
}

func chatThinkingEnabled(kwargs map[string]any) (bool, error) {
	enabled := true
	if value, exists := kwargs["enable_thinking"]; exists {
		var ok bool
		enabled, ok = value.(bool)
		if !ok {
			return false, errors.New(
				"chat_template_kwargs.enable_thinking must be a boolean",
			)
		}
	}
	for key := range kwargs {
		if key != "enable_thinking" {
			return false, fmt.Errorf(
				"unsupported chat_template_kwargs key %q",
				key,
			)
		}
	}
	return enabled, nil
}

func (h *Handler) chatInputTokens(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	formatter, ok := h.generator.(ChatFormatter)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "chat formatting is unavailable")
		return
	}
	_, ok = h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "token counting is unavailable")
		return
	}
	var body chatCompletionRequest
	if !h.decodeMultimodalJSON(response, request, &body) {
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
		return
	}
	if body.N == 0 {
		body.N = 1
	}
	toolSelection, err := selectChatTools(body)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	normalized, err := h.normalizeChatPrompt(formatter, body, toolSelection.prompt)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	prepared, err := h.preparePrompt(request.Context(), normalized, true)
	if err != nil {
		if nativePromptHasMedia(normalized) {
			writeGenerationError(response, err)
		} else {
			writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		}
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{
		"object":       "response.input_tokens",
		"input_tokens": len(prepared.TokenIDs),
	})
}

func (h *Handler) chatCompletions(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	formatter, ok := h.generator.(ChatFormatter)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "chat formatting is unavailable")
		return
	}
	var body chatCompletionRequest
	if !h.decodeMultimodalJSON(response, request, &body) {
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
		return
	}
	if body.N == 0 {
		body.N = 1
	}
	if body.N < 1 || body.N > 8 {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "n must be in [1,8]")
		return
	}
	toolSelection, err := selectChatTools(body)
	if err != nil {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			err.Error(),
		)
		return
	}
	normalizedPrompt, err := h.normalizeChatPrompt(formatter, body, toolSelection.prompt)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if len(toolSelection.active) != 0 {
		if _, ok := h.generator.(ChatOutputParser); !ok {
			writeError(
				response,
				http.StatusNotImplemented,
				"unsupported_operation",
				"tool-call output parsing is unavailable",
			)
			return
		}
	}
	maxTokens := 16
	if body.MaxTokens != nil {
		maxTokens = *body.MaxTokens
	}
	if maxTokens < 0 || maxTokens > h.config.MaxTokens {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			fmt.Sprintf("max_tokens must be in [0,%d]", h.config.MaxTokens),
		)
		return
	}
	stops, err := parseStopSequences(body.Stop)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if err := prepareStructuredOutput(
		&body.samplingParameters,
		body.JSONSchema,
		body.ResponseFormat,
	); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if len(toolSelection.active) != 0 {
		provider, ok := h.generator.(ChatToolGrammarProvider)
		if !ok {
			writeError(
				response,
				http.StatusNotImplemented,
				"unsupported_operation",
				"tool-call grammar generation is unavailable",
			)
			return
		}
		enableThinking, err := chatThinkingEnabled(body.TemplateKwargs)
		if err != nil {
			writeError(
				response,
				http.StatusBadRequest,
				"invalid_request_error",
				err.Error(),
			)
			return
		}
		source, root, patterns, err := provider.ChatToolGrammar(
			toolSelection.active,
			toolSelection.required,
			enableThinking,
			!toolSelection.named &&
				(body.ParallelTools == nil || *body.ParallelTools),
		)
		if err != nil {
			writeError(
				response,
				http.StatusBadRequest,
				"invalid_request_error",
				err.Error(),
			)
			return
		}
		body.Grammar = source
		body.GrammarRoot = root
		body.GrammarLazy = len(patterns) != 0
		body.GrammarTriggerPatterns = patterns
	}
	sampler, err := h.newSampler(body.samplingParameters)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	slotID, acquired := h.acquireSlot(-1)
	if !acquired {
		response.Header().Set("Retry-After", "1")
		writeError(response, http.StatusTooManyRequests, "server_busy", "generation capacity is busy")
		return
	}
	defer h.releaseSlot(slotID)
	prepared, err := h.preparePrompt(request.Context(), normalizedPrompt, true)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	id := "chatcmpl-" + strconv.FormatUint(h.nextID.Add(1), 10)
	if body.Stream {
		h.streamChatCompletion(
			response,
			request,
			slotID,
			prepared.Text,
			sampler,
			maxTokens,
			id,
			stops,
			body.N,
			toolSelection.active,
			prepared.TokenIDs,
			prepared.ProjectedInputs,
		)
		return
	}
	h.completeChat(
		response,
		request,
		slotID,
		prepared.Text,
		sampler,
		maxTokens,
		id,
		stops,
		body.N,
		toolSelection.active,
		prepared.TokenIDs,
		prepared.ProjectedInputs,
	)
}

type chatToolSelection struct {
	prompt   []inference.ChatTool
	active   []inference.ChatTool
	required bool
	named    bool
	parallel bool
}

func selectChatTools(
	body chatCompletionRequest,
) (chatToolSelection, error) {
	if len(body.Tools) == 0 {
		if rawJSONConfigured(body.ToolChoice) {
			var choice string
			if json.Unmarshal(body.ToolChoice, &choice) != nil ||
				choice != "auto" {
				return chatToolSelection{}, errors.New(
					"tool_choice requires at least one tool",
				)
			}
		}
		if body.ParallelTools != nil {
			return chatToolSelection{}, errors.New(
				"parallel_tool_calls requires at least one tool",
			)
		}
		return chatToolSelection{}, nil
	}
	if grammarOptionsConfigured(body.samplingParameters) {
		return chatToolSelection{}, errors.New(
			"custom grammar options cannot be combined with tools",
		)
	}
	if rawJSONConfigured(body.JSONSchema) ||
		rawJSONConfigured(body.ResponseFormat) {
		return chatToolSelection{}, errors.New(
			"response_format and json_schema cannot be combined with tools",
		)
	}
	selection := chatToolSelection{
		prompt: body.Tools,
		active: body.Tools,
	}
	if rawJSONConfigured(body.ToolChoice) {
		var choice string
		if json.Unmarshal(body.ToolChoice, &choice) == nil {
			switch choice {
			case "auto":
			case "none":
				selection.prompt = nil
				selection.active = nil
			case "required":
				selection.required = true
			default:
				return chatToolSelection{}, fmt.Errorf(
					"unsupported tool_choice %q",
					choice,
				)
			}
		} else {
			var named struct {
				Type     string `json:"type"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			}
			decoder := json.NewDecoder(bytes.NewReader(body.ToolChoice))
			decoder.DisallowUnknownFields()
			if err := decoder.Decode(&named); err != nil {
				return chatToolSelection{}, errors.New(
					"tool_choice must be auto, none, required, or a named function",
				)
			}
			if err := requireEOF(decoder); err != nil {
				return chatToolSelection{}, err
			}
			if named.Type != "function" ||
				named.Function.Name == "" {
				return chatToolSelection{}, errors.New(
					"named tool_choice requires a function name",
				)
			}
			var selected *inference.ChatTool
			for index := range body.Tools {
				if body.Tools[index].Function.Name == named.Function.Name {
					selected = &body.Tools[index]
					break
				}
			}
			if selected == nil {
				return chatToolSelection{}, fmt.Errorf(
					"tool_choice references unknown function %q",
					named.Function.Name,
				)
			}
			selection.prompt = []inference.ChatTool{*selected}
			selection.active = []inference.ChatTool{*selected}
			selection.required = true
			selection.named = true
		}
	}
	if body.ParallelTools != nil &&
		*body.ParallelTools &&
		len(selection.active) == 1 &&
		selection.required &&
		len(body.Tools) > 1 {
		return chatToolSelection{}, errors.New(
			"parallel_tool_calls cannot be used with a named tool_choice",
		)
	}
	if len(selection.active) == 0 && body.ParallelTools != nil {
		return chatToolSelection{}, errors.New(
			"parallel_tool_calls cannot be used with tool_choice none",
		)
	}
	return selection, nil
}

type chatChoice struct {
	Index        int                   `json:"index"`
	Message      inference.ChatMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

type chatResponse struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created int64           `json:"created"`
	Model   string          `json:"model"`
	Choices []chatChoice    `json:"choices"`
	Usage   completionUsage `json:"usage"`
}

func (h *Handler) completeChat(
	response http.ResponseWriter,
	request *http.Request,
	slotID int,
	prompt string,
	sampler *sampling.Sampler,
	maxTokens int,
	id string,
	stops []string,
	n int,
	tools []inference.ChatTool,
	promptIDs []tokenizer.TokenID,
	projectedInputs *inference.ProjectedInputs,
) {
	choices := make([]chatChoice, 0, n)
	promptTokens := 0
	totalCompletionTokens := 0
	for choiceIndex := range n {
		choiceSampler, err := samplerForChoice(sampler, choiceIndex)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		var output strings.Builder
		completionTokens := 0
		generatedTokens := 0
		filter := newStopFilter(stops)
		ids, _, err := h.generate(
			request.Context(),
			slotID,
			prompt,
			inference.GenerateOptions{
				MaxNewTokens:    maxTokens,
				Sampler:         choiceSampler,
				ParseSpecial:    true,
				StopSequences:   stops,
				ContextShift:    h.config.ContextShift,
				PromptTokenIDs:  promptIDs,
				CachePrompt:     projectedInputs != nil,
				ProjectedInputs: projectedInputs,
				OnToken: func(event inference.TokenEvent) error {
					generatedTokens++
					if !filter.Stopped() {
						completionTokens++
					}
					output.WriteString(filter.Accept(event.Piece))
					return nil
				},
			},
		)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		if choiceIndex == 0 {
			promptTokens = len(ids) - generatedTokens
		}
		output.WriteString(filter.Flush())
		finishReason := "stop"
		if !filter.Stopped() && completionTokens == maxTokens {
			finishReason = "length"
		}
		totalCompletionTokens += completionTokens
		message := inference.ChatMessage{
			Role:    "assistant",
			Content: output.String(),
		}
		if len(tools) != 0 {
			parser := h.generator.(ChatOutputParser)
			message, err = parser.ParseChatOutput(output.String(), tools)
			if err != nil {
				writeGenerationError(response, err)
				return
			}
			for callIndex := range message.ToolCalls {
				if message.ToolCalls[callIndex].ID == "" {
					message.ToolCalls[callIndex].ID = fmt.Sprintf(
						"call_%s_%d_%d",
						strings.TrimPrefix(id, "chatcmpl-"),
						choiceIndex,
						callIndex,
					)
				}
			}
			if len(message.ToolCalls) != 0 {
				finishReason = "tool_calls"
			}
		}
		choices = append(choices, chatChoice{
			Index:        choiceIndex,
			Message:      message,
			FinishReason: finishReason,
		})
	}
	writeJSON(response, http.StatusOK, chatResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   h.config.ModelID,
		Choices: choices,
		Usage: completionUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: totalCompletionTokens,
			TotalTokens:      promptTokens + totalCompletionTokens,
		},
	})
}

type chatStreamChoice struct {
	Index        int             `json:"index"`
	Delta        chatStreamDelta `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

type chatStreamDelta struct {
	Role             string               `json:"role,omitempty"`
	Content          string               `json:"content,omitempty"`
	ReasoningContent string               `json:"reasoning_content,omitempty"`
	ToolCalls        []chatStreamToolCall `json:"tool_calls,omitempty"`
}

type chatStreamToolCall struct {
	Index    int                    `json:"index"`
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function chatStreamToolFunction `json:"function"`
}

type chatStreamToolFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type chatStreamResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []chatStreamChoice `json:"choices"`
}

func (h *Handler) streamChatCompletion(
	response http.ResponseWriter,
	request *http.Request,
	slotID int,
	prompt string,
	sampler *sampling.Sampler,
	maxTokens int,
	id string,
	stops []string,
	n int,
	tools []inference.ChatTool,
	promptIDs []tokenizer.TokenID,
	projectedInputs *inference.ProjectedInputs,
) {
	flusher, ok := response.(http.Flusher)
	if !ok {
		writeError(response, http.StatusInternalServerError, "server_error", "streaming is unavailable")
		return
	}
	response.Header().Set("Content-Type", "text/event-stream")
	response.Header().Set("Cache-Control", "no-cache")
	response.Header().Set("X-Accel-Buffering", "no")
	response.WriteHeader(http.StatusOK)
	created := time.Now().Unix()
	writeChunk := func(index int, delta chatStreamDelta, reason *string) error {
		return writeSSE(response, chatStreamResponse{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   h.config.ModelID,
			Choices: []chatStreamChoice{{
				Index:        index,
				Delta:        delta,
				FinishReason: reason,
			}},
		})
	}
	for choiceIndex := range n {
		choiceSampler, err := samplerForChoice(sampler, choiceIndex)
		if err != nil {
			_ = writeSSE(response, errorEnvelope("generation_error", err.Error()))
			break
		}
		if err := writeChunk(choiceIndex, chatStreamDelta{Role: "assistant"}, nil); err != nil {
			return
		}
		flusher.Flush()
		completionTokens := 0
		filter := newStopFilter(stops)
		var buffered strings.Builder
		var toolStream inference.ChatOutputStream
		streamedToolCall := false
		if len(tools) != 0 {
			if provider, ok := h.generator.(ChatOutputStreamProvider); ok {
				toolStream, err = provider.NewChatOutputStream(tools)
				if err != nil {
					_ = writeSSE(
						response,
						errorEnvelope("generation_error", err.Error()),
					)
					break
				}
			}
		}
		emitToolPiece := func(piece string) error {
			buffered.WriteString(piece)
			if toolStream == nil || piece == "" {
				return nil
			}
			deltas, streamErr := toolStream.Accept(piece)
			if streamErr != nil {
				return streamErr
			}
			for _, delta := range deltas {
				if delta.ReasoningContent != "" {
					if writeErr := writeChunk(choiceIndex, chatStreamDelta{
						ReasoningContent: delta.ReasoningContent,
					}, nil); writeErr != nil {
						return writeErr
					}
				}
				if delta.Content != "" {
					if writeErr := writeChunk(choiceIndex, chatStreamDelta{
						Content: delta.Content,
					}, nil); writeErr != nil {
						return writeErr
					}
				}
				call := chatStreamToolCall{
					Index: delta.Index,
					Function: chatStreamToolFunction{
						Arguments: delta.Arguments,
					},
				}
				if delta.Started {
					call.ID = fmt.Sprintf(
						"call_%s_%d_%d",
						strings.TrimPrefix(id, "chatcmpl-"),
						choiceIndex,
						delta.Index,
					)
					call.Type = "function"
					call.Function.Name = delta.Name
					streamedToolCall = true
				}
				if writeErr := writeChunk(choiceIndex, chatStreamDelta{
					ToolCalls: []chatStreamToolCall{call},
				}, nil); writeErr != nil {
					return writeErr
				}
				flusher.Flush()
			}
			return nil
		}
		_, _, err = h.generate(
			request.Context(),
			slotID,
			prompt,
			inference.GenerateOptions{
				MaxNewTokens:    maxTokens,
				Sampler:         choiceSampler,
				ParseSpecial:    true,
				StopSequences:   stops,
				ContextShift:    h.config.ContextShift,
				PromptTokenIDs:  promptIDs,
				CachePrompt:     projectedInputs != nil,
				ProjectedInputs: projectedInputs,
				OnToken: func(event inference.TokenEvent) error {
					if !filter.Stopped() {
						completionTokens++
					}
					piece := filter.Accept(event.Piece)
					if piece != "" {
						if len(tools) != 0 {
							if streamErr := emitToolPiece(piece); streamErr != nil {
								return streamErr
							}
						} else {
							if writeErr := writeChunk(choiceIndex, chatStreamDelta{Content: piece}, nil); writeErr != nil {
								return writeErr
							}
							flusher.Flush()
						}
					}
					return request.Context().Err()
				},
			},
		)
		if err != nil {
			_ = writeSSE(response, errorEnvelope("generation_error", err.Error()))
			break
		}
		if piece := filter.Flush(); piece != "" {
			if len(tools) != 0 {
				if streamErr := emitToolPiece(piece); streamErr != nil {
					_ = writeSSE(
						response,
						errorEnvelope("generation_error", streamErr.Error()),
					)
					break
				}
			} else {
				_ = writeChunk(choiceIndex, chatStreamDelta{Content: piece}, nil)
				flusher.Flush()
			}
		}
		reason := "stop"
		if !filter.Stopped() && completionTokens == maxTokens {
			reason = "length"
		}
		if len(tools) != 0 {
			parser := h.generator.(ChatOutputParser)
			message, parseErr := parser.ParseChatOutput(buffered.String(), tools)
			if parseErr != nil {
				_ = writeSSE(
					response,
					errorEnvelope("generation_error", parseErr.Error()),
				)
				break
			}
			if message.ReasoningContent != "" && !streamedToolCall {
				if err := writeChunk(choiceIndex, chatStreamDelta{
					ReasoningContent: message.ReasoningContent,
				}, nil); err != nil {
					return
				}
				flusher.Flush()
			}
			if message.Content != "" && !streamedToolCall {
				if err := writeChunk(choiceIndex, chatStreamDelta{
					Content: message.Content,
				}, nil); err != nil {
					return
				}
				flusher.Flush()
			}
			for callIndex, call := range message.ToolCalls {
				if streamedToolCall {
					break
				}
				callID := call.ID
				if callID == "" {
					callID = fmt.Sprintf(
						"call_%s_%d_%d",
						strings.TrimPrefix(id, "chatcmpl-"),
						choiceIndex,
						callIndex,
					)
				}
				if err := writeChunk(choiceIndex, chatStreamDelta{
					ToolCalls: []chatStreamToolCall{{
						Index: callIndex,
						ID:    callID,
						Type:  call.Type,
						Function: chatStreamToolFunction{
							Name:      call.Function.Name,
							Arguments: call.Function.Arguments,
						},
					}},
				}, nil); err != nil {
					return
				}
				flusher.Flush()
			}
			if len(message.ToolCalls) != 0 {
				reason = "tool_calls"
			}
		}
		_ = writeChunk(choiceIndex, chatStreamDelta{}, &reason)
		flusher.Flush()
	}
	_, _ = io.WriteString(response, "data: [DONE]\n\n")
	flusher.Flush()
}
