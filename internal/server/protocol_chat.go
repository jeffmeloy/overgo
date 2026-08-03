package server

import (
	"context"
	"slices"

	"encoding/json"

	"errors"

	"fmt"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/projector"
	"llamacpp2go/internal/strictjson"

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
	ctx context.Context,
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
	thinking, err := chatThinkingEnabled(body.TemplateKwargs)
	if err != nil {
		return nativePrompt{}, err
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
		Thinking:    &thinking,
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
			decoded, err := h.resolveImageData(ctx, media.Data)
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
		decoded, err := h.resolveImageData(ctx, media.Data)
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
		decoded, _, err := h.resolveAudioData(ctx, media.Data, media.Format)
		if err != nil {
			return nativePrompt{}, err
		}
		prompt.Audio = decoded
	case "video":
		if _, ok := h.config.ImageProjector.(projector.VideoProjector); !ok {
			return nativePrompt{}, errors.New("video data provided, but the server has no video projector")
		}
		decoded, err := h.resolveVideoData(ctx, media.Data)
		if err != nil {
			return nativePrompt{}, err
		}
		prompt.Video = decoded
		prompt.VideoFPS = media.FPS
	default:
		return nativePrompt{}, fmt.Errorf("unsupported multimodal chat media type %q", media.Type)
	}
	return prompt, nil
}

const chatMediaMarkerPrefix = "<__llamacpp2go_media_"

func (h *Handler) parseChatMultimodalPrompt(
	ctx context.Context,
	formatter ChatFormatter,
	body chatCompletionRequest,
	promptTools []inference.ChatTool,
) (nativePrompt, error) {
	hasToolConfig := len(body.Tools) != 0 || rawJSONConfigured(body.ToolChoice) || body.ParallelTools != nil
	if len(body.Messages) == 1 && chatSingleMediaCompatible(body.Messages[0].Media) &&
		!hasToolConfig && len(promptTools) == 0 {
		return h.parseChatSingleMultimodalPrompt(ctx, body)
	}
	if body.N != 1 {
		return nativePrompt{}, errors.New("multimodal chat requires n=1")
	}
	mediaCount := chatMediaCount(body.Messages)
	if len(body.Messages) == 0 || mediaCount == 0 || mediaCount > 8 {
		return nativePrompt{}, errors.New("multimodal chat requires one to eight media items")
	}
	if body.AddPrompt != nil && !*body.AddPrompt {
		return nativePrompt{}, errors.New("multimodal chat requires add_generation_prompt")
	}
	messages := slices.Clone(body.Messages)
	markers := make([]string, 0, mediaCount)
	images := make([][]byte, 0, mediaCount)
	mediaInputs := make([]nativeMedia, 0, mediaCount)
	hasAudio := false
	totalMediaBytes := 0
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
			content.WriteString(message.Content[cursor:media.TextOffset])
			marker := fmt.Sprintf("%s%08d__>", chatMediaMarkerPrefix, len(markers))
			content.WriteString(marker)
			markers = append(markers, marker)
			switch media.Type {
			case "image":
				if h.config.ImageProjector == nil && h.config.Qwen3VLProjector == nil {
					return nativePrompt{}, errors.New("image data provided, but the server has no image projector")
				}
				decoded, err := h.resolveImageData(ctx, media.Data)
				if err != nil {
					return nativePrompt{}, err
				}
				images = append(images, decoded)
				mediaInputs = append(mediaInputs, nativeMedia{Kind: projector.MediaImage, Image: decoded})
				totalMediaBytes += len(decoded)
			case "audio":
				if h.config.AudioProjector == nil {
					return nativePrompt{}, errors.New("audio data provided, but the server has no audio projector")
				}
				decoded, mediaBytes, err := h.resolveAudioData(ctx, media.Data, media.Format)
				if err != nil {
					return nativePrompt{}, err
				}
				hasAudio = true
				mediaInputs = append(mediaInputs, nativeMedia{Kind: projector.MediaAudio, Audio: decoded})
				totalMediaBytes += mediaBytes
			default:
				return nativePrompt{}, fmt.Errorf("unsupported multimodal chat media type %q", media.Type)
			}
			if totalMediaBytes > maxMediaBytes {
				return nativePrompt{}, errors.New("multimodal media exceeds aggregate byte limit")
			}
			cursor = media.TextOffset
		}
		content.WriteString(message.Content[cursor:])
		message.Content = content.String()
		message.Media = nil
	}
	if err := validateMultimodalImages(images); err != nil {
		return nativePrompt{}, err
	}
	formatted, err := formatChatRequest(
		formatter,
		messages,
		promptTools,
		body.AddPrompt,
		body.TemplateKwargs,
	)
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
	prompt := nativePrompt{
		Text: formatted, Response: formatted, Images: images,
		MediaText: segments, BeforeMedia: segments[0], AfterMedia: segments[1],
		MediaHistory: true,
	}
	if len(images) != 0 {
		prompt.Image = images[0]
	}
	if hasAudio {
		prompt.Image = nil
		prompt.Images = nil
		prompt.Media = mediaInputs
	}
	return prompt, nil
}

func chatSingleMediaCompatible(media []inference.ChatMediaPart) bool {
	if len(media) == 1 {
		return true
	}
	for _, item := range media {
		if item.Type != "image" {
			return false
		}
	}
	return true
}

func (h *Handler) normalizeChatPrompt(
	ctx context.Context,
	formatter ChatFormatter,
	body chatCompletionRequest,
	promptTools []inference.ChatTool,
) (nativePrompt, error) {
	if chatMediaCount(body.Messages) != 0 {
		return h.parseChatMultimodalPrompt(ctx, formatter, body, promptTools)
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
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	formatter, ok := h.requireChatFormatter(response)
	if !ok {
		return
	}
	if !h.requireTokenCounting(response) {
		return
	}
	var body chatCompletionRequest
	if !h.decodeMultimodalJSON(response, request, &body) {
		return
	}
	if !h.requireModel(response, body.Model) {
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
	normalized, err := h.normalizeChatPrompt(request.Context(), formatter, body, toolSelection.prompt)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	h.writeProtocolInputTokenCount(response, request, normalized, true)
}

func (h *Handler) chatCompletions(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	formatter, ok := h.requireChatFormatter(response)
	if !ok {
		return
	}
	var body chatCompletionRequest
	if !h.decodeMultimodalJSON(response, request, &body) {
		return
	}
	if !h.requireModel(response, body.Model) {
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
	normalizedPrompt, err := h.normalizeChatPrompt(request.Context(), formatter, body, toolSelection.prompt)
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
	maxTokens, err := boundedProtocolTokens(
		body.MaxTokens, 16, h.config.MaxTokens, "max_tokens", false,
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
		if !h.configureChatToolGrammar(
			response,
			&body.samplingParameters,
			toolSelection.active,
			toolSelection.required,
			enableThinking,
			!toolSelection.named &&
				(body.ParallelTools == nil || *body.ParallelTools),
		) {
			return
		}
	}
	sampler, err := h.newSampler(body.samplingParameters)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	prepared, slotID, ok := h.prepareProtocolGeneration(response, request, normalizedPrompt)
	if !ok {
		return
	}
	defer h.releaseSlot(slotID)
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
			if err := strictjson.DecodeBytes(body.ToolChoice, &named); err != nil {
				return chatToolSelection{}, errors.New(
					"tool_choice must be auto, none, required, or a named function",
				)
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
		ids, pump, err := h.generateWithPump(
			request.Context(),
			slotID,
			prompt,
			h.protocolGenerationOptions(maxTokens, choiceSampler, promptIDs, projectedInputs),
			stops,
			nil,
		)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		if choiceIndex == 0 {
			promptTokens = len(ids) - pump.generated
		}
		finishReason := pump.finishReason(maxTokens, "stop", "length")
		totalCompletionTokens += pump.completion
		message := inference.ChatMessage{
			Role:    "assistant",
			Content: pump.text(),
		}
		if len(tools) != 0 {
			parser := h.generator.(ChatOutputParser)
			message, err = parser.ParseChatOutput(pump.text(), tools)
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
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSSEEmitter(request.Context(), response, flusher)
	created := time.Now().Unix()
	writeChunk := func(index int, delta chatStreamDelta, reason *string) error {
		return stream.write(chatStreamResponse{
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
			_ = stream.write(errorEnvelope("generation_error", err.Error()))
			break
		}
		if err := writeChunk(choiceIndex, chatStreamDelta{Role: "assistant"}, nil); err != nil {
			return
		}
		toolStream, err := newToolDeltaStream(h.generator, tools)
		if err != nil {
			_ = stream.write(errorEnvelope("generation_error", err.Error()))
			break
		}
		emitToolPiece := func(piece string) error {
			deltas, streamErr := toolStream.accept(piece)
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
				}
				if writeErr := writeChunk(choiceIndex, chatStreamDelta{
					ToolCalls: []chatStreamToolCall{call},
				}, nil); writeErr != nil {
					return writeErr
				}
			}
			return nil
		}
		_, pump, err := h.generateWithPump(
			request.Context(),
			slotID,
			prompt,
			h.protocolGenerationOptions(maxTokens, choiceSampler, promptIDs, projectedInputs),
			stops,
			func(piece string) error {
				if piece != "" {
					if len(tools) != 0 {
						if streamErr := emitToolPiece(piece); streamErr != nil {
							return streamErr
						}
					} else {
						if writeErr := writeChunk(choiceIndex, chatStreamDelta{Content: piece}, nil); writeErr != nil {
							return writeErr
						}
					}
				}
				return request.Context().Err()
			},
		)
		if err != nil {
			_ = stream.write(errorEnvelope("generation_error", err.Error()))
			break
		}
		reason := pump.finishReason(maxTokens, "stop", "length")
		if len(tools) != 0 {
			parser := h.generator.(ChatOutputParser)
			message, parseErr := parser.ParseChatOutput(toolStream.text(), tools)
			if parseErr != nil {
				_ = stream.write(
					errorEnvelope("generation_error", parseErr.Error()),
				)
				break
			}
			if message.ReasoningContent != "" && !toolStream.started {
				if err := writeChunk(choiceIndex, chatStreamDelta{
					ReasoningContent: message.ReasoningContent,
				}, nil); err != nil {
					return
				}
			}
			if message.Content != "" && !toolStream.started {
				if err := writeChunk(choiceIndex, chatStreamDelta{
					Content: message.Content,
				}, nil); err != nil {
					return
				}
			}
			for callIndex, call := range message.ToolCalls {
				if toolStream.started {
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
			}
			if len(message.ToolCalls) != 0 {
				reason = "tool_calls"
			}
		}
		_ = writeChunk(choiceIndex, chatStreamDelta{}, &reason)
	}
	_ = stream.done()
}
