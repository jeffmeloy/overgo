package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"path"
	"strings"
	"unicode/utf8"

	"overgo/internal/inference"
	"overgo/internal/strictjson"
)

func selectResponsesTools(
	rawTools, rawChoice json.RawMessage,
	parallelTools *bool,
	policy ResponseToolPolicy,
) (chatToolSelection, error) {
	var tools []inference.ChatTool
	if strictjson.HasValue(rawTools) {
		var rawDefinitions []json.RawMessage
		if err := json.Unmarshal(rawTools, &rawDefinitions); err != nil {
			return chatToolSelection{}, errors.New("tools must be an array of Responses tool definitions")
		}
		for index, rawDefinition := range rawDefinitions {
			var header struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(rawDefinition, &header); err != nil || header.Type == "" {
				return chatToolSelection{}, fmt.Errorf("tool %d requires a type", index)
			}
			if header.Type == "function" {
				continue
			}
			category := "hosted"
			if header.Type == "custom" {
				category = "custom"
			}
			mode := policy.Hosted
			if category == "custom" {
				mode = policy.Custom
			}
			if mode == "" {
				mode = "deny"
			}
			return chatToolSelection{}, fmt.Errorf(
				"tool %d type %q is %s by response_tools.%s policy; an external executor is required",
				index, header.Type, mode, category,
			)
		}
		var definitions []struct {
			Type        string         `json:"type"`
			Name        string         `json:"name"`
			Description string         `json:"description"`
			Parameters  map[string]any `json:"parameters"`
			Strict      *bool          `json:"strict"`
		}
		if err := strictjson.DecodeBytes(rawTools, &definitions); err != nil {
			return chatToolSelection{}, errors.New(
				"tools must be an array of Responses function definitions",
			)
		}
		if err := validateProtocolToolCount(len(definitions)); err != nil {
			return chatToolSelection{}, err
		}
		tools = make([]inference.ChatTool, len(definitions))
		for index, definition := range definitions {
			if definition.Type != "function" {
				return chatToolSelection{}, fmt.Errorf(
					"tool %d has unsupported type %q",
					index,
					definition.Type,
				)
			}
			if definition.Name == "" {
				return chatToolSelection{}, fmt.Errorf(
					"tool %d name is required",
					index,
				)
			}
			if definition.Parameters == nil {
				return chatToolSelection{}, fmt.Errorf(
					"tool %d parameters are required",
					index,
				)
			}
			tools[index] = inference.ChatTool{
				Type: inference.ChatToolTypeFunction,
				Function: inference.ChatToolDefinition{
					Name:        definition.Name,
					Description: definition.Description,
					Parameters:  definition.Parameters,
				},
			}
		}
	}

	var openAIChoice json.RawMessage
	if strictjson.HasValue(rawChoice) {
		var mode string
		if json.Unmarshal(rawChoice, &mode) == nil {
			openAIChoice = rawChoice
		} else {
			var named struct {
				Type string `json:"type"`
				Name string `json:"name"`
			}
			if err := strictjson.DecodeBytes(rawChoice, &named); err != nil {
				return chatToolSelection{}, errors.New(
					"tool_choice must be auto, none, required, or a named function",
				)
			}
			if named.Type != "function" || named.Name == "" {
				return chatToolSelection{}, errors.New(
					"named tool_choice requires type function and a name",
				)
			}
			encoded, err := json.Marshal(map[string]any{
				"type": "function",
				"function": map[string]string{
					"name": named.Name,
				},
			})
			if err != nil {
				return chatToolSelection{}, err
			}
			openAIChoice = encoded
		}
	}
	return selectChatTools(chatCompletionRequest{
		Tools:         tools,
		ToolChoice:    openAIChoice,
		ParallelTools: parallelTools,
	})
}

func (h *Handler) parseResponsesMessages(
	ctx context.Context,
	raw json.RawMessage,
	instructions string,
) ([]inference.ChatMessage, error) {
	if len(raw) == 0 {
		return nil, errors.New("input is required")
	}
	messages := make([]inference.ChatMessage, 0, 4)
	if instructions != "" {
		messages = append(messages, inference.ChatMessage{
			Role:    inference.ChatRoleSystem,
			Content: instructions,
		})
	}
	var textInput string
	if err := json.Unmarshal(raw, &textInput); err == nil {
		return append(messages, inference.ChatMessage{
			Role:    inference.ChatRoleUser,
			Content: textInput,
		}), nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, errors.New("input must be a string or message array")
	}
	if len(items) == 0 {
		return nil, errors.New("input message array must not be empty")
	}
	if len(items) > 1024 {
		return nil, errors.New("input message count exceeds 1024")
	}
	for index, rawItem := range items {
		var header struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(rawItem, &header); err != nil {
			return nil, fmt.Errorf("input item %d: %w", index, err)
		}
		switch header.Type {
		case "", "message":
			var item struct {
				Content json.RawMessage    `json:"content"`
				ID      string             `json:"id"`
				Role    inference.ChatRole `json:"role"`
				Status  string             `json:"status"`
				Type    string             `json:"type"`
			}
			if err := strictjson.DecodeBytes(rawItem, &item); err != nil {
				return nil, fmt.Errorf("input item %d: %w", index, err)
			}
			if !item.Role.Valid() || len(item.Content) == 0 {
				return nil, fmt.Errorf(
					"input item %d requires role and content",
					index,
				)
			}
			content, media, err := h.parseResponsesMessageContent(
				ctx,
				item.Content,
				fmt.Sprintf("input item %d content", index),
			)
			if err != nil {
				return nil, err
			}
			messages = append(messages, inference.ChatMessage{
				Role: item.Role, Content: content, Media: media,
			})
		case "input_file":
			content, err := h.parseResponsesFile(ctx, rawItem, fmt.Sprintf("input item %d", index))
			if err != nil {
				return nil, err
			}
			messages = append(messages, inference.ChatMessage{Role: inference.ChatRoleUser, Content: content})
		case "function_call":
			var item struct {
				Arguments string `json:"arguments"`
				CallID    string `json:"call_id"`
				ID        string `json:"id"`
				Name      string `json:"name"`
				Status    string `json:"status"`
				Type      string `json:"type"`
			}
			if err := strictjson.DecodeBytes(rawItem, &item); err != nil {
				return nil, fmt.Errorf("input item %d: %w", index, err)
			}
			if item.CallID == "" || item.Name == "" ||
				!json.Valid([]byte(item.Arguments)) {
				return nil, fmt.Errorf(
					"input item %d has invalid function_call fields",
					index,
				)
			}
			call := inference.ChatToolCall{
				ID:   item.CallID,
				Type: inference.ChatToolTypeFunction,
				Function: inference.ChatToolFunction{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			}
			if len(messages) != 0 &&
				messages[len(messages)-1].Role == inference.ChatRoleAssistant {
				last := &messages[len(messages)-1]
				last.ToolCalls = append(last.ToolCalls, call)
			} else {
				messages = append(messages, inference.ChatMessage{
					Role:      inference.ChatRoleAssistant,
					ToolCalls: []inference.ChatToolCall{call},
				})
			}
		case "function_call_output":
			var item struct {
				CallID string          `json:"call_id"`
				ID     string          `json:"id"`
				Output json.RawMessage `json:"output"`
				Status string          `json:"status"`
				Type   string          `json:"type"`
			}
			if err := strictjson.DecodeBytes(rawItem, &item); err != nil {
				return nil, fmt.Errorf("input item %d: %w", index, err)
			}
			if item.CallID == "" || len(item.Output) == 0 {
				return nil, fmt.Errorf(
					"input item %d has invalid function_call_output fields",
					index,
				)
			}
			output, err := parseResponsesTextContent(
				item.Output,
				fmt.Sprintf("input item %d output", index),
			)
			if err != nil {
				return nil, err
			}
			messages = append(messages, inference.ChatMessage{
				Role:       inference.ChatRoleTool,
				Content:    output,
				ToolCallID: item.CallID,
			})
		case "reasoning":
			var item struct {
				ID      string                     `json:"id"`
				Type    string                     `json:"type"`
				Summary []responseReasoningSummary `json:"summary"`
				Content []struct {
					Text string `json:"text"`
					Type string `json:"type"`
				} `json:"content"`
				EncryptedContent string `json:"encrypted_content"`
			}
			if err := strictjson.DecodeBytes(rawItem, &item); err != nil {
				return nil, fmt.Errorf("input item %d: %w", index, err)
			}
			parts := make([]string, 0, len(item.Content)+len(item.Summary))
			for partIndex, part := range item.Content {
				if part.Type != "reasoning_text" || part.Text == "" {
					return nil, fmt.Errorf("input item %d reasoning content %d is invalid", index, partIndex)
				}
				parts = append(parts, part.Text)
			}
			if len(parts) == 0 {
				for partIndex, part := range item.Summary {
					if part.Type != "summary_text" || part.Text == "" {
						return nil, fmt.Errorf("input item %d reasoning summary %d is invalid", index, partIndex)
					}
					parts = append(parts, part.Text)
				}
			}
			if len(parts) == 0 {
				if item.EncryptedContent != "" {
					return nil, fmt.Errorf("input item %d encrypted reasoning is unavailable in the local runtime", index)
				}
				return nil, fmt.Errorf("input item %d reasoning content is empty", index)
			}
			messages = append(messages, inference.ChatMessage{
				Role: inference.ChatRoleAssistant, ReasoningContent: strings.Join(parts, "\n\n"),
			})
		default:
			return nil, fmt.Errorf(
				"input item %d has unsupported type %q",
				index,
				header.Type,
			)
		}
	}
	return messages, nil
}

func (h *Handler) parseResponsesMessageContent(
	ctx context.Context,
	raw json.RawMessage,
	label string,
) (string, []inference.ChatMediaPart, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil, nil
	}
	var rawParts []json.RawMessage
	if err := json.Unmarshal(raw, &rawParts); err != nil {
		return "", nil, fmt.Errorf("%s must be a string or content-part array", label)
	}
	var result strings.Builder
	var media []inference.ChatMediaPart
	for index, rawPart := range rawParts {
		var header struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(rawPart, &header); err != nil {
			return "", nil, fmt.Errorf("%s part %d: %w", label, index, err)
		}
		switch header.Type {
		case "text", "input_text", "output_text":
			text, err := decodeResponsesTextPart(rawPart, label, index)
			if err != nil {
				return "", nil, err
			}
			result.WriteString(text)
		case "input_image":
			var part struct {
				Type     string `json:"type"`
				ImageURL string `json:"image_url"`
				FileID   string `json:"file_id"`
				Detail   string `json:"detail"`
			}
			if err := strictjson.DecodeBytes(rawPart, &part); err != nil {
				return "", nil, fmt.Errorf("%s part %d: %w", label, index, err)
			}
			if (part.ImageURL == "") == (part.FileID == "") {
				return "", nil, fmt.Errorf("%s part %d requires exactly one of image_url or file_id", label, index)
			}
			source := part.ImageURL
			if part.FileID != "" {
				file, ok := h.resolveResponseFile(part.FileID)
				if !ok {
					return "", nil, fmt.Errorf("%s part %d file_id %q is unavailable", label, index, part.FileID)
				}
				if !strings.HasPrefix(file.MediaType, "image/") {
					return "", nil, fmt.Errorf("%s part %d file_id %q is not an image", label, index, part.FileID)
				}
				source = "data:" + file.MediaType + ";base64," + base64.StdEncoding.EncodeToString(file.Data)
			}
			media = append(media, inference.ChatMediaPart{
				Type: "image", Data: source, TextOffset: result.Len(),
			})
		case "input_file":
			content, err := h.parseResponsesFile(ctx, rawPart, fmt.Sprintf("%s part %d", label, index))
			if err != nil {
				return "", nil, err
			}
			result.WriteString(content)
		case "input_audio":
			var part struct {
				Type       string `json:"type"`
				InputAudio struct {
					Data   string `json:"data"`
					URL    string `json:"url"`
					Format string `json:"format"`
				} `json:"input_audio"`
			}
			if err := strictjson.DecodeBytes(rawPart, &part); err != nil {
				return "", nil, fmt.Errorf("%s part %d: %w", label, index, err)
			}
			if (part.InputAudio.Data == "") == (part.InputAudio.URL == "") {
				return "", nil, fmt.Errorf("%s part %d input_audio requires exactly one of data or url", label, index)
			}
			if part.InputAudio.Data != "" && part.InputAudio.Format == "" {
				return "", nil, fmt.Errorf("%s part %d input_audio data requires format", label, index)
			}
			source := part.InputAudio.Data
			if source == "" {
				source = part.InputAudio.URL
			}
			media = append(media, inference.ChatMediaPart{
				Type: "audio", Data: source, Format: part.InputAudio.Format, TextOffset: result.Len(),
			})
		case "input_video":
			var part struct {
				Type       string `json:"type"`
				InputVideo struct {
					Data string  `json:"data"`
					URL  string  `json:"url"`
					FPS  float64 `json:"fps,omitempty"`
				} `json:"input_video"`
			}
			if err := strictjson.DecodeBytes(rawPart, &part); err != nil {
				return "", nil, fmt.Errorf("%s part %d: %w", label, index, err)
			}
			if (part.InputVideo.Data == "") == (part.InputVideo.URL == "") {
				return "", nil, fmt.Errorf("%s part %d input_video requires exactly one of data or url", label, index)
			}
			source := part.InputVideo.Data
			if source == "" {
				source = part.InputVideo.URL
			}
			media = append(media, inference.ChatMediaPart{
				Type: "video", Data: source, FPS: part.InputVideo.FPS, TextOffset: result.Len(),
			})
		default:
			return "", nil, fmt.Errorf(
				"%s part %d has unsupported type %q",
				label, index, header.Type,
			)
		}
	}
	return result.String(), media, nil
}

func (h *Handler) parseResponsesFile(ctx context.Context, raw json.RawMessage, label string) (string, error) {
	var part struct {
		Type     string `json:"type"`
		FileData string `json:"file_data"`
		FileID   string `json:"file_id"`
		FileURL  string `json:"file_url"`
		Filename string `json:"filename"`
	}
	if err := strictjson.DecodeBytes(raw, &part); err != nil {
		return "", fmt.Errorf("%s: %w", label, err)
	}
	if part.Type != "input_file" {
		return "", fmt.Errorf("%s has invalid file type %q", label, part.Type)
	}
	sources := 0
	for _, source := range []string{part.FileData, part.FileID, part.FileURL} {
		if strings.TrimSpace(source) != "" {
			sources++
		}
	}
	if sources != 1 {
		return "", fmt.Errorf("%s requires exactly one of file_data, file_id, or file_url", label)
	}
	var file ResponseFile
	var err error
	switch {
	case part.FileID != "":
		var ok bool
		file, ok = h.resolveResponseFile(part.FileID)
		if !ok {
			return "", fmt.Errorf("%s file_id %q is unavailable", label, part.FileID)
		}
	case part.FileURL != "":
		if h.mediaFetcher == nil {
			return "", fmt.Errorf("%s file_url is disabled", label)
		}
		file.Data, file.MediaType, err = h.mediaFetcher.fetchDocument(ctx, part.FileURL)
		if err != nil {
			return "", fmt.Errorf("%s file_url: %w", label, err)
		}
		if target, parseErr := url.Parse(part.FileURL); parseErr == nil {
			file.Filename = path.Base(target.Path)
		}
	default:
		file, err = decodeResponsesFileData(part.FileData)
		if err != nil {
			return "", fmt.Errorf("%s file_data: %w", label, err)
		}
	}
	if part.Filename != "" {
		file.Filename = part.Filename
	}
	filename, err := validResponseFilename(file.Filename)
	if err != nil {
		return "", fmt.Errorf("%s filename: %w", label, err)
	}
	if !supportedResponseTextType(file.MediaType) {
		return "", fmt.Errorf("%s media type %q is not textual", label, file.MediaType)
	}
	if !utf8.Valid(file.Data) || bytes.IndexByte(file.Data, 0) >= 0 {
		return "", fmt.Errorf("%s content must be UTF-8 text without NUL bytes", label)
	}
	return "[file name=" + fmt.Sprintf("%q", filename) + "]\n" + string(file.Data) + "\n[/file]", nil
}

func (h *Handler) resolveResponseFile(id string) (ResponseFile, bool) {
	if h == nil || h.responseFiles == nil || !responseFileIDPattern.MatchString(id) {
		return ResponseFile{}, false
	}
	file, ok := h.responseFiles.ResolveResponseFile(id)
	file.MediaType = strings.ToLower(strings.TrimSpace(file.MediaType))
	if !ok || len(file.Data) == 0 || len(file.Data) > maxMediaBytes ||
		!supportedResponseFileType(file.MediaType) {
		return ResponseFile{}, false
	}
	return file, true
}

func decodeResponsesFileData(source string) (ResponseFile, error) {
	header, payload, ok := strings.Cut(strings.TrimSpace(source), ",")
	if !ok || !strings.HasPrefix(strings.ToLower(header), "data:") ||
		!strings.HasSuffix(strings.ToLower(header), ";base64") {
		return ResponseFile{}, errors.New("must be a typed base64 data URI")
	}
	mediaType := strings.ToLower(strings.TrimSpace(strings.TrimSuffix(header[5:], ";base64")))
	if !supportedResponseTextType(mediaType) {
		return ResponseFile{}, fmt.Errorf("media type %q is unsupported", mediaType)
	}
	if base64.StdEncoding.DecodedLen(len(payload)) > maxRequestBytes+2 {
		return ResponseFile{}, errors.New("decoded content exceeds inline byte limit")
	}
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil || len(data) == 0 {
		return ResponseFile{}, errors.New("content is empty or invalid base64")
	}
	return ResponseFile{Data: data, Filename: "inline", MediaType: mediaType}, nil
}

func validResponseFilename(filename string) (string, error) {
	filename = strings.TrimSpace(filename)
	if filename == "" {
		filename = "input"
	}
	if len(filename) > 255 || filename == "." || filename == ".." ||
		strings.ContainsAny(filename, "/\\\x00\r\n") {
		return "", errors.New("must be a single safe path component")
	}
	for _, character := range filename {
		if character < 0x20 || character == 0x7f {
			return "", errors.New("contains a control character")
		}
	}
	return filename, nil
}

func parseResponsesTextContent(raw json.RawMessage, label string) (string, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}
	var rawParts []json.RawMessage
	if err := json.Unmarshal(raw, &rawParts); err != nil {
		return "", fmt.Errorf("%s must be a string or text-part array", label)
	}
	var result strings.Builder
	for index, rawPart := range rawParts {
		text, err := decodeResponsesTextPart(rawPart, label, index)
		if err != nil {
			return "", err
		}
		result.WriteString(text)
	}
	return result.String(), nil
}

func decodeResponsesTextPart(raw json.RawMessage, label string, index int) (string, error) {
	var part struct {
		Type        string          `json:"type"`
		Text        string          `json:"text"`
		Annotations json.RawMessage `json:"annotations"`
		Logprobs    json.RawMessage `json:"logprobs"`
	}
	if err := strictjson.DecodeBytes(raw, &part); err != nil {
		return "", fmt.Errorf("%s part %d: %w", label, index, err)
	}
	switch part.Type {
	case "text", "input_text", "output_text":
		return part.Text, nil
	default:
		return "", fmt.Errorf("%s part %d has unsupported type %q", label, index, part.Type)
	}
}

func responseItems(
	message inference.ChatMessage,
	messageID, idSuffix string,
	includeReasoning bool,
) []responseOutputItem {
	items := make([]responseOutputItem, 0, 2+len(message.ToolCalls))
	if includeReasoning && message.ReasoningContent != "" {
		items = append(items, responseOutputItem{
			ID:     "rs_" + idSuffix,
			Status: "completed",
			Summary: []responseReasoningSummary{{
				Text: message.ReasoningContent, Type: "summary_text",
			}},
			Type: "reasoning",
		})
	}
	if message.Content != "" {
		items = append(items, responseOutputItem{
			Content: []responseOutputText{{
				Type:        "output_text",
				Annotations: []any{},
				Logprobs:    []any{},
				Text:        message.Content,
			}},
			ID:     messageID,
			Role:   inference.ChatRoleAssistant,
			Status: "completed",
			Type:   "message",
		})
	}
	for index, call := range message.ToolCalls {
		callID := call.ID
		if callID == "" {
			callID = fmt.Sprintf("call_%s_%d", idSuffix, index)
		}
		items = append(items, responseOutputItem{
			Arguments: call.Function.Arguments,
			CallID:    callID,
			ID:        fmt.Sprintf("fc_%s_%d", idSuffix, index),
			Name:      call.Function.Name,
			Status:    "completed",
			Type:      "function_call",
		})
	}
	return items
}

func assignResponseCallIDs(message *inference.ChatMessage, idSuffix string) {
	for index := range message.ToolCalls {
		if message.ToolCalls[index].ID == "" {
			message.ToolCalls[index].ID = fmt.Sprintf("call_%s_%d", idSuffix, index)
		}
	}
}
