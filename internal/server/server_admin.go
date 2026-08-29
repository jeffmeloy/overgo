package server

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"overgo/internal/cuda/driver"
	"overgo/internal/inference"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
	"overgo/internal/strictjson"
	"overgo/internal/tokenizer"
)

type propertiesSamplingParams struct {
	NPredict         int                     `json:"n_predict"`
	Seed             int64                   `json:"seed"`
	Temperature      float32                 `json:"temperature"`
	DynatempRange    float32                 `json:"dynatemp_range"`
	DynatempExponent float32                 `json:"dynatemp_exponent"`
	TopK             int                     `json:"top_k"`
	TopP             float32                 `json:"top_p"`
	MinP             float32                 `json:"min_p"`
	TopNSigma        float32                 `json:"top_n_sigma"`
	XTCProbability   float32                 `json:"xtc_probability"`
	XTCThreshold     float32                 `json:"xtc_threshold"`
	TypicalP         float32                 `json:"typical_p"`
	RepeatLastN      int                     `json:"repeat_last_n"`
	RepeatPenalty    float32                 `json:"repeat_penalty"`
	PresencePenalty  float32                 `json:"presence_penalty"`
	FrequencyPenalty float32                 `json:"frequency_penalty"`
	DryMultiplier    float32                 `json:"dry_multiplier"`
	DryBase          float32                 `json:"dry_base"`
	DryAllowedLength int                     `json:"dry_allowed_length"`
	DryPenaltyLastN  int                     `json:"dry_penalty_last_n"`
	DryBreakers      []string                `json:"dry_sequence_breakers"`
	Mirostat         int                     `json:"mirostat"`
	MirostatTau      float32                 `json:"mirostat_tau"`
	MirostatEta      float32                 `json:"mirostat_eta"`
	Stop             []string                `json:"stop"`
	MaxTokens        int                     `json:"max_tokens"`
	IgnoreEOS        bool                    `json:"ignore_eos"`
	Stream           bool                    `json:"stream"`
	MinKeep          int                     `json:"min_keep"`
	Grammar          string                  `json:"grammar"`
	Samplers         []sampling.SamplerStage `json:"samplers"`
}

type propertiesModelMetadata struct {
	Name              string `json:"name"`
	Architecture      string `json:"architecture"`
	ContextLength     uint32 `json:"context_length"`
	EmbeddingLength   uint32 `json:"embedding_length"`
	FeedForwardLength uint32 `json:"feed_forward_length"`
	BlockCount        uint32 `json:"block_count"`
	HeadCount         uint32 `json:"head_count"`
	HeadCountKV       uint32 `json:"head_count_kv"`
	VocabularySize    uint32 `json:"vocabulary_size"`
}

type propertiesResponse struct {
	DefaultGenerationSettings struct {
		Params propertiesSamplingParams `json:"params"`
		NCtx   uint32                   `json:"n_ctx"`
	} `json:"default_generation_settings"`
	TotalSlots       int                     `json:"total_slots"`
	ModelAlias       string                  `json:"model_alias"`
	ModelFType       string                  `json:"model_ftype"`
	ModelPath        string                  `json:"model_path"`
	ModelMetadata    propertiesModelMetadata `json:"model_metadata"`
	Modalities       map[string]bool         `json:"modalities"`
	MediaMarker      string                  `json:"media_marker"`
	EndpointSlots    bool                    `json:"endpoint_slots"`
	EndpointProps    bool                    `json:"endpoint_props"`
	EndpointMetrics  bool                    `json:"endpoint_metrics"`
	UI               bool                    `json:"ui"`
	UISettings       map[string]any          `json:"ui_settings"`
	ChatTemplate     string                  `json:"chat_template"`
	ChatTemplateCaps map[string]bool         `json:"chat_template_caps"`
	BOSToken         string                  `json:"bos_token"`
	EOSToken         string                  `json:"eos_token"`
	BuildInfo        string                  `json:"build_info"`
	IsSleeping       bool                    `json:"is_sleeping"`
	CORSProxyEnabled bool                    `json:"cors_proxy_enabled"`
}

func (h *Handler) properties(response http.ResponseWriter, request *http.Request) {
	api, ok := h.generator.(ModelPropertiesAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "model properties are unavailable")
		return
	}
	model := api.ModelProperties()
	samplingConfig := h.defaultSampling
	result := propertiesResponse{
		TotalSlots: h.config.MaxConcurrent,
		ModelAlias: h.config.ModelID,
		ModelFType: model.FileType,
		ModelPath:  model.Path,
		ModelMetadata: propertiesModelMetadata{
			Name:              model.Name,
			Architecture:      model.Architecture,
			ContextLength:     model.ContextLength,
			EmbeddingLength:   model.EmbeddingLength,
			FeedForwardLength: model.FeedForwardLength,
			BlockCount:        model.BlockCount,
			HeadCount:         model.HeadCount,
			HeadCountKV:       model.HeadCountKV,
			VocabularySize:    model.VocabularySize,
		},
		Modalities: map[string]bool{
			"vision": false,
			"video":  false,
			"audio":  false,
		},
		MediaMarker:      "",
		EndpointSlots:    true,
		EndpointProps:    false,
		EndpointMetrics:  true,
		UI:               false,
		UISettings:       map[string]any{},
		ChatTemplate:     model.ChatTemplate,
		ChatTemplateCaps: map[string]bool{},
		BOSToken:         model.BOSToken,
		EOSToken:         model.EOSToken,
		BuildInfo:        "overgo",
		IsSleeping:       false,
	}
	result.DefaultGenerationSettings.NCtx = model.ContextLength
	result.DefaultGenerationSettings.Params = propertiesSamplingParams{
		NPredict:         h.defaultOutputTokens,
		Seed:             samplingConfig.Seed,
		Temperature:      samplingConfig.Temperature,
		DynatempRange:    samplingConfig.DynatempRange,
		DynatempExponent: samplingConfig.DynatempExponent,
		TopK:             samplingConfig.TopK,
		TopP:             samplingConfig.TopP,
		MinP:             samplingConfig.MinP,
		TopNSigma:        samplingConfig.TopNSigma,
		XTCProbability:   samplingConfig.XTCProbability,
		XTCThreshold:     samplingConfig.XTCThreshold,
		TypicalP:         samplingConfig.TypicalP,
		RepeatLastN:      samplingConfig.RepeatLastN,
		RepeatPenalty:    samplingConfig.RepeatPenalty,
		PresencePenalty:  samplingConfig.PresencePenalty,
		FrequencyPenalty: samplingConfig.FrequencyPenalty,
		DryMultiplier:    samplingConfig.DryMultiplier,
		DryBase:          samplingConfig.DryBase,
		DryAllowedLength: samplingConfig.DryAllowedLength,
		DryPenaltyLastN:  samplingConfig.DryPenaltyLastN,
		DryBreakers:      []string{},
		Mirostat:         samplingConfig.Mirostat,
		MirostatTau:      samplingConfig.MirostatTau,
		MirostatEta:      samplingConfig.MirostatEta,
		Stop:             []string{},
		MaxTokens:        h.defaultOutputTokens,
		IgnoreEOS:        false,
		Stream:           false,
		MinKeep:          samplingConfig.MinKeep,
		Grammar:          "",
		Samplers:         slices.Clone(samplingConfig.Samplers),
	}
	writeJSON(response, http.StatusOK, result)
}

type slotStatusItem struct {
	ID                     int                `json:"id"`
	IDTask                 *uint64            `json:"id_task,omitempty"`
	NCtx                   uint32             `json:"n_ctx"`
	Speculative            bool               `json:"speculative"`
	IsProcessing           bool               `json:"is_processing"`
	NPromptTokens          uint64             `json:"n_prompt_tokens,omitzero"`
	NPromptTokensProcessed uint64             `json:"n_prompt_tokens_processed,omitzero"`
	NPromptTokensCache     uint64             `json:"n_prompt_tokens_cache,omitzero"`
	Prompt                 string             `json:"prompt,omitzero"`
	Generated              string             `json:"generated,omitzero"`
	Params                 *slotStatusParams  `json:"params,omitempty"`
	NextToken              *slotNextToken     `json:"next_token,omitempty"`
	Timings                *slotStatusTimings `json:"timings,omitempty"`
}

type slotNextToken struct {
	HasNextToken bool   `json:"has_next_token"`
	HasNewLine   bool   `json:"has_new_line"`
	NRemain      int    `json:"n_remain"`
	NDecoded     uint64 `json:"n_decoded"`
}

type slotStatusParams struct {
	Seed             int64                   `json:"seed"`
	Temperature      float32                 `json:"temperature"`
	DynatempRange    float32                 `json:"dynatemp_range"`
	DynatempExponent float32                 `json:"dynatemp_exponent"`
	TopK             int                     `json:"top_k"`
	TopP             float32                 `json:"top_p"`
	MinP             float32                 `json:"min_p"`
	TypicalP         float32                 `json:"typical_p"`
	TopNSigma        float32                 `json:"top_n_sigma"`
	XTCProbability   float32                 `json:"xtc_probability"`
	XTCThreshold     float32                 `json:"xtc_threshold"`
	MinKeep          int                     `json:"min_keep"`
	AdaptiveTarget   float32                 `json:"adaptive_target"`
	AdaptiveDecay    float32                 `json:"adaptive_decay"`
	RepeatLastN      int                     `json:"repeat_last_n"`
	RepeatPenalty    float32                 `json:"repeat_penalty"`
	PresencePenalty  float32                 `json:"presence_penalty"`
	FrequencyPenalty float32                 `json:"frequency_penalty"`
	DryMultiplier    float32                 `json:"dry_multiplier"`
	DryBase          float32                 `json:"dry_base"`
	DryAllowedLength int                     `json:"dry_allowed_length"`
	DryPenaltyLastN  int                     `json:"dry_penalty_last_n"`
	Mirostat         int                     `json:"mirostat"`
	MirostatTau      float32                 `json:"mirostat_tau"`
	MirostatEta      float32                 `json:"mirostat_eta"`
	MaxTokens        int                     `json:"max_tokens"`
	NPredict         int                     `json:"n_predict"`
	Stop             []string                `json:"stop"`
	Samplers         []sampling.SamplerStage `json:"samplers"`
	CachePrompt      bool                    `json:"cache_prompt"`
	NKeep            int                     `json:"n_keep"`
	NDiscard         int                     `json:"n_discard"`
	NCacheReuse      int                     `json:"n_cache_reuse"`
	ContextShift     bool                    `json:"context_shift"`
}

type slotStatusTimings struct {
	CacheN              uint64  `json:"cache_n"`
	PromptN             uint64  `json:"prompt_n"`
	PromptMS            float64 `json:"prompt_ms"`
	PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
	PromptPerSecond     float64 `json:"prompt_per_second"`
	PredictedN          uint64  `json:"predicted_n"`
	PredictedMS         float64 `json:"predicted_ms"`
	PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
	PredictedPerSecond  float64 `json:"predicted_per_second"`
}

func perTokenAndRate(tokens uint64, durationMS float64) (float64, float64) {
	if tokens == 0 || durationMS <= 0 {
		return 0, 0
	}
	return durationMS / float64(tokens), 1000 * float64(tokens) / durationMS
}

type slotRuntimeMetrics struct {
	PromptTokens    uint64
	ProcessedTokens uint64
	CachedTokens    uint64
	GeneratedTokens uint64
	Timings         slotStatusTimings
}

func (stats *slotRuntimeStats) metrics(processing bool) slotRuntimeMetrics {
	promptTokens := stats.promptTokens.Load()
	cachedTokens := min(stats.cachedTokens.Load(), promptTokens)
	processedTokens := promptTokens - cachedTokens
	generatedTokens := stats.generatedTokens.Load()
	promptNanos := max(stats.promptNanos.Load(), 0)
	totalNanos := max(stats.totalNanos.Load(), 0)
	if processing {
		started := stats.startedNanos.Load()
		if started > 0 {
			totalNanos = max(time.Now().UnixNano()-started, 0)
		}
	}
	predictedNanos := max(totalNanos-promptNanos, 0)
	if generatedTokens > 0 && predictedNanos == 0 {
		predictedNanos = 1
	}
	promptMS := float64(promptNanos) / float64(time.Millisecond)
	predictedMS := float64(predictedNanos) / float64(time.Millisecond)
	promptPerTokenMS, promptPerSecond := perTokenAndRate(processedTokens, promptMS)
	predictedPerTokenMS, predictedPerSecond := perTokenAndRate(generatedTokens, predictedMS)
	return slotRuntimeMetrics{
		PromptTokens:    promptTokens,
		ProcessedTokens: processedTokens,
		CachedTokens:    cachedTokens,
		GeneratedTokens: generatedTokens,
		Timings: slotStatusTimings{
			CacheN:              cachedTokens,
			PromptN:             processedTokens,
			PromptMS:            promptMS,
			PromptPerTokenMS:    promptPerTokenMS,
			PromptPerSecond:     promptPerSecond,
			PredictedN:          generatedTokens,
			PredictedMS:         predictedMS,
			PredictedPerTokenMS: predictedPerTokenMS,
			PredictedPerSecond:  predictedPerSecond,
		},
	}
}

func (h *Handler) slotStatus(response http.ResponseWriter, request *http.Request) {
	includeText := request.URL.Query().Get("include_text") == "1"
	if request.URL.Query().Has("fail_on_no_slot") && h.sessions.Available() == 0 {
		writeError(response, http.StatusServiceUnavailable, "server_busy", "no slot available")
		return
	}
	writeJSON(response, http.StatusOK, h.sessionStatus(includeText))
}

func (h *Handler) sessionStatus(includeText bool) []slotStatusItem {
	var contextLength uint32
	if api, ok := h.generator.(ModelPropertiesAPI); ok {
		contextLength = api.ModelProperties().ContextLength
	}
	result := make([]slotStatusItem, len(h.slotBusy))
	for id := range h.slotBusy {
		processing := h.slotBusy[id].Load()
		stats := &h.slotStats[id]
		task := h.slotTasks[id].Load()
		metrics := stats.metrics(processing)
		prompt, generated, params := stats.snapshot(includeText)
		result[id] = slotStatusItem{
			ID:                     id,
			NCtx:                   contextLength,
			Speculative:            false,
			IsProcessing:           processing,
			NPromptTokens:          metrics.PromptTokens,
			NPromptTokensProcessed: metrics.ProcessedTokens,
			NPromptTokensCache:     metrics.CachedTokens,
			Prompt:                 prompt,
			Generated:              generated,
		}
		if processing && task > 0 {
			result[id].IDTask = &task
		}
		if task > 0 {
			result[id].Params = &params
			result[id].NextToken = &slotNextToken{
				// Generate drives sampling synchronously and never leaves
				// sampled token buffered between calls
				HasNextToken: false,
				HasNewLine:   false,
				NRemain:      max(params.MaxTokens-int(metrics.GeneratedTokens), 0),
				NDecoded:     metrics.GeneratedTokens,
			}
			result[id].Timings = &metrics.Timings
		}
	}
	return result
}

func (h *Handler) loraAdapters(response http.ResponseWriter, request *http.Request) {
	switch request.Method {
	case http.MethodGet:
		if controller, ok := h.generator.(LoRAControlAPI); ok {
			writeJSON(response, http.StatusOK, controller.LoRAAdapters())
			return
		}
		writeJSON(response, http.StatusOK, []any{})
	case http.MethodPost:
		var adapters []inference.LoRAScale
		if !h.decodeJSONWithLimit(response, request, &adapters, maxRequestBytes) {
			return
		}
		controller, ok := h.generator.(LoRAControlAPI)
		if !ok && len(adapters) != 0 {
			writeError(
				response,
				http.StatusNotImplemented,
				errorCodeUnsupportedOperation,
				"LoRA adapter loading and execution are unavailable",
			)
			return
		}
		if ok {
			if err := controller.SetLoRAScales(request.Context(), adapters); err != nil {
				writeInvalidRequest(response, err)
				return
			}
		}
		writeJSON(response, http.StatusOK, map[string]bool{"success": true})
	}
}

type tokenizeRequest struct {
	Content      json.RawMessage `json:"content"`
	AddSpecial   bool            `json:"add_special"`
	ParseSpecial *bool           `json:"parse_special"`
	WithPieces   bool            `json:"with_pieces"`
}

type tokenPieceResponse struct {
	ID    tokenizer.TokenID `json:"id"`
	Piece any               `json:"piece"`
}

func (h *Handler) tokenize(response http.ResponseWriter, request *http.Request) {
	api, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "tokenization is unavailable")
		return
	}
	var body tokenizeRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	parseSpecial := true
	if body.ParseSpecial != nil {
		parseSpecial = *body.ParseSpecial
	}
	tokens, err := tokenizeMixed(api, body.Content, body.AddSpecial, parseSpecial)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	if body.WithPieces {
		pieceAPI, ok := h.generator.(TokenPieceAPI)
		if !ok {
			writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "token pieces are unavailable")
			return
		}
		pieces := make([]tokenPieceResponse, len(tokens))
		for index, token := range tokens {
			piece, err := pieceAPI.TokenPiece(token)
			if err != nil {
				writeInvalidRequest(response, err)
				return
			}
			var rendered any = piece
			if !utf8.ValidString(piece) {
				values := make([]int, len(piece))
				for byteIndex := range piece {
					values[byteIndex] = int(piece[byteIndex])
				}
				rendered = values
			}
			pieces[index] = tokenPieceResponse{ID: token, Piece: rendered}
		}
		writeJSON(response, http.StatusOK, map[string]any{"tokens": pieces})
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{"tokens": tokens})
}

func tokenizeMixed(
	api TokenizationAPI,
	raw json.RawMessage,
	addSpecial, parseSpecial bool,
) ([]tokenizer.TokenID, error) {
	if len(raw) == 0 {
		return []tokenizer.TokenID{}, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		tokens, err := api.TokenizeText(text, addSpecial, parseSpecial)
		if err != nil {
			return nil, err
		}
		return tokens, nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, errors.New("content must be a string or token/string sequence")
	}
	tokens := make([]tokenizer.TokenID, 0, len(parts))
	first := true
	for index, part := range parts {
		var token int64
		if err := json.Unmarshal(part, &token); err == nil {
			if token < 0 || token >= int64(api.SamplingVocabularySize()) {
				return nil, fmt.Errorf("content token %d is out of range", token)
			}
			tokens = append(tokens, tokenizer.TokenID(token))
		} else {
			var partText string
			if err := json.Unmarshal(part, &partText); err != nil {
				return nil, fmt.Errorf(
					"content element %d must be an integer token ID or string",
					index,
				)
			}
			expanded, err := api.TokenizeText(partText, addSpecial && first, parseSpecial)
			if err != nil {
				return nil, fmt.Errorf("tokenize content element %d: %w", index, err)
			}
			tokens = append(tokens, expanded...)
		}
		first = false
	}
	return tokens, nil
}

type detokenizeRequest struct {
	Tokens []int `json:"tokens"`
}

func (h *Handler) detokenize(response http.ResponseWriter, request *http.Request) {
	api, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "detokenization is unavailable")
		return
	}
	var body detokenizeRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	tokens := make([]tokenizer.TokenID, len(body.Tokens))
	for index, token := range body.Tokens {
		if token < 0 || token >= api.SamplingVocabularySize() {
			writeInvalidRequestMessage(response, fmt.Sprintf("token %d is out of range", token))
			return
		}
		tokens[index] = tokenizer.TokenID(token)
	}
	content := ""
	if pieceAPI, ok := h.generator.(TokenPieceAPI); ok {
		var result strings.Builder
		for _, token := range tokens {
			piece, err := pieceAPI.TokenPiece(token)
			if err != nil {
				writeInvalidRequest(response, err)
				return
			}
			result.WriteString(piece)
		}
		content = result.String()
	} else {
		var err error
		content, err = api.Detokenize(tokens, inference.RenderText)
		if err != nil {
			writeInvalidRequest(response, err)
			return
		}
	}
	writeJSON(response, http.StatusOK, map[string]string{"content": content})
}

func (h *Handler) decodeBoundedJSON(
	response http.ResponseWriter,
	request *http.Request,
	target any,
) bool {
	return h.decodeJSONWithLimit(response, request, target, maxRequestBytes)
}

func (h *Handler) decodeMultimodalJSON(
	response http.ResponseWriter,
	request *http.Request,
	target any,
) bool {
	return h.decodeJSONWithLimit(response, request, target, maxMultimodalRequestBytes)
}

func (h *Handler) decodeJSONWithLimit(
	response http.ResponseWriter,
	request *http.Request,
	target any,
	limit int64,
) bool {
	request.Body = http.MaxBytesReader(response, request.Body, limit)
	if err := strictjson.Decode(request.Body, target); err != nil {
		writeInvalidRequestMessage(response, "invalid JSON request: "+err.Error())
		return false
	}
	return true
}

type applyTemplateRequest struct {
	Messages            []inference.ChatMessage `json:"messages"`
	Tools               []inference.ChatTool    `json:"tools"`
	ToolChoice          json.RawMessage         `json:"tool_choice"`
	ParallelTools       *bool                   `json:"parallel_tool_calls"`
	AddGenerationPrompt *bool                   `json:"add_generation_prompt"`
	ChatTemplateKwargs  map[string]any          `json:"chat_template_kwargs"`
}

func (h *Handler) applyTemplate(response http.ResponseWriter, request *http.Request) {
	formatter, ok := h.generator.(ChatFormatter)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "chat formatting is unavailable")
		return
	}
	var body applyTemplateRequest
	if !h.decodeJSONWithLimit(response, request, &body, maxRequestBytes) {
		return
	}
	selection, err := selectChatTools(chatCompletionRequest{
		Messages:       body.Messages,
		Tools:          body.Tools,
		ToolChoice:     body.ToolChoice,
		ParallelTools:  body.ParallelTools,
		AddPrompt:      body.AddGenerationPrompt,
		TemplateKwargs: body.ChatTemplateKwargs,
	})
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	prompt, err := formatChatRequest(
		formatter,
		body.Messages,
		selection.prompt,
		body.AddGenerationPrompt,
		body.ChatTemplateKwargs,
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	writeJSON(response, http.StatusOK, map[string]string{"prompt": prompt})
}

func (h *Handler) authorized(request *http.Request) bool {
	if h.config.APIKey == "" {
		return true
	}
	const prefix = "Bearer "
	header := request.Header.Get("Authorization")
	if !strings.HasPrefix(header, prefix) {
		return false
	}
	provided := header[len(prefix):]
	return subtle.ConstantTimeCompare([]byte(provided), []byte(h.config.APIKey)) == 1
}

func (h *Handler) metrics(response http.ResponseWriter, request *http.Request) {
	response.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	response.WriteHeader(http.StatusOK)
	_, _ = fmt.Fprintf(
		response,
		"# HELP overgo_up Whether the server is running.\n"+
			"# TYPE overgo_up gauge\n"+
			"overgo_up 1\n"+
			"# TYPE overgo_http_requests_total counter\n"+
			"overgo_http_requests_total %d\n"+
			"# TYPE overgo_http_requests_active gauge\n"+
			"overgo_http_requests_active %d\n"+
			"# TYPE overgo_generation_requests_total counter\n"+
			"overgo_generation_requests_total %d\n"+
			"# TYPE overgo_generation_errors_total counter\n"+
			"overgo_generation_errors_total %d\n"+
			"# TYPE overgo_generated_tokens_total counter\n"+
			"overgo_generated_tokens_total %d\n"+
			"# TYPE overgo_process_uptime_seconds gauge\n"+
			"overgo_process_uptime_seconds %.3f\n",
		h.requestsTotal.Load(),
		h.requestsActive.Load(),
		h.generationRequests.Load(),
		h.generationErrors.Load(),
		h.generatedTokens.Load(),
		time.Since(h.started).Seconds(),
	)
	if api, ok := h.generator.(DeviceMemoryAPI); ok {
		if stats, err := api.DeviceMemoryStats(request.Context()); err == nil {
			_, _ = fmt.Fprintf(
				response,
				"# HELP overgo_cuda_memory_current_bytes CUDA bytes currently allocated by this Runner.\n"+
					"# TYPE overgo_cuda_memory_current_bytes gauge\n"+
					"overgo_cuda_memory_current_bytes %d\n"+
					"# HELP overgo_cuda_memory_peak_bytes Lifetime high-water CUDA bytes allocated by this Runner.\n"+
					"# TYPE overgo_cuda_memory_peak_bytes gauge\n"+
					"overgo_cuda_memory_peak_bytes %d\n"+
					"# TYPE overgo_cuda_allocations_current gauge\n"+
					"overgo_cuda_allocations_current %d\n",
				stats.CurrentBytes,
				stats.PeakBytes,
				stats.Allocations,
			)
		}
	}
	if api, ok := h.generator.(DeviceExecutionAPI); ok {
		if stats, err := api.DeviceExecutionStats(request.Context()); err == nil {
			_, _ = fmt.Fprintf(
				response,
				"# TYPE overgo_cuda_custom_kernel_launches_total counter\n"+
					"overgo_cuda_custom_kernel_launches_total %d\n"+
					"# TYPE overgo_cuda_stream_synchronizations_total counter\n"+
					"overgo_cuda_stream_synchronizations_total %d\n"+
					"# TYPE overgo_cuda_context_synchronizations_total counter\n"+
					"overgo_cuda_context_synchronizations_total %d\n"+
					"# TYPE overgo_cuda_host_to_device_bytes_total counter\n"+
					"overgo_cuda_host_to_device_bytes_total %d\n"+
					"# TYPE overgo_cuda_device_to_host_bytes_total counter\n"+
					"overgo_cuda_device_to_host_bytes_total %d\n"+
					"# TYPE overgo_cuda_device_to_device_bytes_total counter\n"+
					"overgo_cuda_device_to_device_bytes_total %d\n"+
					"# TYPE overgo_cuda_device_memset_bytes_total counter\n"+
					"overgo_cuda_device_memset_bytes_total %d\n",
				stats.KernelLaunches,
				stats.StreamSynchronizations,
				stats.ContextSynchronizations,
				stats.HostToDeviceBytes,
				stats.DeviceToHostBytes,
				stats.DeviceToDeviceBytes,
				stats.DeviceMemsetBytes,
			)
		}
	}
}

func (h *Handler) generate(
	ctx context.Context,
	session *requestSession,
	prompt string,
	options inference.GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	started := time.Now()
	hardware := newServingHardwareCollector(h.generator, started)
	hardware.sample(ctx, runrecord.ServingHardwareStart)
	var promptTokens, outputTokens atomic.Uint64
	var promptDuration atomic.Int64
	before := driver.ExecutionStats{}
	if api, ok := h.generator.(DeviceExecutionAPI); ok {
		before, _ = api.DeviceExecutionStats(ctx)
	}
	h.generationRequests.Add(1)
	var stats *slotRuntimeStats
	slotID := -1
	if session != nil {
		slotID = session.ID
	}
	if slotID >= 0 && slotID < len(h.slotStats) {
		stats = &h.slotStats[slotID]
		stats.beginGeneration(prompt, options)
	}
	onPromptEvaluated := options.OnPromptEvaluated
	options.OnPromptEvaluated = func(evaluation inference.PromptEvaluation) {
		hardware.sample(ctx, runrecord.ServingHardwarePrefill)
		promptTokens.Add(uint64(max(evaluation.Tokens, 0)))
		promptDuration.Add(max(evaluation.Duration.Nanoseconds(), 0))
		if stats != nil {
			stats.promptTokens.Add(uint64(max(evaluation.Tokens, 0)))
			stats.cachedTokens.Add(uint64(max(evaluation.Cached, 0)))
			stats.promptNanos.Add(max(evaluation.Duration.Nanoseconds(), 0))
		}
		if onPromptEvaluated != nil {
			onPromptEvaluated(evaluation)
		}
	}
	onToken := options.OnToken
	options.OnToken = func(event inference.TokenEvent) error {
		outputTokens.Add(1)
		h.generatedTokens.Add(1)
		if stats != nil {
			stats.generatedTokens.Add(1)
			stats.appendGenerated(event.Piece)
		}
		if onToken != nil {
			return onToken(event)
		}
		return nil
	}
	ids, text, err := session.Model().Generate(ctx, prompt, options)
	hardware.sample(ctx, runrecord.ServingHardwareFinish)
	if err != nil {
		h.generationErrors.Add(1)
	}
	if modelID, recipeID, ok := h.servingIdentity(recipe.TaskInference); ok {
		after := before
		if api, available := h.generator.(DeviceExecutionAPI); available {
			after, _ = api.DeviceExecutionStats(ctx)
		}
		elapsed := time.Since(started)
		outcome, failure := executionOutcome(err)
		h.publishServing(ctx, runrecord.ServingObservation{
			Model: modelID, Recipe: recipeID, Task: recipe.TaskInference,
			Outcome: outcome, Failure: failure, StartedUnixNS: started.UnixNano(), MeasuredNS: uint64(max(elapsed.Nanoseconds(), 0)),
			Usage: runrecord.ServingUsage{InputTokens: promptTokens.Load(), OutputTokens: outputTokens.Load(), InputBytes: uint64(len(prompt)), OutputBytes: uint64(len(text))},
			Resources: runrecord.ServingResources{
				PeakDeviceBytes:   hardware.peakDeviceBytes(),
				HostToDeviceBytes: servingTransferDelta(before.HostToDeviceBytes, after.HostToDeviceBytes),
				DeviceToHostBytes: servingTransferDelta(before.DeviceToHostBytes, after.DeviceToHostBytes),
			},
			Phases: servingPhases(time.Duration(promptDuration.Load()), elapsed), Hardware: hardware.samples,
		})
	}
	return ids, text, err
}

func (h *Handler) models(response http.ResponseWriter, request *http.Request) {
	model := inference.ModelProperties{}
	if api, ok := h.generator.(ModelPropertiesAPI); ok {
		model = api.ModelProperties()
	}
	created := time.Now().Unix()
	capabilities := []string{"completion", "embedding"}
	if capability, ok := h.generator.(RankCapability); ok && capability.SupportsRank() {
		capabilities = append(capabilities, "rerank")
	}
	writeJSON(response, http.StatusOK, map[string]any{
		"models": []map[string]any{{
			"name":         h.config.ModelID,
			"model":        h.config.ModelID,
			"modified_at":  "",
			"size":         "",
			"digest":       "",
			"type":         "model",
			"description":  "",
			"tags":         []string{},
			"capabilities": capabilities,
			"parameters":   "",
			"details": map[string]any{
				"parent_model":       "",
				"format":             "gguf",
				"family":             model.Architecture,
				"families":           []string{model.Architecture},
				"parameter_size":     strconv.FormatUint(model.ParameterCount, 10),
				"quantization_level": model.FileType,
			},
		}},
		"object": "list",
		"data": []map[string]any{{
			"id":       h.config.ModelID,
			"aliases":  []string{h.config.ModelID},
			"tags":     []string{},
			"object":   "model",
			"created":  created,
			"owned_by": "overgo",
			"meta": map[string]any{
				"vocab_type":  model.VocabularyType,
				"n_vocab":     model.VocabularySize,
				"n_ctx":       model.ContextLength,
				"n_ctx_train": model.ContextLength,
				"n_embd":      model.EmbeddingLength,
				"n_params":    model.ParameterCount,
				"size":        model.ModelSize,
				"ftype":       model.FileType,
			},
		}},
	})
}

func (h *Handler) health(response http.ResponseWriter, request *http.Request) {
	writeJSON(response, http.StatusOK, map[string]any{
		"status": "ok",
		"model":  h.config.ModelID,
	})
}
