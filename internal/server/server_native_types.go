package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"overgo/internal/inference"
	"overgo/internal/projector"
	"overgo/internal/strictjson"
	"overgo/internal/tokenizer"
)

type nativeCompletionRequest struct {
	Prompt            json.RawMessage                `json:"prompt"`
	Model             string                         `json:"model"`
	NPredict          *int                           `json:"n_predict"`
	NCmpl             int                            `json:"n_cmpl"`
	Stop              json.RawMessage                `json:"stop"`
	Stream            bool                           `json:"stream"`
	ReturnTokens      bool                           `json:"return_tokens"`
	CachePrompt       *bool                          `json:"cache_prompt"`
	NIndent           int                            `json:"n_indent"`
	NKeep             int                            `json:"n_keep"`
	NDiscard          int                            `json:"n_discard"`
	NCacheReuse       int                            `json:"n_cache_reuse"`
	NProbs            int                            `json:"n_probs"`
	TMaxPredictMS     int                            `json:"t_max_predict_ms"`
	IDSlot            *int                           `json:"id_slot"`
	TimingsPerToken   bool                           `json:"timings_per_token"`
	ReturnProgress    bool                           `json:"return_progress"`
	SSEPingInterval   *float64                       `json:"sse_ping_interval"`
	PostSamplingProbs bool                           `json:"post_sampling_probs"`
	ResponseFields    []string                       `json:"response_fields"`
	JSONSchema        json.RawMessage                `json:"json_schema"`
	LoRA              json.RawMessage                `json:"lora"`
	ProjectedInputs   *inference.ProjectedInputsJSON `json:"projected_inputs"`
	samplingParameters
}

type infillExtraRequest struct {
	Text     *string `json:"text"`
	Filename *string `json:"filename"`
}

type infillRequest struct {
	InputPrefix json.RawMessage `json:"input_prefix"`
	InputSuffix json.RawMessage `json:"input_suffix"`
	InputExtra  json.RawMessage `json:"input_extra"`
	nativeCompletionRequest
}

func (h *Handler) infill(
	response http.ResponseWriter,
	request *http.Request,
) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	formatter, ok := h.generator.(InfillFormatter)
	if !ok {
		writeError(
			response,
			http.StatusNotImplemented,
			errorCodeUnsupportedOperation,
			"infill formatting is unavailable",
		)
		return
	}
	api, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(
			response,
			http.StatusNotImplemented,
			errorCodeUnsupportedOperation,
			"infill tokenization is unavailable",
		)
		return
	}
	var body infillRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if !strictjson.HasValue(body.InputPrefix) {
		writeInvalidRequestMessage(response, "input_prefix is required")
		return
	}
	if !strictjson.HasValue(body.InputSuffix) {
		writeInvalidRequestMessage(response, "input_suffix is required")
		return
	}
	prefix, err := tokenizeMixed(api, body.InputPrefix, false, false)
	if err != nil {
		writeInvalidRequestMessage(response, "invalid input_prefix: "+err.Error())
		return
	}
	suffix, err := tokenizeMixed(api, body.InputSuffix, false, false)
	if err != nil {
		writeInvalidRequestMessage(response, "invalid input_suffix: "+err.Error())
		return
	}
	var prompt []tokenizer.TokenID
	if strictjson.HasValue(body.Prompt) {
		var promptText string
		if err := json.Unmarshal(body.Prompt, &promptText); err != nil {
			writeInvalidRequestMessage(response, "prompt must be a string")
			return
		}
		prompt, err = api.TokenizeText(promptText, false, true)
		if err != nil {
			writeInvalidRequestMessage(response, "tokenize prompt: "+err.Error())
			return
		}
	}
	var extraRequests []infillExtraRequest
	if len(body.InputExtra) != 0 {
		if err := json.Unmarshal(body.InputExtra, &extraRequests); err != nil ||
			!strictjson.HasValue(body.InputExtra) {
			writeInvalidRequestMessage(response, "input_extra must be an array")
			return
		}
	}
	extra := make([]inference.InfillExtra, len(extraRequests))
	for index, chunk := range extraRequests {
		if chunk.Text == nil {
			writeInvalidRequestMessage(
				response, fmt.Sprintf("input_extra %d requires string text", index),
			)
			return
		}
		extra[index].Filename = "tmp"
		if chunk.Filename != nil {
			extra[index].Filename = *chunk.Filename
		}
		extra[index].Tokens, err = api.TokenizeText(
			*chunk.Text,
			false,
			false,
		)
		if err != nil {
			writeInvalidRequestMessage(
				response, fmt.Sprintf("tokenize input_extra %d: %v", index, err),
			)
			return
		}
	}
	maxTokens := h.config.MaxTokens
	if body.NPredict != nil && *body.NPredict != -1 {
		maxTokens = *body.NPredict
	}
	if maxTokens < 0 || maxTokens > h.config.MaxTokens {
		writeInvalidRequestMessage(
			response, fmt.Sprintf("n_predict must be -1 or in [0,%d]", h.config.MaxTokens),
		)
		return
	}
	formatted, err := formatter.FormatInfillTokens(
		prefix,
		suffix,
		prompt,
		extra,
		inference.InfillFormatOptions{
			BatchSize:    h.config.InfillBatchSize,
			MaxNewTokens: maxTokens,
			SuffixPrefix: h.config.SPMInfill,
		},
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	body.Prompt, err = json.Marshal(formatted)
	if err != nil {
		writeError(
			response,
			http.StatusInternalServerError,
			"internal_error",
			"encode formatted infill prompt",
		)
		return
	}
	encoded, err := json.Marshal(body.nativeCompletionRequest)
	if err != nil {
		writeError(
			response,
			http.StatusInternalServerError,
			"internal_error",
			"encode infill completion request",
		)
		return
	}
	next := request.Clone(request.Context())
	next.Body = io.NopCloser(bytes.NewReader(encoded))
	next.ContentLength = int64(len(encoded))
	h.nativeCompletions(response, next)
}

type nativeCompletionTimings struct {
	CacheN              int     `json:"cache_n"`
	PromptN             int     `json:"prompt_n"`
	PromptMS            float64 `json:"prompt_ms"`
	PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
	PromptPerSecond     float64 `json:"prompt_per_second"`
	PredictedN          int     `json:"predicted_n"`
	PredictedMS         float64 `json:"predicted_ms"`
	PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
	PredictedPerSecond  float64 `json:"predicted_per_second"`
}

type nativeCompletionResponse struct {
	Index                   int                      `json:"index"`
	Content                 string                   `json:"content"`
	Tokens                  []tokenizer.TokenID      `json:"tokens"`
	IDSlot                  int                      `json:"id_slot"`
	Stop                    bool                     `json:"stop"`
	Model                   string                   `json:"model"`
	TokensPredicted         int                      `json:"tokens_predicted"`
	TokensEvaluated         int                      `json:"tokens_evaluated"`
	GenerationSettings      map[string]any           `json:"generation_settings"`
	Prompt                  any                      `json:"prompt"`
	HasNewLine              bool                     `json:"has_new_line"`
	Truncated               bool                     `json:"truncated"`
	StopType                string                   `json:"stop_type"`
	StoppingWord            string                   `json:"stopping_word"`
	TokensCached            int                      `json:"tokens_cached"`
	Timings                 nativeCompletionTimings  `json:"timings"`
	CompletionProbabilities []nativeTokenProbability `json:"completion_probabilities,omitempty"`
}

type nativeCompletionChunk struct {
	Index                   int                      `json:"index"`
	Content                 string                   `json:"content"`
	Tokens                  []tokenizer.TokenID      `json:"tokens"`
	Stop                    bool                     `json:"stop"`
	IDSlot                  int                      `json:"id_slot"`
	TokensPredicted         int                      `json:"tokens_predicted"`
	TokensEvaluated         int                      `json:"tokens_evaluated"`
	PromptProgress          *nativePromptProgress    `json:"prompt_progress,omitempty"`
	Timings                 *nativeCompletionTimings `json:"timings,omitempty"`
	CompletionProbabilities []nativeTokenProbability `json:"completion_probabilities,omitempty"`
}

type nativeTokenLogProbability struct {
	ID      tokenizer.TokenID `json:"id"`
	Token   string            `json:"token"`
	Bytes   []int             `json:"bytes"`
	LogProb float64           `json:"logprob"`
}

type nativeTokenProbability struct {
	ID          tokenizer.TokenID             `json:"id"`
	Token       string                        `json:"token"`
	Bytes       []int                         `json:"bytes"`
	LogProb     *float64                      `json:"logprob,omitempty"`
	Prob        *float64                      `json:"prob,omitempty"`
	Entropy     *float64                      `json:"entropy,omitempty"`
	TopLogProbs []nativeTokenLogProbability   `json:"top_logprobs,omitempty"`
	TopProbs    []nativeTokenProbabilityValue `json:"top_probs,omitempty"`
}

type nativeTokenProbabilityValue struct {
	ID    tokenizer.TokenID `json:"id"`
	Token string            `json:"token"`
	Bytes []int             `json:"bytes"`
	Prob  float64           `json:"prob"`
}

type nativePromptProgress struct {
	Total     int   `json:"total"`
	Cache     int   `json:"cache"`
	Processed int   `json:"processed"`
	TimeMS    int64 `json:"time_ms"`
}

type nativePrompt struct {
	Text         string
	TokenIDs     []tokenizer.TokenID
	Response     any
	Image        []byte
	Images       [][]byte
	MediaText    []string
	Audio        []float32
	BeforeMedia  string
	AfterMedia   string
	MediaHistory bool
	Media        []nativeMedia
	Video        []byte
	VideoFPS     float64
	Thinking     *bool
}

type nativeMedia struct {
	Kind  projector.MediaKind
	Image []byte
	Audio []float32
}

type preparedPrompt struct {
	nativePrompt
	ProjectedInputs *inference.ProjectedInputs
	Multimodal      bool
}

func (h *Handler) nativeCompletions(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	var body nativeCompletionRequest
	if !h.decodeMultimodalJSON(response, request, &body) {
		return
	}
	prompts, err := h.parseNativePrompts(request.Context(), body.Prompt)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	if !h.requireModel(response, body.Model) {
		return
	}
	if err := validateNativeCompletionOptions(body); err != nil {
		writeInvalidRequest(response, err)
		return
	}
	lora, loraConfigured, err := h.parseRequestLoRA(body.LoRA)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	var projectedInputs *inference.ProjectedInputs
	if body.ProjectedInputs != nil {
		projected, projectedErr := body.ProjectedInputs.ProjectedInputs()
		if projectedErr != nil {
			writeInvalidRequest(response, projectedErr)
			return
		}
		projectedInputs = &projected
	}
	if err := prepareStructuredOutput(
		&body.samplingParameters,
		body.JSONSchema,
		nil,
	); err != nil {
		writeInvalidRequest(response, err)
		return
	}
	requestedSlot := -1
	if body.IDSlot != nil {
		requestedSlot = *body.IDSlot
	}
	if requestedSlot < -1 || requestedSlot >= h.config.MaxConcurrent {
		writeInvalidRequestMessage(
			response, fmt.Sprintf("id_slot must be -1 or in [0,%d]", h.config.MaxConcurrent-1),
		)
		return
	}
	maxTokens := h.config.MaxTokens
	if body.NPredict != nil && *body.NPredict != -1 {
		maxTokens = *body.NPredict
	}
	if maxTokens < 0 || maxTokens > h.config.MaxTokens {
		writeInvalidRequestMessage(
			response, fmt.Sprintf("n_predict must be -1 or in [0,%d]", h.config.MaxTokens),
		)
		return
	}
	if body.NCmpl == 0 {
		body.NCmpl = 1
	}
	if body.NCmpl < 1 || body.NCmpl > maxCompletionChoices {
		writeInvalidRequestMessage(response, fmt.Sprintf("n_cmpl must be in [1,%d]", maxCompletionChoices))
		return
	}
	if projectedInputs != nil && (len(prompts) != 1 || body.NCmpl != 1) {
		writeInvalidRequestMessage(response, "projected_inputs requires one prompt and one completion")
		return
	}
	multimodal := len(prompts) == 1 && nativePromptHasMedia(prompts[0])
	if multimodal && (projectedInputs != nil || body.NCmpl != 1) {
		writeInvalidRequestMessage(response, "multimodal prompt requires one completion and no projected_inputs")
		return
	}
	if multimodal && body.CachePrompt != nil && *body.CachePrompt {
		writeInvalidRequestMessage(response, "multimodal prompt cannot use cache_prompt")
		return
	}
	stops, err := parseStopSequences(body.Stop)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	sampler, err := h.newSampler(body.samplingParameters)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	slotID, acquired := h.acquireRequestSlot(response, requestedSlot)
	if !acquired {
		return
	}
	defer h.releaseSlot(slotID)
	for index := range prompts {
		prepared, prepareErr := h.preparePrompt(request.Context(), prompts[index], false)
		if prepareErr != nil {
			writeGenerationError(response, prepareErr)
			return
		}
		prompts[index] = prepared.nativePrompt
		if prepared.ProjectedInputs != nil {
			projectedInputs = prepared.ProjectedInputs
		}
	}
	settings := h.nativeGenerationSettings(body, sampler.Config(), maxTokens, stops)
	settings["multimodal"] = multimodal
	plan := &nativeCompletionPlan{
		handler: h, request: request, body: body, prompts: prompts,
		sampler: sampler, maxTokens: maxTokens, stops: stops, settings: settings,
		slotID: slotID, lora: lora, loraConfigured: loraConfigured,
		projectedInputs: projectedInputs,
	}
	if body.Stream {
		h.streamNativeCompletion(response, plan)
		return
	}
	results := make([]nativeCompletionResponse, 0, len(prompts)*body.NCmpl)
	resultIndex := 0
	for _, prompt := range plan.prompts {
		for range body.NCmpl {
			result, generationErr := plan.run(
				prompt,
				resultIndex,
				body.ReturnTokens,
				nil,
				nil,
			)
			if generationErr != nil {
				writeGenerationError(response, generationErr)
				return
			}
			results = append(results, result)
			resultIndex++
		}
	}
	if len(body.ResponseFields) > 0 {
		projected := make([]map[string]any, 0, len(results))
		for _, result := range results {
			item, projectErr := projectNativeResponse(result, body.ResponseFields)
			if projectErr != nil {
				writeError(response, http.StatusInternalServerError, "server_error", projectErr.Error())
				return
			}
			projected = append(projected, item)
		}
		if len(projected) == 1 {
			writeJSON(response, http.StatusOK, projected[0])
			return
		}
		writeJSON(response, http.StatusOK, projected)
		return
	}
	if len(results) == 1 {
		writeJSON(response, http.StatusOK, results[0])
		return
	}
	writeJSON(response, http.StatusOK, results)
}
