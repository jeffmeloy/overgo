package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"overgo/internal/inference"
	"overgo/internal/sampling"
	"overgo/internal/strictjson"
)

func validateNativeCompletionOptions(body nativeCompletionRequest) error {
	switch {
	case body.NIndent < 0:
		return errors.New("n_indent must be non-negative")
	case body.NKeep < -1:
		return errors.New("n_keep must be at least -1")
	case body.NDiscard < 0:
		return errors.New("n_discard must be non-negative")
	case body.NCacheReuse < 0:
		return errors.New("n_cache_reuse is negative")
	case body.NCacheReuse > 0 && (body.CachePrompt == nil || !*body.CachePrompt):
		return errors.New("n_cache_reuse requires cache_prompt")
	case body.NProbs < 0:
		return errors.New("n_probs must be non-negative")
	case body.TMaxPredictMS < -1:
		return errors.New("t_max_predict_ms must be at least -1")
	case body.SSEPingInterval != nil &&
		(math.IsNaN(*body.SSEPingInterval) ||
			math.IsInf(*body.SSEPingInterval, 0) ||
			*body.SSEPingInterval < -1 ||
			*body.SSEPingInterval > math.MaxInt32 ||
			math.Trunc(*body.SSEPingInterval) != *body.SSEPingInterval):
		return errors.New("sse_ping_interval must be an integer in [-1,2147483647]")
	case body.PostSamplingProbs && body.NProbs == 0:
		return errors.New("post_sampling_probs requires positive n_probs")
	case len(body.ResponseFields) > 64:
		return errors.New("response_fields count exceeds 64")
	case nativeJSONSchemaConfigured(body.JSONSchema) &&
		(body.Grammar != "" ||
			len(body.GrammarChoices) > 0 ||
			body.GrammarLazy ||
			len(body.GrammarTriggerPatterns) > 0 ||
			len(body.GrammarTriggerTokens) > 0):
		return errors.New("json_schema cannot be combined with grammar options")
	default:
		for index, path := range body.ResponseFields {
			if len(path) > 256 {
				return fmt.Errorf("response_fields path %d exceeds 256 bytes", index)
			}
			if strings.Count(path, "/") >= 16 {
				return fmt.Errorf("response_fields path %d exceeds 16 components", index)
			}
		}
		return nil
	}
}

func (h *Handler) parseRequestLoRA(raw json.RawMessage) ([]inference.LoRAScale, bool, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return nil, false, nil
	}
	controller, ok := h.generator.(LoRAControlAPI)
	if !ok {
		return nil, false, errors.New("per-request lora is unavailable")
	}
	var requested []inference.LoRAScale
	if err := strictjson.Decode(strings.NewReader(trimmed), &requested); err != nil {
		return nil, false, fmt.Errorf("invalid lora: %w", err)
	}
	loaded := controller.LoRAAdapters()
	valid := make(map[int]struct{}, len(loaded))
	for _, adapter := range loaded {
		valid[adapter.ID] = struct{}{}
	}
	seen := make(map[int]struct{}, len(requested))
	for _, adapter := range requested {
		if _, exists := valid[adapter.ID]; !exists {
			return nil, false, fmt.Errorf("lora adapter ID %d is not loaded", adapter.ID)
		}
		if _, duplicate := seen[adapter.ID]; duplicate {
			return nil, false, fmt.Errorf("lora adapter ID %d is duplicated", adapter.ID)
		}
		if math.IsNaN(float64(adapter.Scale)) || math.IsInf(float64(adapter.Scale), 0) {
			return nil, false, fmt.Errorf("lora adapter ID %d scale is invalid", adapter.ID)
		}
		seen[adapter.ID] = struct{}{}
	}
	return requested, true, nil
}

func nativeJSONSchemaConfigured(value json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(value))
	return trimmed != "" && trimmed != "null"
}

func grammarOptionsConfigured(parameters samplingParameters) bool {
	return parameters.Grammar != "" ||
		len(parameters.GrammarChoices) > 0 ||
		parameters.GrammarLazy ||
		len(parameters.GrammarTriggerPatterns) > 0 ||
		len(parameters.GrammarTriggerTokens) > 0
}

func prepareStructuredOutput(
	parameters *samplingParameters,
	topLevelSchema, responseFormat json.RawMessage,
) error {
	schema := topLevelSchema
	if nativeJSONSchemaConfigured(schema) && grammarOptionsConfigured(*parameters) {
		return errors.New("json_schema cannot be combined with grammar options")
	}
	if nativeJSONSchemaConfigured(responseFormat) {
		var format map[string]json.RawMessage
		if err := json.Unmarshal(responseFormat, &format); err != nil || format == nil {
			return errors.New("response_format must be an object")
		}
		var responseType string
		if rawType, ok := format["type"]; ok {
			if err := json.Unmarshal(rawType, &responseType); err != nil {
				return errors.New("response_format.type must be a string")
			}
		}
		switch responseType {
		case "", "text":
		case "json_object":
			rawSchema, hasSchema := format["schema"]
			if hasSchema || !nativeJSONSchemaConfigured(schema) {
				if hasSchema {
					schema = rawSchema
				} else {
					schema = json.RawMessage(`{}`)
				}
			}
		case "json_schema":
			schema = json.RawMessage(`{}`)
			if rawWrapper, ok := format["json_schema"]; ok {
				var wrapper map[string]json.RawMessage
				if err := json.Unmarshal(rawWrapper, &wrapper); err != nil || wrapper == nil {
					return errors.New("response_format.json_schema must be an object")
				}
				if rawSchema, ok := wrapper["schema"]; ok {
					schema = rawSchema
				}
			}
		default:
			return fmt.Errorf(
				"response_format.type must be one of %q, %q, or %q, got %q",
				"text",
				"json_object",
				"json_schema",
				responseType,
			)
		}
	}
	if !nativeJSONSchemaConfigured(schema) {
		return nil
	}
	if grammarOptionsConfigured(*parameters) {
		return errors.New("json_schema cannot be combined with grammar options")
	}
	grammar, err := sampling.JSONSchemaToGrammar(schema)
	if err != nil {
		return fmt.Errorf("json_schema: %w", err)
	}
	parameters.Grammar = grammar
	parameters.GrammarRoot = "root"
	return nil
}

func projectNativeResponse(
	response nativeCompletionResponse,
	paths []string,
) (map[string]any, error) {
	encoded, err := json.Marshal(response)
	if err != nil {
		return nil, fmt.Errorf("marshal native response for projection: %w", err)
	}
	var source map[string]any
	if err := json.Unmarshal(encoded, &source); err != nil {
		return nil, fmt.Errorf("decode native response for projection: %w", err)
	}
	result := make(map[string]any, len(paths))
	for _, path := range paths {
		var current any = source
		valid := true
		for _, component := range strings.Split(path, "/") {
			object, ok := current.(map[string]any)
			if !ok {
				valid = false
				break
			}
			current, ok = object[component]
			if !ok {
				valid = false
				break
			}
		}
		if valid {
			result[path] = current
		}
	}
	return result, nil
}

func (h *Handler) nativeGenerationSettings(
	body nativeCompletionRequest,
	config sampling.Config,
	maxTokens int,
	stops []string,
) map[string]any {
	contextLength := uint32(0)
	if api, ok := h.generator.(ModelPropertiesAPI); ok {
		contextLength = api.ModelProperties().ContextLength
	}
	return map[string]any{
		"model":                 h.config.ModelID,
		"n_ctx":                 contextLength,
		"n_predict":             maxTokens,
		"n_cmpl":                body.NCmpl,
		"seed":                  config.Seed,
		"temperature":           config.Temperature,
		"dynatemp_range":        config.DynatempRange,
		"dynatemp_exponent":     config.DynatempExponent,
		"top_k":                 config.TopK,
		"top_p":                 config.TopP,
		"min_p":                 config.MinP,
		"typical_p":             config.TypicalP,
		"top_n_sigma":           config.TopNSigma,
		"xtc_probability":       config.XTCProbability,
		"xtc_threshold":         config.XTCThreshold,
		"min_keep":              config.MinKeep,
		"repeat_last_n":         config.RepeatLastN,
		"repeat_penalty":        config.RepeatPenalty,
		"presence_penalty":      config.PresencePenalty,
		"frequency_penalty":     config.FrequencyPenalty,
		"dry_multiplier":        config.DryMultiplier,
		"dry_base":              config.DryBase,
		"dry_allowed_length":    config.DryAllowedLength,
		"dry_penalty_last_n":    config.DryPenaltyLastN,
		"dry_sequence_breakers": append([]string{}, body.DryBreakers...),
		"mirostat":              config.Mirostat,
		"mirostat_tau":          config.MirostatTau,
		"mirostat_eta":          config.MirostatEta,
		"samplers":              slices.Clone(config.Samplers),
		"stop":                  append([]string{}, stops...),
		"ignore_eos":            body.IgnoreEOS,
		"stream":                body.Stream,
		"return_tokens":         body.ReturnTokens,
		"return_progress":       body.ReturnProgress,
		"timings_per_token":     body.TimingsPerToken,
		"t_max_predict_ms":      body.TMaxPredictMS,
		"n_indent":              body.NIndent,
		"n_probs":               body.NProbs,
		"post_sampling_probs":   body.PostSamplingProbs,
		"n_keep":                body.NKeep,
		"n_discard":             body.NDiscard,
		"cache_prompt":          body.CachePrompt != nil && *body.CachePrompt,
		"projected_inputs":      body.ProjectedInputs != nil,
		"sse_ping_interval":     nativeSSEPingInterval(body),
	}
}

func nativeSSEPingInterval(body nativeCompletionRequest) int {
	if body.SSEPingInterval == nil {
		return 30
	}
	return int(*body.SSEPingInterval)
}
