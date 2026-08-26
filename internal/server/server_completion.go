package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/strictjson"
)

func (h *Handler) parseStopSequences(raw json.RawMessage) ([]string, error) {
	if !strictjson.HasValue(raw) {
		return nil, nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if single == "" {
			return nil, errors.New("stop sequence must not be empty")
		}
		return []string{single}, nil
	}
	var multiple []string
	if err := json.Unmarshal(raw, &multiple); err != nil {
		return nil, errors.New("stop must be a string or string array")
	}
	limit := h.config.RuntimePolicy.Serving.Limits.StopSequences
	if len(multiple) > limit {
		return nil, fmt.Errorf("stop sequence count exceeds %d", limit)
	}
	for _, stop := range multiple {
		if stop == "" {
			return nil, errors.New("stop sequence must not be empty")
		}
	}
	return multiple, nil
}

type stopFilter struct {
	stops   []string
	pending string
	stopped bool
	word    string
}

func newStopFilter(stops []string) *stopFilter {
	return &stopFilter{stops: slices.Clone(stops)}
}

func (filter *stopFilter) Accept(piece string) string {
	if filter == nil || filter.stopped {
		return ""
	}
	if len(filter.stops) == 0 {
		return piece
	}
	filter.pending += piece
	match := -1
	for _, stop := range filter.stops {
		if index := strings.Index(filter.pending, stop); index >= 0 &&
			(match < 0 || index < match) {
			match = index
			filter.word = stop
		}
	}
	if match >= 0 {
		safe := filter.pending[:match]
		filter.pending = ""
		filter.stopped = true
		return safe
	}
	hold := 0
	for _, stop := range filter.stops {
		limit := min(len(filter.pending), len(stop)-1)
		for length := limit; length > hold; length-- {
			if strings.HasSuffix(filter.pending, stop[:length]) {
				hold = length
				break
			}
		}
	}
	safeLength := len(filter.pending) - hold
	safe := filter.pending[:safeLength]
	filter.pending = filter.pending[safeLength:]
	return safe
}

func (filter *stopFilter) Flush() string {
	if filter == nil || filter.stopped {
		return ""
	}
	safe := filter.pending
	filter.pending = ""
	return safe
}

func (filter *stopFilter) Stopped() bool {
	return filter != nil && filter.stopped
}

func (filter *stopFilter) StoppingWord() string {
	if filter == nil {
		return ""
	}
	return filter.word
}

type samplingParameters struct {
	Temperature            *float32        `json:"temperature"`
	DynatempRange          float32         `json:"dynatemp_range"`
	DynatempExponent       float32         `json:"dynatemp_exponent"`
	TopP                   *float32        `json:"top_p"`
	TopK                   *int            `json:"top_k"`
	MinP                   float32         `json:"min_p"`
	TypicalP               float32         `json:"typical_p"`
	TopNSigma              float32         `json:"top_n_sigma"`
	XTCProbability         float32         `json:"xtc_probability"`
	XTCThreshold           float32         `json:"xtc_threshold"`
	MinKeep                int             `json:"min_keep"`
	AdaptiveTarget         *float32        `json:"adaptive_target"`
	AdaptiveDecay          *float32        `json:"adaptive_decay"`
	RepeatLastN            int             `json:"repeat_last_n"`
	RepeatPenalty          float32         `json:"repeat_penalty"`
	PresencePenalty        float32         `json:"presence_penalty"`
	FrequencyPenalty       float32         `json:"frequency_penalty"`
	DryMultiplier          float32         `json:"dry_multiplier"`
	DryBase                float32         `json:"dry_base"`
	DryAllowedLength       int             `json:"dry_allowed_length"`
	DryPenaltyLastN        int             `json:"dry_penalty_last_n"`
	DryBreakers            []string        `json:"dry_sequence_breakers"`
	Mirostat               int             `json:"mirostat"`
	MirostatTau            float32         `json:"mirostat_tau"`
	MirostatEta            float32         `json:"mirostat_eta"`
	Seed                   int64           `json:"seed"`
	GrammarChoices         []string        `json:"grammar_choices"`
	Grammar                string          `json:"grammar"`
	GrammarRoot            string          `json:"grammar_root"`
	GrammarLazy            bool            `json:"grammar_lazy"`
	GrammarTriggerPatterns []string        `json:"grammar_trigger_patterns"`
	GrammarTriggerTokens   []int           `json:"grammar_trigger_tokens"`
	Samplers               []string        `json:"samplers"`
	LogitBias              json.RawMessage `json:"logit_bias"`
	IgnoreEOS              bool            `json:"ignore_eos"`
}

type completionRequest struct {
	Model      string          `json:"model"`
	Prompt     json.RawMessage `json:"prompt"`
	MaxTokens  *int            `json:"max_tokens"`
	Stop       json.RawMessage `json:"stop"`
	N          int             `json:"n"`
	JSONSchema json.RawMessage `json:"json_schema"`
	samplingParameters
	Stream bool `json:"stream"`
}

func (h *Handler) completions(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	var body completionRequest
	if !h.decodeJSONWithLimit(response, request, &body, maxRequestBytes) {
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
	if body.N == 0 {
		body.N = 1
	}
	if body.N < 1 || body.N > maxCompletionChoices {
		writeInvalidRequestMessage(response, fmt.Sprintf("n must be in [1,%d]", maxCompletionChoices))
		return
	}
	maxTokens, err := boundedProtocolTokens(
		body.MaxTokens, h.defaultOutputTokens, h.config.MaxTokens, "max_tokens", false,
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	stops, err := h.parseStopSequences(body.Stop)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	if err := prepareStructuredOutput(
		&body.samplingParameters,
		body.JSONSchema,
		nil,
	); err != nil {
		writeInvalidRequest(response, err)
		return
	}
	plan, ok := h.prepareProtocolBatchGenerationPlan(
		response, request, prompts, body.samplingParameters, maxTokens, stops,
	)
	if !ok {
		return
	}
	defer plan.release()
	id := "cmpl-" + strconv.FormatUint(h.nextID.Add(1), identifierRadix)
	if body.Stream {
		h.streamCompletion(response, request, plan, id, body.N)
		return
	}
	h.complete(response, plan, id, body.N)
}
