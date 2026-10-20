package server

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/inference"
	"overgo/internal/objectstore"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
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
	api, ok := h.generator.(modelPropertiesAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "model properties are unavailable")
		return
	}
	model := api.ModelProperties()
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
	result.DefaultGenerationSettings.Params = h.defaultSamplingParams()
	writeJSON(response, http.StatusOK, result)
}

// defaultSamplingParams is the served model's generation defaults as the
// API and the workspace capability document both publish them.
func (h *Handler) defaultSamplingParams() propertiesSamplingParams {
	samplingConfig := h.defaultSampling
	return propertiesSamplingParams{
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
	// DraftN and DraftNAccepted count a speculative turn's drafted tokens
	// and those the model accepted.
	DraftN         uint64 `json:"draft_n,omitzero"`
	DraftNAccepted uint64 `json:"draft_n_accepted,omitzero"`
	// TokensPerForward is the generated tokens per model forward: one each,
	// and each accepted draft saves a forward.
	TokensPerForward float64 `json:"tokens_per_forward,omitzero"`
}

// tokensPerForward is generated tokens over the forwards that produced them:
// each accepted draft rides on its verifying forward.
func tokensPerForward(generated, accepted uint64) float64 {
	forwards := generated - min(accepted, generated)
	if forwards == 0 {
		return 0
	}
	return float64(generated) / float64(forwards)
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
	accepted := stats.acceptedDrafts.Load()
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
			DraftN:              stats.draftedTokens.Load(),
			DraftNAccepted:      accepted,
			TokensPerForward:    tokensPerForward(generatedTokens, accepted),
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

const (
	slotActionSave    = "save"
	slotActionRestore = "restore"
	// slotStateSchema and slotStateMediaType are the stored form of a saved prompt state, the bytes SavePromptState writes.
	slotStateSchema    = "overgo/prompt-state/v1"
	slotStateMediaType = "application/octet-stream"
	// slotStateAliasRoot opens a saved state's name, slot-state/<model id>/<filename>, so a state is found under its model alone.
	slotStateAliasRoot = "slot-state/"
	// slotFilenameBytes bounds a saved state's filename.
	//   - source = the 255-byte name limit of the filesystems llama-server writes its slot files to
	//   - scope = the filename of POST /slots/{id_slot}
	//   - reconsider when a client needs longer names, within the store's alias bound
	slotFilenameBytes = 255
)

// slotStateAPI saves and restores the prompt state a served model keeps.
type slotStateAPI interface {
	SavePromptState(context.Context) ([]byte, int, error)
	RestorePromptState(context.Context, []byte) (int, error)
}

type slotActionRequest struct {
	Filename string `json:"filename"`
}

type slotSaveResponse struct {
	IDSlot   int    `json:"id_slot"`
	Filename string `json:"filename"`
	ID       string `json:"id"`
	NSaved   int    `json:"n_saved"`
	NWritten int    `json:"n_written"`
	Timings  struct {
		SaveMS float64 `json:"save_ms"`
	} `json:"timings"`
}

type slotRestoreResponse struct {
	IDSlot    int    `json:"id_slot"`
	Filename  string `json:"filename"`
	ID        string `json:"id"`
	NRestored int    `json:"n_restored"`
	NRead     int    `json:"n_read"`
	Timings   struct {
		RestoreMS float64 `json:"restore_ms"`
	} `json:"timings"`
}

// slotAction saves or restores a slot's prompt state in llama-server's shape (POST /slots/{id_slot}?action=save|restore).
//   - the state is a stored object under a per-model alias, so another model never finds it
//   - it serves max_concurrent 1 alone, because the prompt cache belongs to the runner and the continuous scheduler keeps none
//   - a slot running a task is refused, never waited for
func (h *Handler) slotAction(response http.ResponseWriter, request *http.Request) {
	text := request.URL.Path[strings.LastIndex(request.URL.Path, "/")+1:]
	id, err := strconv.Atoi(text)
	if err != nil || id < 0 || id >= len(h.slotBusy) {
		writeInvalidRequestMessage(response, fmt.Sprintf("id_slot %q must be in [0,%d]", text, len(h.slotBusy)-1))
		return
	}
	action := request.URL.Query().Get("action")
	if action != slotActionSave && action != slotActionRestore {
		writeInvalidRequestMessage(response, fmt.Sprintf("action %q must be %s or %s", action, slotActionSave, slotActionRestore))
		return
	}
	var body slotActionRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if !textcheck.BoundedToken(body.Filename, slotFilenameBytes, `/\`) || strings.Contains(body.Filename, "..") {
		writeInvalidRequestMessage(response, fmt.Sprintf(
			"filename %q must be one name of at most %d bytes with no space, separator or ..", body.Filename, slotFilenameBytes))
		return
	}
	if h.slotStates == nil {
		writeError(response, http.StatusServiceUnavailable, "repository_unavailable", "no artifact repository is configured")
		return
	}
	if len(h.slotBusy) != 1 {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, fmt.Sprintf(
			"slot save and restore serve max_concurrent 1, the server runs %d", len(h.slotBusy)))
		return
	}
	modelID, recipeID, identified := h.servingIdentity(recipe.TaskInference)
	if !identified {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "the served model has no stored identity")
		return
	}
	lease, leased := h.sessions.TryLease(id)
	if !leased {
		writeError(response, http.StatusConflict, "slot_busy", fmt.Sprintf("slot %d is running a task", id))
		return
	}
	h.slotBusy[id].Store(true)
	defer func() {
		h.slotBusy[id].Store(false)
		_ = lease.Release()
	}()
	api, ok := lease.Model().(slotStateAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "the served model keeps no prompt state")
		return
	}
	call := slotCall{
		slot: id, filename: body.Filename, model: modelID, recipe: recipeID, api: api,
		clock: processmeasure.NewStopwatch(),
	}
	if action == slotActionSave {
		h.saveSlot(request.Context(), response, call)
		return
	}
	h.restoreSlot(request.Context(), response, call)
}

// slotCall holds one admitted slot action's slot, filename and serving model.
type slotCall struct {
	slot          int
	filename      string
	model, recipe artifact.ID
	api           slotStateAPI
	clock         processmeasure.Stopwatch
}

// names is the alias prefix of every saved name of the call's model.
func (c slotCall) names() string { return slotStateAliasRoot + c.model.String() + "/" }

// milliseconds reports the time since the call was admitted, zero when the clock cannot be read.
func (c slotCall) milliseconds() float64 {
	elapsed, _ := c.clock.Elapsed()
	return float64(elapsed) / float64(time.Millisecond)
}

// saveSlot stores the served model's prompt state and moves its name to it in one commit.
//   - the state depends on the model and the recipe, and its lineage says so
//   - a state the name held before is retired unless another name of the model still holds it
func (h *Handler) saveSlot(ctx context.Context, response http.ResponseWriter, call slotCall) {
	data, tokens, err := call.api.SavePromptState(ctx)
	if errors.Is(err, inference.ErrPromptStateRefused) {
		writeError(response, http.StatusConflict, "prompt_state_refused", err.Error())
		return
	}
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	id, err := artifact.IdentifyBytes(artifact.KindOutput, data)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	// A name that holds this state already holds a published object: nothing is written.
	move, moved, err := artifact.MoveAlias(ctx, h.repository, call.names()+call.filename, id)
	if err == nil && moved {
		_, err = h.slotStates.Publish(ctx, objectstore.PublishRequest{
			Kind: artifact.KindOutput, Expected: &id, MediaType: slotStateMediaType, Schema: slotStateSchema,
			Reader: bytes.NewReader(data), Lineage: artifact.DependencyLineage(id, call.model, call.recipe),
			Aliases: []artifact.AliasBinding{move},
		})
	}
	if errors.Is(err, overgodb.ErrAliasConflict) {
		writeError(response, http.StatusConflict, "slot_state_conflict", err.Error())
		return
	}
	if err == nil && moved && move.Previous != nil {
		err = h.retireSlotState(ctx, call, *move.Previous)
	}
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	result := slotSaveResponse{IDSlot: call.slot, Filename: call.filename, ID: id.String(), NSaved: tokens, NWritten: len(data)}
	result.Timings.SaveMS = call.milliseconds()
	writeJSON(response, http.StatusOK, result)
}

// retireSlotState retires a state no saved name of the model holds any more.
func (h *Handler) retireSlotState(ctx context.Context, call slotCall, state artifact.ID) error {
	held := false
	if err := h.repository.VisitAliases(ctx, call.names(), func(alias overgodb.AliasView) error {
		held = held || alias.Target == state
		return nil
	}); err != nil || held {
		return err
	}
	return h.slotStates.Retire(ctx, state)
}

// restoreSlot reads the state the model's name holds into the served model's prompt cache.
//   - a name the model never saved answers 404
//   - a state the runner refuses answers 409
func (h *Handler) restoreSlot(ctx context.Context, response http.ResponseWriter, call slotCall) {
	id, found, err := artifact.ResolveAlias(ctx, h.repository, call.names()+call.filename)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	if !found {
		writeError(response, http.StatusNotFound, "not_found", fmt.Sprintf("no prompt state %q is saved for model %s", call.filename, call.model))
		return
	}
	descriptor, reader, err := h.slotStates.Open(ctx, id)
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	data, err := io.ReadAll(io.LimitReader(reader, int64(descriptor.Size)))
	err = errors.Join(err, reader.Close())
	if err != nil {
		writeError(response, http.StatusInternalServerError, "overgodb_error", err.Error())
		return
	}
	tokens, err := call.api.RestorePromptState(ctx, data)
	if errors.Is(err, inference.ErrPromptStateRefused) {
		writeError(response, http.StatusConflict, "prompt_state_refused", err.Error())
		return
	}
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	result := slotRestoreResponse{IDSlot: call.slot, Filename: call.filename, ID: id.String(), NRestored: tokens, NRead: len(data)}
	result.Timings.RestoreMS = call.milliseconds()
	writeJSON(response, http.StatusOK, result)
}

func (h *Handler) sessionStatus(includeText bool) []slotStatusItem {
	var contextLength uint32
	if api, ok := h.generator.(modelPropertiesAPI); ok {
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
		if controller, ok := h.generator.(loRAControlAPI); ok {
			writeJSON(response, http.StatusOK, controller.LoRAAdapters())
			return
		}
		writeJSON(response, http.StatusOK, []any{})
	case http.MethodPost:
		var adapters []inference.LoRAScale
		if !h.decodeJSONWithLimit(response, request, &adapters, maxRequestBytes) {
			return
		}
		controller, ok := h.generator.(loRAControlAPI)
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
	api, ok := h.generator.(tokenizationAPI)
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
		pieceAPI, ok := h.generator.(tokenPieceAPI)
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
	api tokenizationAPI,
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
	api, ok := h.generator.(tokenizationAPI)
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
	if pieceAPI, ok := h.generator.(tokenPieceAPI); ok {
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

func (h *Handler) decodeJSONWithLimit(response http.ResponseWriter, request *http.Request, target any, limit int64) bool {
	return decodeJSONBounded(response, request, target, limit)
}

// decodeJSONBounded: the bounded strict decode every JSON route shares (the idle shell decodes the same way).
func decodeJSONBounded(response http.ResponseWriter, request *http.Request, target any, limit int64) bool {
	// Retain header-less non-browser API calls. Browser JSON requests must
	// declare their media type; explicit non-JSON types are never decoded.
	contentTypes := request.Header.Values("Content-Type")
	if len(contentTypes) != 0 || request.Header.Get("Origin") != "" || request.Header.Get("Sec-Fetch-Site") != "" {
		mediaType, _, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
		if len(contentTypes) != 1 || err != nil || mediaType != "application/json" {
			writeError(response, http.StatusUnsupportedMediaType, "unsupported_media_type", "Content-Type must be application/json")
			return false
		}
	}
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
	formatter, ok := h.generator.(chatFormatter)
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
			"# HELP overgo_prompt_tokens_total Prompt tokens evaluated, those the cache held left out.\n"+
			"# TYPE overgo_prompt_tokens_total counter\n"+
			"overgo_prompt_tokens_total %d\n"+
			"# TYPE overgo_prompt_seconds_total counter\n"+
			"overgo_prompt_seconds_total %g\n"+
			"# TYPE overgo_process_uptime_seconds gauge\n"+
			"overgo_process_uptime_seconds %.3f\n",
		h.requestsTotal.Load(),
		h.requestsActive.Load(),
		h.generationRequests.Load(),
		h.generationErrors.Load(),
		h.generatedTokens.Load(),
		h.prefilledTokens.Load(),
		time.Duration(h.prefillNanos.Load()).Seconds(),
		time.Since(h.started).Seconds(),
	)
	if api, ok := h.generator.(deviceMemoryOwnersAPI); ok {
		if memory, err := api.DeviceMemory(request.Context()); err == nil {
			_, _ = fmt.Fprintf(response,
				"# TYPE overgo_device_memory_total_bytes gauge\novergo_device_memory_total_bytes %d\n"+
					"# TYPE overgo_device_memory_free_bytes gauge\novergo_device_memory_free_bytes %d\n",
				memory.TotalBytes, memory.FreeBytes)
			if experts := memory.Experts; experts != nil {
				_, _ = fmt.Fprintf(response,
					"# TYPE overgo_experts_card gauge\novergo_experts_card %d\n"+
						"# TYPE overgo_experts_host gauge\novergo_experts_host %d\n"+
						"# TYPE overgo_experts_card_bytes gauge\novergo_experts_card_bytes %d\n"+
						"# TYPE overgo_experts_host_bytes gauge\novergo_experts_host_bytes %d\n"+
						"# HELP overgo_expert_routes_total Experts the runs' routes named, a bank's expert once a run.\n"+
						"# TYPE overgo_expert_routes_total counter\novergo_expert_routes_total %d\n"+
						"# TYPE overgo_expert_routes_card_total counter\novergo_expert_routes_card_total %d\n"+
						"# HELP overgo_experts_load_card Experts the load placed on the card, few where it found the card held; more move there once memory frees.\n"+
						"# TYPE overgo_experts_load_card gauge\novergo_experts_load_card %d\n",
					experts.CardExperts, experts.HostExperts, experts.CardBytes, experts.HostBytes, experts.Routes, experts.CardRoutes, experts.LoadCardExperts)
			}
		}
	}
	// A reading the card's management library does not answer is left out.
	if api, ok := h.generator.(cardReadingsAPI); ok {
		if readings, err := api.CardReadings(request.Context()); err == nil {
			for _, reading := range readings {
				_, _ = fmt.Fprintf(response, "# TYPE overgo_card_%[1]s gauge\novergo_card_%[1]s %[2]g\n", reading.Name, reading.Value)
			}
		}
	}
	if api, ok := h.generator.(deviceMemoryAPI); ok {
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
	if api, ok := h.generator.(deviceExecutionAPI); ok {
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
	clock := processmeasure.NewStopwatch()
	hardware := newServingHardwareCollector(h.generator, clock)
	hardware.sample(ctx, runrecord.ServingHardwareStart)
	var promptTokens, cachedTokens, outputTokens, promptEvaluated atomic.Uint64
	var promptDuration atomic.Int64
	before := driver.ExecutionStats{}
	if api, ok := h.generator.(deviceExecutionAPI); ok {
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
		if promptEvaluated.Load() == 0 {
			promptEvaluated.CompareAndSwap(0, hardware.walls.Elapsed(clock))
		}
		hardware.sample(ctx, runrecord.ServingHardwarePrefill)
		promptTokens.Add(uint64(max(evaluation.Tokens, 0)))
		cachedTokens.Add(uint64(max(evaluation.Cached, 0)))
		promptDuration.Add(max(evaluation.Duration.Nanoseconds(), 0))
		h.prefilledTokens.Add(uint64(max(evaluation.Tokens-evaluation.Cached, 0)))
		h.prefillNanos.Add(uint64(max(evaluation.Duration.Nanoseconds(), 0)))
		if stats != nil {
			stats.promptTokens.Add(uint64(max(evaluation.Tokens, 0)))
			stats.cachedTokens.Add(uint64(max(evaluation.Cached, 0)))
			stats.promptNanos.Add(max(evaluation.Duration.Nanoseconds(), 0))
		}
		if onPromptEvaluated != nil {
			onPromptEvaluated(evaluation)
		}
	}
	if onDraftSpan := options.OnDraftSpan; options.SpeculativeDecode {
		options.OnDraftSpan = func(drafted, accepted int) {
			if stats != nil {
				stats.draftedTokens.Add(uint64(drafted))
				stats.acceptedDrafts.Add(uint64(accepted))
			}
			if onDraftSpan != nil {
				onDraftSpan(drafted, accepted)
			}
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
	// Decode runs from the prompt's first evaluation to the end, or the whole generation when no prompt was evaluated.
	finished := hardware.walls.Elapsed(clock)
	h.requests.addTokens(ctx, promptTokens.Load(), cachedTokens.Load(), outputTokens.Load(),
		time.Duration(promptDuration.Load()), time.Duration(finished-min(promptEvaluated.Load(), finished)))
	hardware.finish(ctx)
	if err != nil {
		h.generationErrors.Add(1)
	}
	if modelID, recipeID, ok := h.servingIdentity(recipe.TaskInference); ok {
		after := before
		if api, available := h.generator.(deviceExecutionAPI); available {
			// A cancelled request still reads what it moved: the device refuses a read under a cancelled context.
			after, _ = api.DeviceExecutionStats(context.WithoutCancel(ctx))
		}
		wall, wallErr := clock.Elapsed()
		if wallErr = errors.Join(wallErr, hardware.walls.Err); wallErr != nil {
			return ids, text, errors.Join(err, wallErr)
		}
		outcome, failure := executionOutcome(err)
		h.publishServing(ctx, runrecord.ServingObservation{
			Model: modelID, Recipe: recipeID, Task: recipe.TaskInference,
			Outcome: outcome, Failure: failure, StartedUnixNS: started.UnixNano(), MeasuredNS: wall,
			Usage: runrecord.ServingUsage{
				InputTokens: promptTokens.Load(), CachedTokens: min(cachedTokens.Load(), promptTokens.Load()), OutputTokens: outputTokens.Load(),
				InputBytes: uint64(len(prompt)), OutputBytes: uint64(len(text)),
			},
			Resources: runrecord.ServingResources{
				PeakDeviceBytes:   hardware.peakDeviceBytes(),
				HostToDeviceBytes: servingTransferDelta(before.HostToDeviceBytes, after.HostToDeviceBytes),
				DeviceToHostBytes: servingTransferDelta(before.DeviceToHostBytes, after.DeviceToHostBytes),
			},
			Phases: servingPhases(uint64(promptDuration.Load()), promptEvaluated.Load(), wall), Hardware: hardware.samples,
		})
	}
	return ids, text, err
}

func (h *Handler) models(response http.ResponseWriter, request *http.Request) {
	model := inference.ModelProperties{}
	if api, ok := h.generator.(modelPropertiesAPI); ok {
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

// health answers the public liveness probe with the served model and, when
// the generator runs on a device, its current and peak memory for the
// shell's device status dot, and the device's memory by owner for its bar.
func (h *Handler) health(response http.ResponseWriter, request *http.Request) {
	payload := map[string]any{"status": "ok", "model": h.config.ModelID}
	if api, ok := h.generator.(deviceMemoryAPI); ok {
		if stats, err := api.DeviceMemoryStats(request.Context()); err == nil {
			device := map[string]any{"current_bytes": stats.CurrentBytes, "peak_bytes": stats.PeakBytes}
			if owners, ok := h.generator.(deviceMemoryOwnersAPI); ok {
				if memory, err := owners.DeviceMemory(request.Context()); err == nil {
					device["total_bytes"], device["free_bytes"], device["owners"] = memory.TotalBytes, memory.FreeBytes, memory.Owners
				}
			}
			payload["device"] = device
		}
	}
	writeJSON(response, http.StatusOK, payload)
}
