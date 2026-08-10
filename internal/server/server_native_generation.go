package server

import (
	"errors"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"time"

	"overgo/internal/inference"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

type nativeCompletionPlan struct {
	handler         *Handler
	request         *http.Request
	body            nativeCompletionRequest
	prompts         []nativePrompt
	sampler         *sampling.Sampler
	maxTokens       int
	stops           []string
	settings        map[string]any
	slotID          int
	lora            []inference.LoRAScale
	loraConfigured  bool
	projectedInputs *inference.ProjectedInputs
}

func (plan *nativeCompletionPlan) run(
	prompt nativePrompt,
	index int,
	returnTokens bool,
	onChunk func(nativeCompletionChunk) error,
	onPromptProgress func(nativePromptProgress) error,
) (nativeCompletionResponse, error) {
	sampler, err := samplerForChoice(plan.sampler, index)
	if err != nil {
		return nativeCompletionResponse{}, err
	}
	return plan.handler.runNativeCompletion(
		plan, prompt, sampler, index, returnTokens, onChunk, onPromptProgress,
	)
}

func (h *Handler) runNativeCompletion(
	plan *nativeCompletionPlan,
	prompt nativePrompt,
	sampler *sampling.Sampler,
	index int,
	returnTokens bool,
	onChunk func(nativeCompletionChunk) error,
	onPromptProgress func(nativePromptProgress) error,
) (nativeCompletionResponse, error) {
	ctx := plan.request.Context()
	body := plan.body
	maxTokens, stops := plan.maxTokens, plan.stops
	started := time.Now()
	var output strings.Builder
	filter := newStopFilter(stops)
	generated := make([]tokenizer.TokenID, 0, maxTokens)
	probabilities := make([]nativeTokenProbability, 0, maxTokens)
	promptTokensEstimate := 0
	promptEvaluation := inference.PromptEvaluation{}
	if prompt.TokenIDs != nil {
		promptTokensEstimate = len(prompt.TokenIDs)
	} else if api, ok := h.generator.(TokenizationAPI); ok {
		if promptIDs, err := api.TokenizeText(prompt.Text, true, false); err == nil {
			promptTokensEstimate = len(promptIDs)
		}
	}
	if onPromptProgress != nil {
		if err := onPromptProgress(nativePromptProgress{
			Total: promptTokensEstimate,
		}); err != nil {
			return nativeCompletionResponse{}, err
		}
	}
	var promptProgressErr error
	var predictionStarted time.Time
	timeLimitReached := false
	indentationLimitReached := false
	ids, _, err := h.generate(
		ctx,
		plan.slotID,
		prompt.Text,
		inference.GenerateOptions{
			MaxNewTokens:    maxTokens,
			Sampler:         sampler,
			DeviceGreedy:    body.NProbs == 0,
			StopSequences:   stops,
			ContextShift:    h.config.ContextShift,
			KeepTokens:      body.NKeep,
			DiscardTokens:   body.NDiscard,
			PromptTokenIDs:  prompt.TokenIDs,
			CachePrompt:     body.CachePrompt != nil && *body.CachePrompt,
			MinCacheReuse:   body.NCacheReuse,
			LoRA:            plan.lora,
			LoRAConfigured:  plan.loraConfigured,
			ProjectedInputs: plan.projectedInputs,
			PostSamplingProbabilities: func() int {
				if body.PostSamplingProbs {
					return body.NProbs
				}
				return 0
			}(),
			OnPromptEvaluated: func(evaluation inference.PromptEvaluation) {
				promptEvaluation = evaluation
				if onPromptProgress != nil {
					promptProgressErr = onPromptProgress(nativePromptProgress{
						Total:     evaluation.Tokens,
						Cache:     evaluation.Cached,
						Processed: evaluation.Tokens,
						TimeMS:    evaluation.Duration.Milliseconds(),
					})
				}
			},
			OnToken: func(event inference.TokenEvent) error {
				if promptProgressErr != nil {
					return promptProgressErr
				}
				generated = append(generated, event.ID)
				var probability *nativeTokenProbability
				if body.NProbs > 0 {
					var item nativeTokenProbability
					var probabilityErr error
					if body.PostSamplingProbs {
						item, probabilityErr =
							h.nativePostSamplingProbability(event)
					} else {
						item, probabilityErr = h.nativeTokenProbability(
							event.ID,
							event.Logits,
							body.NProbs,
						)
					}
					if probabilityErr != nil {
						return probabilityErr
					}
					probabilities = append(probabilities, item)
					probability = &probabilities[len(probabilities)-1]
				}
				piece := filter.Accept(event.Piece)
				previousLength := output.Len()
				output.WriteString(piece)
				if body.NIndent > 0 {
					if trimmed, stop := enforceNativeIndentation(
						output.String(),
						body.NIndent,
					); stop {
						trimmed = strings.Clone(trimmed)
						output.Reset()
						output.WriteString(trimmed)
						indentationLimitReached = true
						if previousLength < len(trimmed) {
							piece = trimmed[previousLength:]
						} else {
							piece = ""
						}
					}
				}
				if onChunk != nil && piece != "" {
					chunk := nativeCompletionChunk{
						Index:           index,
						Content:         piece,
						Tokens:          []tokenizer.TokenID{event.ID},
						Stop:            false,
						IDSlot:          -1,
						TokensPredicted: len(generated),
						TokensEvaluated: promptTokensEstimate,
					}
					if body.TimingsPerToken {
						timings := measuredNativeTimings(
							promptTokensEstimate,
							len(generated),
							promptEvaluation,
							time.Since(started),
						)
						chunk.Timings = &timings
					}
					if probability != nil {
						chunk.CompletionProbabilities =
							[]nativeTokenProbability{*probability}
					}
					return onChunk(chunk)
				}
				return ctx.Err()
			},
			ShouldStop: func(event inference.TokenEvent) bool {
				if indentationLimitReached {
					return true
				}
				now := time.Now()
				if predictionStarted.IsZero() {
					predictionStarted = now
				}
				if body.TMaxPredictMS > 0 &&
					strings.Contains(event.Piece, "\n") &&
					now.Sub(predictionStarted) >
						time.Duration(body.TMaxPredictMS)*time.Millisecond {
					timeLimitReached = true
					return true
				}
				return false
			},
		},
	)
	if err != nil {
		return nativeCompletionResponse{}, err
	}
	if promptProgressErr != nil {
		return nativeCompletionResponse{}, promptProgressErr
	}
	promptTokens := len(ids) - len(generated)
	if pending := filter.Flush(); pending != "" {
		previousLength := output.Len()
		output.WriteString(pending)
		if body.NIndent > 0 {
			if trimmed, stop := enforceNativeIndentation(output.String(), body.NIndent); stop {
				trimmed = strings.Clone(trimmed)
				output.Reset()
				output.WriteString(trimmed)
				indentationLimitReached = true
				if previousLength < len(trimmed) {
					pending = trimmed[previousLength:]
				} else {
					pending = ""
				}
			}
		}
		if onChunk != nil && pending != "" {
			if err := onChunk(nativeCompletionChunk{
				Index:           index,
				Content:         pending,
				Tokens:          []tokenizer.TokenID{},
				Stop:            false,
				IDSlot:          -1,
				TokensPredicted: len(generated),
				TokensEvaluated: promptTokens,
			}); err != nil {
				return nativeCompletionResponse{}, err
			}
		}
	}
	stopType := "eos"
	if filter.Stopped() {
		stopType = "word"
	} else if indentationLimitReached ||
		timeLimitReached ||
		len(generated) >= maxTokens {
		stopType = "limit"
	}
	timings := measuredNativeTimings(
		promptTokens,
		len(generated),
		promptEvaluation,
		time.Since(started),
	)
	truncated := false
	if api, ok := h.generator.(ModelPropertiesAPI); ok {
		contextLength := api.ModelProperties().ContextLength
		truncated = contextLength > 0 && len(ids) > int(contextLength)
	}
	tokens := []tokenizer.TokenID{}
	if returnTokens || onChunk != nil {
		tokens = append(tokens, generated...)
	}
	content := output.String()
	return nativeCompletionResponse{
		Index:                   index,
		Content:                 content,
		Tokens:                  tokens,
		IDSlot:                  plan.slotID,
		Stop:                    true,
		Model:                   h.config.ModelID,
		TokensPredicted:         len(generated),
		TokensEvaluated:         promptTokens,
		GenerationSettings:      plan.settings,
		Prompt:                  prompt.Response,
		HasNewLine:              strings.Contains(content, "\n"),
		Truncated:               truncated,
		StopType:                stopType,
		StoppingWord:            filter.StoppingWord(),
		TokensCached:            promptEvaluation.Cached,
		Timings:                 timings,
		CompletionProbabilities: probabilities,
	}, nil
}

func nativeTimings(promptTokens, generatedTokens int, elapsed time.Duration) nativeCompletionTimings {
	milliseconds := float64(elapsed) / float64(time.Millisecond)
	result := nativeCompletionTimings{
		PromptN:     promptTokens,
		PredictedN:  generatedTokens,
		PredictedMS: milliseconds,
	}
	if generatedTokens > 0 && milliseconds > 0 {
		result.PredictedPerTokenMS = milliseconds / float64(generatedTokens)
		result.PredictedPerSecond = float64(generatedTokens) * 1000 / milliseconds
	}
	return result
}

func enforceNativeIndentation(content string, minimum int) (string, bool) {
	if minimum <= 0 {
		return content, false
	}
	for searchFrom := 0; ; {
		newline := strings.IndexByte(content[searchFrom:], '\n')
		if newline < 0 {
			return content, false
		}
		lineStart := searchFrom + newline + 1
		position := lineStart
		for position < len(content) &&
			(content[position] == ' ' || content[position] == '\t') {
			position++
		}
		if position == len(content) {
			return content, false
		}
		if position-lineStart < minimum {
			return content[:position], true
		}
		searchFrom = position
	}
}

func (h *Handler) nativeTokenProbability(
	selected tokenizer.TokenID,
	logits []float32,
	n int,
) (nativeTokenProbability, error) {
	if n <= 0 {
		return nativeTokenProbability{}, nil
	}
	if int(selected) < 0 || int(selected) >= len(logits) {
		return nativeTokenProbability{}, fmt.Errorf(
			"selected token %d is outside %d logits",
			selected,
			len(logits),
		)
	}
	pieces, ok := h.generator.(TokenPieceAPI)
	if !ok {
		return nativeTokenProbability{}, errors.New(
			"token probability text is unavailable",
		)
	}
	maximum := math.Inf(-1)
	for _, logit := range logits {
		if float64(logit) > maximum {
			maximum = float64(logit)
		}
	}
	total := 0.0
	for _, logit := range logits {
		total += math.Exp(float64(logit) - maximum)
	}
	if total <= 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		return nativeTokenProbability{}, errors.New(
			"token probability softmax normalization is invalid",
		)
	}
	logNormalization := maximum + math.Log(total)
	// Full-vocabulary Shannon entropy of the model's next-token distribution,
	// in nats. This is a distribution-free functional of the empirical softmax
	// (H = sum p_i * -log p_i, with -log p_i = logNormalization - logit_i >= 0);
	// it assumes nothing about the shape of the logits.
	entropy := 0.0
	for _, logit := range logits {
		surprisal := logNormalization - float64(logit)
		entropy += math.Exp(-surprisal) * surprisal
	}
	indices := make([]int, len(logits))
	for index := range indices {
		indices[index] = index
	}
	sort.Slice(indices, func(left, right int) bool {
		leftLogit := logits[indices[left]]
		rightLogit := logits[indices[right]]
		if leftLogit == rightLogit {
			return indices[left] < indices[right]
		}
		return leftLogit > rightLogit
	})
	if n < len(indices) {
		indices = indices[:n]
	}
	itemFor := func(id tokenizer.TokenID) (nativeTokenLogProbability, error) {
		piece, err := pieces.TokenPiece(id)
		if err != nil {
			return nativeTokenLogProbability{}, err
		}
		bytes := []byte(piece)
		byteValues := make([]int, len(bytes))
		for index, value := range bytes {
			byteValues[index] = int(value)
		}
		return nativeTokenLogProbability{
			ID:      id,
			Token:   piece,
			Bytes:   byteValues,
			LogProb: float64(logits[int(id)]) - logNormalization,
		}, nil
	}
	selectedItem, err := itemFor(selected)
	if err != nil {
		return nativeTokenProbability{}, err
	}
	result := nativeTokenProbability{
		ID:          selectedItem.ID,
		Token:       selectedItem.Token,
		Bytes:       selectedItem.Bytes,
		TopLogProbs: make([]nativeTokenLogProbability, 0, len(indices)),
	}
	selectedLogProbability := selectedItem.LogProb
	result.LogProb = &selectedLogProbability
	result.Entropy = &entropy
	for _, index := range indices {
		item, itemErr := itemFor(tokenizer.TokenID(index))
		if itemErr != nil {
			return nativeTokenProbability{}, itemErr
		}
		result.TopLogProbs = append(result.TopLogProbs, item)
	}
	return result, nil
}

func (h *Handler) nativePostSamplingProbability(
	event inference.TokenEvent,
) (nativeTokenProbability, error) {
	pieces, ok := h.generator.(TokenPieceAPI)
	if !ok {
		return nativeTokenProbability{}, errors.New(
			"token probability text is unavailable",
		)
	}
	itemFor := func(id tokenizer.TokenID) (string, []int, error) {
		piece, err := pieces.TokenPiece(id)
		if err != nil {
			return "", nil, err
		}
		raw := []byte(piece)
		bytes := make([]int, len(raw))
		for index, value := range raw {
			bytes[index] = int(value)
		}
		return piece, bytes, nil
	}
	piece, bytes, err := itemFor(event.ID)
	if err != nil {
		return nativeTokenProbability{}, err
	}
	selectedProbability := event.SelectedProbability
	result := nativeTokenProbability{
		ID:       event.ID,
		Token:    piece,
		Bytes:    bytes,
		Prob:     &selectedProbability,
		TopProbs: make([]nativeTokenProbabilityValue, 0, len(event.TopProbabilities)),
	}
	for _, item := range event.TopProbabilities {
		itemPiece, itemBytes, itemErr := itemFor(tokenizer.TokenID(item.ID))
		if itemErr != nil {
			return nativeTokenProbability{}, itemErr
		}
		result.TopProbs = append(result.TopProbs, nativeTokenProbabilityValue{
			ID:    tokenizer.TokenID(item.ID),
			Token: itemPiece,
			Bytes: itemBytes,
			Prob:  item.Probability,
		})
	}
	return result, nil
}

func measuredNativeTimings(
	promptTokens int,
	generatedTokens int,
	evaluation inference.PromptEvaluation,
	elapsed time.Duration,
) nativeCompletionTimings {
	result := nativeTimings(promptTokens, generatedTokens, elapsed)
	result.CacheN = evaluation.Cached
	result.PromptN = max(promptTokens-evaluation.Cached, 0)
	result.PromptMS = float64(evaluation.Duration) / float64(time.Millisecond)
	if result.PromptN > 0 && result.PromptMS > 0 {
		result.PromptPerTokenMS = result.PromptMS / float64(result.PromptN)
		result.PromptPerSecond = float64(result.PromptN) * 1000 / result.PromptMS
	}
	predictedDuration := elapsed - evaluation.Duration
	if predictedDuration < 0 {
		predictedDuration = 0
	}
	result.PredictedMS = float64(predictedDuration) / float64(time.Millisecond)
	if generatedTokens > 0 && result.PredictedMS > 0 {
		result.PredictedPerTokenMS = result.PredictedMS / float64(generatedTokens)
		result.PredictedPerSecond =
			float64(generatedTokens) * 1000 / result.PredictedMS
	}
	return result
}

func (h *Handler) streamNativeCompletion(
	response http.ResponseWriter,
	plan *nativeCompletionPlan,
) {
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSynchronizedSSE(response, flusher)
	stopHeartbeat := stream.startHeartbeat(
		plan.request.Context(),
		time.Duration(nativeSSEPingInterval(plan.body))*time.Second,
	)
	defer stopHeartbeat()
	resultIndex := 0
	for _, prompt := range plan.prompts {
		for range plan.body.NCmpl {
			result, err := plan.run(
				prompt,
				resultIndex,
				true,
				func(chunk nativeCompletionChunk) error {
					if err := stream.write(chunk); err != nil {
						return err
					}
					return plan.request.Context().Err()
				},
				func(progress nativePromptProgress) error {
					if !plan.body.ReturnProgress {
						return nil
					}
					return stream.write(nativeCompletionChunk{
						Index:           resultIndex,
						Content:         "",
						Tokens:          []tokenizer.TokenID{},
						Stop:            false,
						IDSlot:          -1,
						TokensPredicted: 0,
						TokensEvaluated: progress.Processed,
						PromptProgress:  &progress,
					})
				},
			)
			if err != nil {
				_ = emitGenerationError(stream.write, err)
				return
			}
			// native llama.cpp stream terminates with full metadata
			// envelope, but content and tokens in that final event are empty
			// because they were already delivered by partial events
			result.Content = ""
			result.Tokens = []tokenizer.TokenID{}
			var final any = result
			if len(plan.body.ResponseFields) > 0 {
				projected, projectErr := projectNativeResponse(result, plan.body.ResponseFields)
				if projectErr != nil {
					_ = stream.write(errorEnvelope("server_error", projectErr.Error()))
					return
				}
				final = projected
			}
			if err := stream.write(final); err != nil {
				return
			}
			resultIndex++
		}
	}
}
