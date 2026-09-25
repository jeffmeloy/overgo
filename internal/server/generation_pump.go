package server

import (
	"context"
	"strings"

	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

// generationPump: stop filtering plus token accounting
type generationPump struct {
	filter     *stopFilter
	emit       func(string) error
	output     strings.Builder
	generated  int
	completion int
	evaluation inference.PromptEvaluation
	// usage: the provider's accounting when the generator reports one (a
	// hosted turn); nil for a local runner, whose ids are the count.
	usage *inference.Usage
	// limit: an optional cut of the whole output; limited once it cut.
	limit   func(string) (string, bool)
	limited bool
}

func newGenerationPump(stops []string, emit func(string) error) *generationPump {
	return &generationPump{filter: newStopFilter(stops), emit: emit}
}

func (pump *generationPump) accept(event inference.TokenEvent) error {
	pump.generated++
	if !pump.filter.Stopped() {
		pump.completion++
	}
	piece := pump.keep(pump.filter.Accept(event.Piece))
	if pump.emit != nil {
		return pump.emit(piece)
	}
	return nil
}

func (pump *generationPump) flush() error {
	piece := pump.keep(pump.filter.Flush())
	if piece != "" && pump.emit != nil {
		return pump.emit(piece)
	}
	return nil
}

// keep appends a released piece; a limit that cuts the output back keeps
// only the part of the piece that survives it.
func (pump *generationPump) keep(piece string) string {
	previous := pump.output.Len()
	pump.output.WriteString(piece)
	if pump.limit == nil {
		return piece
	}
	trimmed, cut := pump.limit(pump.output.String())
	if !cut {
		return piece
	}
	trimmed = strings.Clone(trimmed)
	pump.output.Reset()
	pump.output.WriteString(trimmed)
	pump.limited = true
	if previous < len(trimmed) {
		return trimmed[previous:]
	}
	return ""
}

func (pump *generationPump) text() string {
	return pump.output.String()
}

func (pump *generationPump) stopped() bool {
	return pump.filter.Stopped()
}

func (pump *generationPump) stoppingWord() string {
	return pump.filter.StoppingWord()
}

func (pump *generationPump) finishReason(maxTokens int, stopped, length string) string {
	if !pump.stopped() && pump.completion >= maxTokens {
		return length
	}
	return stopped
}

func (h *Handler) generateWithPump(
	ctx context.Context,
	session *requestSession,
	prompt string,
	options inference.GenerateOptions,
	stops []string,
	emit func(string) error,
) ([]tokenizer.TokenID, *generationPump, error) {
	pump := newGenerationPump(stops, emit)
	ids, err := h.pumpGeneration(ctx, session, prompt, options, pump)
	return ids, pump, err
}

// pumpGeneration runs one generation through a caller's pump. A caller's
// OnToken sees each token before the pump filters and counts it.
func (h *Handler) pumpGeneration(
	ctx context.Context,
	session *requestSession,
	prompt string,
	options inference.GenerateOptions,
	pump *generationPump,
) ([]tokenizer.TokenID, error) {
	options.StopSequences = pump.filter.stops
	onToken := options.OnToken
	options.OnToken = func(event inference.TokenEvent) error {
		if onToken != nil {
			if err := onToken(event); err != nil {
				return err
			}
		}
		return pump.accept(event)
	}
	options.OnUsage = func(usage inference.Usage) { pump.usage = &usage }
	onPromptEvaluated := options.OnPromptEvaluated
	options.OnPromptEvaluated = func(evaluation inference.PromptEvaluation) {
		pump.evaluation = evaluation
		if onPromptEvaluated != nil {
			onPromptEvaluated(evaluation)
		}
	}
	ids, _, err := h.generate(ctx, session, prompt, options)
	if err != nil {
		return ids, err
	}
	return ids, pump.flush()
}
