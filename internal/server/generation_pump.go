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
}

func newGenerationPump(stops []string, emit func(string) error) *generationPump {
	return &generationPump{filter: newStopFilter(stops), emit: emit}
}

func (pump *generationPump) accept(event inference.TokenEvent) error {
	pump.generated++
	if !pump.filter.Stopped() {
		pump.completion++
	}
	piece := pump.filter.Accept(event.Piece)
	pump.output.WriteString(piece)
	if pump.emit != nil {
		return pump.emit(piece)
	}
	return nil
}

func (pump *generationPump) flush() error {
	piece := pump.filter.Flush()
	pump.output.WriteString(piece)
	if piece != "" && pump.emit != nil {
		return pump.emit(piece)
	}
	return nil
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
	slotID int,
	prompt string,
	options inference.GenerateOptions,
	stops []string,
	emit func(string) error,
) ([]tokenizer.TokenID, *generationPump, error) {
	pump := newGenerationPump(stops, emit)
	options.StopSequences = stops
	options.OnToken = pump.accept
	ids, _, err := h.generate(ctx, slotID, prompt, options)
	if err != nil {
		return ids, pump, err
	}
	if err := pump.flush(); err != nil {
		return ids, pump, err
	}
	return ids, pump, nil
}
