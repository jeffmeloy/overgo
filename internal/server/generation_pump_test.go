package server

import (
	"reflect"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/testcheck"
)

func TestGenerationPumpFiltersCountsAndFlushes(t *testing.T) {
	t.Parallel()
	var pieces []string
	pump := newGenerationPump([]string{"STOP"}, func(piece string) error {
		pieces = append(pieces, piece)
		return nil
	})
	for _, piece := range []string{"aST", "OPtail", "ignored"} {
		testcheck.NoError(t, pump.accept(inference.TokenEvent{Piece: piece}))
	}
	testcheck.NoError(t, pump.flush())
	if pump.text() != "a" || pump.generated != 3 || pump.completion != 2 ||
		!pump.stopped() || pump.stoppingWord() != "STOP" ||
		(protocolGenerationResult{pump: pump}).end(2, inference.ChatMessage{}) != turnStopped {
		t.Fatalf("pump = text %q generated %d completion %d stopped %v word %q",
			pump.text(), pump.generated, pump.completion, pump.stopped(), pump.stoppingWord())
	}
	if !reflect.DeepEqual(pieces, []string{"a", "", ""}) {
		t.Fatalf("pieces = %#v", pieces)
	}
}
