package latentvideo

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/testcheck"
)

// countedCard is a card encoder that counts its closes and encodes nothing.
type countedCard struct{ closes *int }

func (card countedCard) encode(context.Context, []int) ([]float32, error) {
	return nil, errors.New("counted card: encodes nothing")
}

func (card countedCard) Close(context.Context) error {
	*card.closes++
	return nil
}

// TestCardEncodersCloseWithTheIdleSessions holds ReleaseCardEncoders: a text encoder kept on the card for every
// session that reads it is closed and its place left for the next prompt to open again, and one closed is not closed
// twice. Wan's encoder had stayed on the card when its sessions closed, and kept the card's turn from a waiter
// (card-handover/encoder-release).
func TestCardEncodersCloseWithTheIdleSessions(t *testing.T) {
	forgetPromptEncoders(t)
	t.Cleanup(func() { forgetPromptEncoders(t) })
	closes := 0
	encoder := &promptEncoder{card: countedCard{&closes}}
	promptEncoders.Lock()
	promptEncoders.byKey[promptEncoderKey{content: "kept on the card"}] = encoder
	promptEncoders.Unlock()
	if closed := testcheck.Must(ReleaseCardEncoders(t.Context()))(t); closed != 1 || closes != 1 || encoder.card != nil {
		t.Fatalf("the release closed %d encoders, the card's Close ran %d times and the encoder holds %v; want the one closed once and its place empty", closed, closes, encoder.card)
	}
	if closed := testcheck.Must(ReleaseCardEncoders(t.Context()))(t); closed != 0 || closes != 1 {
		t.Fatalf("a second release closed %d encoders and Close ran %d times in all; want none closed again", closed, closes)
	}
}
