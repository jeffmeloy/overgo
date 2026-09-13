package latentvideo

import (
	"context"
	"errors"
	"testing"
)

type encoderCancellationAfterEntry struct {
	context.Context
	entered bool
	cancel  context.CancelCauseFunc
}

func (ctx *encoderCancellationAfterEntry) Err() error {
	if ctx.entered {
		ctx.cancel(context.Canceled)
	}
	ctx.entered = true
	return ctx.Context.Err()
}

func TestTextConditioningCancellationBeforeIO(t *testing.T) {
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	// Missing artifact paths make any attempted acquisition observable as a
	// different error; cancellation must be returned before those reads.
	if _, err := TextConditioning(ctx, TextConditioningSpec{}, "prompt"); !errors.Is(err, context.Canceled) {
		t.Fatal("text conditioning attempted I/O after cancellation", err)
	}
	if _, _, err := promptContexts(ctx, TextConditioningSpec{}, WanRequest{}); !errors.Is(err, context.Canceled) {
		t.Fatal("prompt pair attempted I/O after cancellation", err)
	}
	if _, _, _, err := RawTextRows(ctx, TextConditioningSpec{}, "prompt"); !errors.Is(err, context.Canceled) {
		t.Fatal("raw text rows attempted I/O after cancellation", err)
	}
	if _, _, err := EncodeTokensStreamed(ctx, "", EncoderPlan{}, nil, nil); !errors.Is(err, context.Canceled) {
		t.Fatal("streamed encoder attempted I/O after cancellation", err)
	}
}
