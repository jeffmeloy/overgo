package main

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"overgo/internal/evaluation"
	"overgo/internal/runrecord"
)

// TestADependentMeasurementStopsWhenItsWarmUpFails holds a speed reading to the preparation it depends on (SQA-335):
// a warm-up that failed was printed and its dependent request run anyway, so a failure then a success left no durable
// failure. A failing unmeasured prompt is kept as a failed attempt, its measured reading is not taken and the failure
// is returned; a failing first warm request is kept and the warm reading not taken; and a cancelled warm-up is
// recorded cancelled and ends the readings.
func TestADependentMeasurementStopsWhenItsWarmUpFails(t *testing.T) {
	t.Parallel()
	lost := errors.New("the device lost the request")
	held := uint32(evaluation.PromptLengths[0] + lengthTokens)
	warming := &servedFixture{t: t, context: held, fail: func(_ context.Context, prompt string) error {
		if strings.HasPrefix(prompt, "["+warmingOpening+" of") {
			return lost
		}
		return nil
	}}
	observed, err := observeLengths(t.Context(), warming, io.Discard)
	measured := 0
	for _, prompt := range warming.prompts {
		if strings.HasPrefix(prompt, "["+measuredOpening+" of") {
			measured++
		}
	}
	if !errors.Is(err, lost) || measured != 0 || len(observed) == 0 || observed[0].Outcome != runrecord.OutcomeFailed {
		t.Fatalf("failing warm-ups ended %v with %d measured readings taken and %d attempts kept; want the failure returned, no reading taken and each failed warm-up kept", err, measured, len(observed))
	}

	first := &servedFixture{t: t, context: warmPromptTokens + warmTokens}
	first.fail = func(context.Context, string) error {
		if len(first.prompts) == 1 {
			return lost
		}
		return nil
	}
	warm, err := observeWarm(t.Context(), first, io.Discard)
	if !errors.Is(err, lost) || len(first.prompts) != 1 || len(warm) != 1 || warm[0].Outcome != runrecord.OutcomeFailed {
		t.Fatalf("a failing first warm request ended %v after %d requests, keeping %+v; want the failure returned, no second request and the first kept failed", err, len(first.prompts), warm)
	}

	stopped, stop := context.WithCancel(t.Context())
	stop()
	cancelled := &servedFixture{t: t, context: held, fail: func(ctx context.Context, _ string) error { return ctx.Err() }}
	observed, err = observeLengths(stopped, cancelled, io.Discard)
	if !errors.Is(err, context.Canceled) || len(cancelled.prompts) != 1 || len(observed) != 1 || observed[0].Outcome != runrecord.OutcomeCancelled || observed[0].Failure != "" {
		t.Fatalf("a cancelled warm-up ended %v after %d requests, keeping %+v; want it kept cancelled and no later request", err, len(cancelled.prompts), observed)
	}
}
