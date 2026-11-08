package main

import (
	"testing"
	"time"

	"overgo/internal/processmeasure"
)

// TestAGenerationRecordsItsStageTimes holds a measured request's record to where its wall went: on 2026-11-08 LiveEdit
// took about 88 s a clip against its paper's 7.89 s and no record said how much of it the CPU text encoder and the
// autoencoder took. The ledger's named parts are recorded with what none names as one more, so they sum to the wall,
// and a request no ledger timed records none.
func TestAGenerationRecordsItsStageTimes(t *testing.T) {
	t.Parallel()
	ctx, stages := processmeasure.WithStages(t.Context())
	started := time.Now()
	for _, name := range []string{"text encoder run", "clip denoised", "clip decoded"} {
		processmeasure.Stage(ctx, name)()
	}
	wall := time.Since(started) + time.Millisecond
	parts := stagedParts(stages, wall)
	var sum time.Duration
	for _, part := range parts {
		sum += part.Wall
	}
	if len(parts) != 4 || parts[3].Name != unnamedPart || sum != wall || parts[0].Name != "text encoder run" {
		t.Fatalf("the parts read %+v summing to %s, want the three named and what none names, summing to %s", parts, sum, wall)
	}
	if parts := stagedParts(nil, wall); parts != nil {
		t.Fatalf("a request no ledger timed records %+v", parts)
	}
}
