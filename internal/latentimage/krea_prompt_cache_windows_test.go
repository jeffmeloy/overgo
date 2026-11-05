//go:build windows

package latentimage

import (
	"bytes"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
)

// TestAKreaPromptMetAgainPlacesNoTextEncoder: a Krea session given back a prompt it has drawn keeps that prompt's text
// conditioning, so it copies to the card less than one layer of the text encoder holds where the prompt it left for
// placed the whole encoder and the fusion, and draws the picture its first encoding drew, byte for byte.
func TestAKreaPromptMetAgainPlacesNoTextEncoder(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	first := Request{Prompt: "a red fox licking a vanilla ice cream cone in snow", Width: 256, Height: 256, Steps: 2, Seed: 42}
	other := first
	other.Prompt = "a lighthouse on a cliff at dusk"
	generator, err := LoadGenerator(ctx, kreaDirOrSkip(t), kreaProfileOrSkip(t), first)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := generator.Close(ctx); err != nil {
			t.Errorf("close: %v", err)
		}
	}()
	fresh, _ := runResidentGeneration(t, ctx, generator, first)
	// copied is what giving the session request copies to the card, and how long it takes.
	copied := func(request Request) (uint64, time.Duration) {
		t.Helper()
		before, err := generator.pipeline.runtime.Stats(ctx)
		if err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		if err := generator.Reset(ctx, request); err != nil {
			t.Fatal(err)
		}
		wall := time.Since(started)
		after, err := generator.pipeline.runtime.Stats(ctx)
		if err != nil {
			t.Fatal(err)
		}
		return after.Execution.HostToDeviceBytes - before.Execution.HostToDeviceBytes, wall
	}
	encoded, encodedWall := copied(other)
	runResidentGeneration(t, ctx, generator, other)
	again, againWall := copied(first)
	picture, _ := runResidentGeneration(t, ctx, generator, first)
	selected := generator.pipeline.Encoder.E.SelectLayers
	layers := uint64(selected[len(selected)-1])
	if again*layers >= encoded {
		t.Fatalf("the prompt met again copied %d bytes to the card, the new prompt %d through %d encoder layers and the fusion; want under one layer's share", again, encoded, layers)
	}
	t.Logf("a new prompt copied %d bytes in %s, the prompt met again %d bytes in %s", encoded, encodedWall, again, againWall)
	if !bytes.Equal(picture.Data, fresh.Data) {
		t.Fatal("the prompt met again drew another picture than its first encoding drew")
	}
}
