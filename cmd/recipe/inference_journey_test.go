package main

import (
	"bytes"
	"context"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/processmeasure"
	"overgo/internal/runrecord"
)

// requestMoments reads the three moments of a request out of what a
// measurement said of it.
var requestMoments = regexp.MustCompile(`the prompt was evaluated at (\S+), device memory was sampled at (\S+) and the first token came at (\S+)\n`)

// TestMeasurementSeparatesLoadPrefillAndVisibleOutput holds what a measurement
// of an active inference recipe says of where a first request's time goes.
//
// One measured request says how long its load took and names the parts of the
// load that named themselves. It says three moments of the request apart, each
// from the request's start: when the prompt was evaluated, when device memory
// had been sampled for it, and when the first token came. The fixture takes a
// tick of the clock between each, so the three differ, in that order, and the
// sample the scorecard reads as first output lies between the first two.
//
// Asked for the journey, the measurement says the same of a first load in the
// process, of a second load in it, of the same request again on a loaded
// model and of the verification's second request on a loaded model, in that
// order. It puts the first request to the model five times and the second
// once, and records no serving attempt.
func TestMeasurementSeparatesLoadPrefillAndVisibleOutput(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	directory, path, model, _, first := measuredFixture(t)
	// open loads the fixture, and one part of its load names itself.
	open := func(ctx context.Context) (servedModel, error) {
		named := processmeasure.Stage(ctx, "a part of the fixture's load")
		tick(t, tick(t, 0))
		named()
		tick(t, tick(t, 0))
		return model, nil
	}
	_, _, requests, err := measuredRequest(ctx, directory, path)
	if err != nil || len(requests) != 2 || requests[0].Prompt != first.Prompt {
		t.Fatalf("the recipe's verification ran the requests %+v (%v), want the fixture's two, the first %q", requests, err, first.Prompt)
	}
	request, other := requests[0], requests[1]
	var said bytes.Buffer
	observation, _, err := observeInference(ctx, open, request, &said)
	if err != nil {
		t.Fatal(err)
	}
	if text := said.String(); !strings.Contains(text, "recipe: the load took ") || !strings.Contains(text, "a part of the fixture's load") || !strings.Contains(text, "(no part names)") {
		t.Fatalf("the measurement said\n%s\nwant how long the load took, the part of it that named itself and the time no part names", text)
	}
	moments := requestMoments.FindStringSubmatch(said.String())
	if moments == nil {
		t.Fatalf("the measurement said\n%s\nwant the three moments of its request", said.String())
	}
	var evaluated, sampled, emitted time.Duration
	for at, moment := range []*time.Duration{&evaluated, &sampled, &emitted} {
		if *moment, err = time.ParseDuration(moments[at+1]); err != nil {
			t.Fatal(err)
		}
	}
	if evaluated <= 0 || sampled < evaluated || emitted <= sampled {
		t.Fatalf("the prompt was evaluated at %s, memory sampled at %s and the first token came at %s; want each after the one before, the first token strictly after the sample", evaluated, sampled, emitted)
	}
	// The scorecard reads the prefill sample's elapsed time as first output: it lies between the
	// prompt's evaluation and the moment after the sample, and so before the first token.
	for _, sample := range observation.Hardware {
		if elapsed := time.Duration(sample.ElapsedNS); sample.Stage == runrecord.ServingHardwarePrefill && (elapsed < evaluated || elapsed > sampled) {
			t.Fatalf("the prefill sample reads %s, want it between the prompt's evaluation at %s and %s", elapsed, evaluated, sampled)
		}
	}

	asked := len(model.prompts)
	said.Reset()
	if err := journeyInference(ctx, directory, path, open, &said); err != nil {
		t.Fatal(err)
	}
	ways := []string{
		"recipe: a first load in this process\n", "recipe: a second load in this process\n",
		"recipe: the same request again on a loaded model\n", "recipe: another request on a loaded model\n",
	}
	text, at := said.String(), 0
	for _, way := range ways {
		found := strings.Index(text[at:], way)
		if found < 0 {
			t.Fatalf("the journey lacks %q after its first %d bytes:\n%s", way, at, text)
		}
		at += found + len(way)
	}
	if loads, timed := strings.Count(text, "recipe: the load took "), len(requestMoments.FindAllString(text, -1)); loads != len(ways) || timed != len(ways) {
		t.Fatalf("the journey says %d loads and %d requests, want %d of each:\n%s", loads, timed, len(ways), text)
	}
	// Each load is followed by its request, and the two ways on a loaded model put the first request once more before theirs.
	if put, want := model.prompts[asked:], []string{request.Prompt, request.Prompt, request.Prompt, request.Prompt, request.Prompt, other.Prompt}; !slices.Equal(put, want) {
		t.Fatalf("the journey put %q to the model, want %q", put, want)
	}
	if recorded := servingAttempts(t, directory); len(recorded) != 0 {
		t.Fatalf("after one observed request and a journey the store holds %d serving attempts, want none: neither records one", len(recorded))
	}
}
