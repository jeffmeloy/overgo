package speechsynth

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// wholeTextRun is one speech program on one store, run as a served request
// with no frame limit is: under the key such a request is kept under, with
// the binding a test hands in.
type wholeTextRun struct {
	store      artifact.Repository
	model      artifact.ID
	definition recipe.Definition
	program    recipe.Program
}

func newWholeTextRun(t *testing.T, store artifact.Repository, name string) wholeTextRun {
	t.Helper()
	model := testutil.ArtifactID(t, artifact.KindModel, name)
	testutil.PublishArtifact(t, store, model)
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, model)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	return wholeTextRun{store: store, model: model, definition: definition, program: program}
}

// execute runs the request through the stages bind registers and returns the
// recording with the bytes the store keeps for it.
func (run wholeTextRun) execute(ctx context.Context, t *testing.T, request SynthesisRequest, bind func(*workflowruntime.Runtime) error) (Audio, *artifact.Content, error) {
	t.Helper()
	content, err := capabilityruntime.InputContent("overgo.speech-input.v1", request)
	if err != nil {
		t.Fatal(err)
	}
	input := run.definition.Inputs[0]
	audio, result, err := capabilityruntime.ExecuteMeasured[Audio](ctx, run.store, run.model, run.program, capabilityruntime.RunKey(run.definition, content),
		map[recipe.PortName]workflowruntime.Value{input.Name: workflowruntime.ArtifactValue(input.Data, request, content)}, bind)
	stored, _ := result.Outputs[run.definition.Outputs[0].Name].Single()
	return audio, stored.Content, err
}

// TestWholeTextBinding holds the rules of the binding that reads a whole text
// aloud, with a stand-in for the reading so that no model is needed.
//
// A request is admitted with no frame limit, as it is with a positive one, and
// still refused with a negative one. The reading is asked for with the
// request's own text, voice, seed and temperature. A reading that ended on its
// own is returned as it was read and stored as the WAV a speech request
// publishes; after the store is opened again the same request is answered with
// the same bytes and nothing is read aloud, and the store takes that WAV
// published again. A reading that was cut is refused, in words that say what
// to change, and since nothing of it was stored the next request reads it
// again. A request stopped before it began returns the stop and no samples,
// and the same request then succeeds. A binding with no synthesizer to read is
// refused when it is made.
func TestWholeTextBinding(t *testing.T) {
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if store != nil {
			if err := store.Close(); err != nil {
				t.Error(err)
			}
		}
	}()
	run := newWholeTextRun(t, store, "speech-whole-text-binding-fixture")
	const (
		rate = 24000
		// cutSeed marks the request whose reading the stand-in cuts short.
		cutSeed = 8
	)
	var asked []DocumentRequest
	read := func(request DocumentRequest) Audio {
		return Audio{PCM: []float32{0, 0.25, -0.25, 0.5}, SampleRate: rate, Channels: 1, Complete: request.Seed != cutSeed}
	}
	bind := func(runtime *workflowruntime.Runtime) error {
		return registerWholeText(runtime, run.model, func(_ context.Context, request DocumentRequest) (Audio, error) {
			asked = append(asked, request)
			return read(request), nil
		})
	}
	request := SynthesisRequest{Text: "The tide went out before dawn.", Voice: "alba", Seed: 7, Temperature: speakOrderTemperature}
	reading := DocumentRequest{Text: request.Text, Voice: request.Voice, Seed: request.Seed, Temperature: request.Temperature}
	for limit, admitted := range map[int]bool{-1: false, 0: true, 2: true} {
		limited := request
		limited.MaxFrames = limit
		if err := ValidateSynthesisRequest(limited); (err == nil) != admitted {
			t.Fatalf("a request with a frame limit of %d was admitted: %v, want %v (%v)", limit, err == nil, admitted, err)
		}
	}

	first, stored, err := run.execute(t.Context(), t, request, bind)
	if err != nil {
		t.Fatal(err)
	}
	if len(asked) != 1 || asked[0] != reading {
		t.Fatalf("the reading was asked for as %+v, want once as %+v", asked, reading)
	}
	if !reflect.DeepEqual(first, read(reading)) {
		t.Fatalf("the stages returned %+v, and the reading was %+v", first, read(reading))
	}
	want, err := WAVContent(first)
	if err != nil {
		t.Fatal(err)
	}
	if stored == nil {
		t.Fatal("the reading was returned and nothing of it was stored")
	}
	if kind := stored.Descriptor; kind.ID.Kind() != artifact.KindOutput || kind.MediaType != media.WAVMediaType ||
		kind.Schema != "overgo.encoded-audio.wav.v1" || !bytes.Equal(stored.Data, want.Data) {
		t.Fatalf("the reading was stored as %d bytes of %s %q under %q, want the %d of the WAV a speech request publishes: output %q under \"overgo.encoded-audio.wav.v1\"",
			len(stored.Data), kind.ID.Kind(), kind.MediaType, kind.Schema, len(want.Data), media.WAVMediaType)
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = nil
	store, err = overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	run.store = store
	replayed, kept, err := run.execute(t.Context(), t, request, bind)
	if err != nil {
		t.Fatal(err)
	}
	if len(asked) != 1 {
		t.Fatalf("a request the store had answered was read aloud again: %d readings", len(asked))
	}
	again, err := WAVContent(replayed)
	if err != nil {
		t.Fatal(err)
	}
	if kept == nil || !bytes.Equal(kept.Data, want.Data) || !bytes.Equal(again.Data, want.Data) || again.Descriptor != want.Descriptor {
		t.Fatalf("a request the store had answered came back as %d samples, and the first reading held %d", len(replayed.PCM), len(first.PCM))
	}
	// A served request publishes its answer's WAV in a run of its own, into the
	// store that holds the same bytes from the stage.
	published, err := artifact.NewDocumentBatch("speech-whole-text-binding/published", []artifact.Content{again}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(t.Context(), store, published); err != nil {
		t.Fatalf("the store that holds the reading from its stage refused the same WAV published again: %v", err)
	}

	cut := request
	cut.Seed = cutSeed
	for attempt := 2; attempt <= 3; attempt++ {
		refused, short, err := run.execute(t.Context(), t, cut, bind)
		if err == nil || !strings.Contains(err.Error(), "did not finish by itself") || !strings.Contains(err.Error(), "seed") || !strings.Contains(err.Error(), "temperature") {
			t.Fatalf("a reading that was cut was not refused in words that say what to change: %v", err)
		}
		if len(refused.PCM) != 0 || short != nil {
			t.Fatalf("a reading that was cut returned %d samples (stored: %v)", len(refused.PCM), short != nil)
		}
		if len(asked) != attempt {
			t.Fatalf("after %d requests in all the reading had been asked for %d times; a refused reading is read again", attempt, len(asked))
		}
	}

	stopped := request
	stopped.Seed = cutSeed + 1
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	halted, _, err := run.execute(ctx, t, stopped, bind)
	if !errors.Is(err, context.Canceled) || len(halted.PCM) != 0 || len(asked) != 3 {
		t.Fatalf("a request stopped before it began returned %d samples after %d readings: %v", len(halted.PCM), len(asked), err)
	}
	retried, _, err := run.execute(t.Context(), t, stopped, bind)
	if err != nil || len(asked) != 4 || !slices.Equal(retried.PCM, first.PCM) || !retried.Complete {
		t.Fatalf("the stopped request, sent again, gave %d samples after %d readings: %v", len(retried.PCM), len(asked), err)
	}

	runtime, err := workflowruntime.NewForProgram(store, run.program)
	if err != nil {
		t.Fatal(err)
	}
	if err := RegisterWholeText(runtime, run.model, nil); err == nil {
		t.Fatal("a binding with no synthesizer to read was accepted")
	}
}

// TestWholeTextStagesReadAsSpeak holds the binding a served request with no
// frame limit runs to the reading it is meant to be: on the real model, a text
// too long for one segment gives through the three stages, sample for sample,
// the recording Speak gives for the same text, voice, seed and temperature,
// and it ended on its own.
func TestWholeTextStagesReadAsSpeak(t *testing.T) {
	synth := loadArtifactSynthesizer(t)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	}()
	run := newWholeTextRun(t, store, "speech-whole-text-stages-fixture")
	request := SynthesisRequest{Text: speakOrderText, Voice: "alba", Seed: 7, Temperature: speakOrderTemperature}
	reading := DocumentRequest{Text: request.Text, Voice: request.Voice, Seed: request.Seed, Temperature: request.Temperature}
	plan, err := synth.PlanDocument(t.Context(), reading)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Segments) < 2 {
		t.Fatalf("the text is planned as %d segments, and a whole reading needs more than one to differ from a clip", len(plan.Segments))
	}
	staged, stored, err := run.execute(t.Context(), t, request, func(runtime *workflowruntime.Runtime) error {
		return RegisterWholeText(runtime, run.model, synth)
	})
	if err != nil {
		t.Fatal(err)
	}
	direct, err := synth.Speak(t.Context(), reading)
	if err != nil {
		t.Fatal(err)
	}
	if !staged.Complete || !direct.Complete || !reflect.DeepEqual(staged, direct) {
		same := 0
		for same < min(len(staged.PCM), len(direct.PCM)) && staged.PCM[same] == direct.PCM[same] {
			same++
		}
		t.Fatalf("the stages read %d segments as %d samples (ended on its own: %v), and Speak gives %d (%v); the first %d are the same",
			len(plan.Segments), len(staged.PCM), staged.Complete, len(direct.PCM), direct.Complete, same)
	}
	want, err := WAVContent(direct)
	if err != nil {
		t.Fatal(err)
	}
	if stored == nil || !bytes.Equal(stored.Data, want.Data) {
		t.Fatal("the stages stored other bytes than the WAV of what Speak gives")
	}
}
