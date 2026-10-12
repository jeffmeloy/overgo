package mediacapability

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/speechsynth"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// TestSpeechReplaysWholeTextBeforeLoad holds that a speech request naming no
// frame limit is admitted as a reading of the whole text, and that when the
// store already holds its answer it is answered from the store before any
// model is loaded. The answer is put in the store by stand-in stages, under
// the key a served request is kept under and in the WAV a speech request
// publishes. The catalog's own executor is then given the request against a
// directory that holds no model, so a load could only fail: it must return the
// stored WAV's samples, having loaded nothing and run no stage. A limit
// written as zero is the same request and is answered the same way.
func TestSpeechReplaysWholeTextBeforeLoad(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	}()
	model := testutil.ArtifactID(t, artifact.KindModel, "speech-whole-text-replay-fixture")
	testutil.PublishArtifact(t, store, model)
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, model)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	request := speechsynth.SynthesisRequest{Text: "The tide went out before dawn.", Voice: "alba"}
	content, err := capabilityruntime.InputContent("overgo.speech-input.v1", request)
	if err != nil {
		t.Fatal(err)
	}
	const rate = 24000
	recording := speechsynth.Audio{PCM: []float32{0, 0.25, -0.25, 0.5}, SampleRate: rate, Channels: 1, Complete: true}
	want, err := OutputContent(recording)
	if err != nil {
		t.Fatal(err)
	}
	input := definition.Inputs[0]
	if _, _, err := capabilityruntime.ExecuteMeasured[speechsynth.Audio](t.Context(), store, model, program, capabilityruntime.RunKey(definition, content),
		map[recipe.PortName]workflowruntime.Value{input.Name: workflowruntime.ArtifactValue(input.Data, request, content)},
		func(runtime *workflowruntime.Runtime) error {
			return workflowruntime.RegisterContextPipeline(runtime, model,
				modelrecipe.ModuleSpeechTokenize, func(context.Context, speechsynth.SynthesisRequest) (string, error) { return "", nil },
				modelrecipe.ModuleSpeechGenerate, func(context.Context, string) (int, error) { return 0, nil },
				modelrecipe.ModuleSpeechDecode, func(context.Context, int) (speechsynth.Audio, error) { return recording, nil },
				func(audio speechsynth.Audio) (artifact.Content, error) { return OutputContent(audio) })
		}); err != nil {
		t.Fatal(err)
	}
	empty := t.TempDir()
	for _, raw := range []string{
		`{"text":"The tide went out before dawn.","voice":"alba"}`,
		`{"text":"The tide went out before dawn.","voice":"alba","max_frames":0}`,
	} {
		output, err := Catalog[recipe.TaskSpeech].Execute(t.Context(), store, empty, modelrecipe.CapabilityEvidenceSelection{Program: program}, raw)
		if err != nil {
			t.Fatalf("%s was not answered from the store: %v", raw, err)
		}
		measured, enveloped := output.(capabilityruntime.Measured)
		audio, spoken := measured.Output.(speechsynth.Audio)
		if !enveloped || !spoken {
			t.Fatalf("%s was answered with %T", raw, output)
		}
		got, err := OutputContent(audio)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got.Data, want.Data) || got.Descriptor != want.Descriptor {
			t.Fatalf("%s was answered with %d samples as %q, and the store holds %d as %q", raw, len(audio.PCM), got.Descriptor.Schema, len(recording.PCM), want.Descriptor.Schema)
		}
		if measured.Load != 0 || len(measured.Phases) != 0 {
			t.Fatalf("%s was answered after a load of %v and %d stages; the store held its answer", raw, measured.Load, len(measured.Phases))
		}
	}
}

// TestSpeechRoutesByFrameLimit holds which of its two executors a speech
// request is given to, with stand-ins that record the call. A request with no
// frame limit, or a limit of zero, is read whole. Any other goes to the
// bounded preview, which answers or refuses it as it always has: a positive
// limit, a negative one, and a request that does not decode at all. The
// executor chosen is given the request as it arrived, and what it returns,
// its error with it, is what the caller gets.
func TestSpeechRoutesByFrameLimit(t *testing.T) {
	type call struct{ executor, path, raw string }
	var calls []call
	failure := errors.New("the executor's own failure")
	record := func(name string) capabilityruntime.Executor {
		return func(_ context.Context, _ artifact.Repository, path string, _ modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
			calls = append(calls, call{executor: name, path: path, raw: raw})
			return name + " answered", failure
		}
	}
	execute := speechExecutor(record("preview"), record("whole"))
	for _, request := range []struct{ raw, executor string }{
		{`{"text":"Read me.","voice":"alba"}`, "whole"},
		{`{"text":"Read me.","voice":"alba","max_frames":0}`, "whole"},
		{`{"text":"Read me.","voice":"alba","seed":7,"temperature":0.5}`, "whole"},
		{`{"text":"Read me.","voice":"alba","max_frames":2}`, "preview"},
		{`{"text":"Read me.","voice":"alba","max_frames":-1}`, "preview"},
		{`{"text":"Read me.","voice":"alba","frames":0}`, "preview"},
		{`{"text":"Read me.","voice":"alba","max_frames":"0"}`, "preview"},
		{`{"text":`, "preview"},
		{``, "preview"},
	} {
		calls = nil
		output, err := execute(t.Context(), nil, "a model's directory", modelrecipe.CapabilityEvidenceSelection{}, request.raw)
		want := call{executor: request.executor, path: "a model's directory", raw: request.raw}
		if len(calls) != 1 || calls[0] != want {
			t.Fatalf("%q was given to %+v, want one call to %+v", request.raw, calls, want)
		}
		if output != request.executor+" answered" || !errors.Is(err, failure) {
			t.Fatalf("%q returned %v and %v, not what the %s executor gave", request.raw, output, err, request.executor)
		}
	}
}

// TestSpeechCapabilityBindsEachReadingToItsStages holds which stages the
// catalog's speech entry binds on each side of its router, with no model file.
// The entry is built over a loader that gives a synthesizer with no model in
// it, so each request is refused by the first stage that reaches for the
// model, and the two bindings refuse in different words: a request with no
// frame limit by the document plan that Speak begins with, and a request with
// a limit by the preview's tokenizer. With the bindings exchanged each request
// would get the other's refusal. Nothing but those words tells the bindings
// apart before a model is loaded, which is why the test reads them. Each
// request asks the loader for its model once, at the directory it was given.
func TestSpeechCapabilityBindsEachReadingToItsStages(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	}()
	model := testutil.ArtifactID(t, artifact.KindModel, "speech-binding-sides-fixture")
	testutil.PublishArtifact(t, store, model)
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, model)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	var loaded []string
	capability := speechCapability(func(path string) (*speechsynth.Synthesizer, error) {
		loaded = append(loaded, path)
		return &speechsynth.Synthesizer{}, nil
	})
	const directory = "a model's directory"
	for _, request := range []struct{ raw, reading, refusal string }{
		{`{"text":"Read me.","voice":"alba"}`, "the whole reading", "speech document: incomplete context or synthesizer"},
		{`{"text":"Read me.","voice":"alba","max_frames":2}`, "the bounded preview", "speechsynth: incomplete synthesizer"},
	} {
		loaded = nil
		_, err := capability.Execute(t.Context(), store, directory, modelrecipe.CapabilityEvidenceSelection{Program: program}, request.raw)
		if err == nil || !strings.Contains(err.Error(), request.refusal) {
			t.Fatalf("%s was not refused as %s refuses a synthesizer with no model (%q): %v", request.raw, request.reading, request.refusal, err)
		}
		if len(loaded) != 1 || loaded[0] != directory {
			t.Fatalf("%s asked for its model at %q, want once at %q", request.raw, loaded, directory)
		}
	}
}
