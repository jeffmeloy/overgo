//go:build modeltest

package mediacapability

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/speechsynth"
	"overgo/internal/testutil"
)

// wholeTextReading is a short sentence and then one longer than a segment
// holds, so a reading of it is more than one segment and differs from a clip.
const wholeTextReading = "The tide went out before dawn. A grey heron stood at the edge of the harbour and watched the fishing boats leave one after another, each with a lamp swinging at its bow and a dog asleep on its nets."

// TestSpeechReadsWholeText holds the catalog's speech capability, on the real
// model, to reading a whole text aloud when the request names no frame limit.
// A text of more than one segment comes back as the WAV that Speak gives for
// the same text, voice, seed and temperature, having loaded the model. After
// the store is opened again the same request, served against a directory that
// holds no model, is answered with the same bytes and no load. A request
// stopped before it began returns the stop. The same text with a frame limit
// is still the bounded preview: a clip shorter than the reading, cut before
// the model ended it.
func TestSpeechReadsWholeText(t *testing.T) {
	directory := testutil.ModelArtifactDir(t, "pocket-tts")
	capability := Catalog[recipe.TaskSpeech]
	source, err := capability.Resolve(directory)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, source.Inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	stored := t.TempDir()
	store, err := overgodb.Open(stored)
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
	// serve gives the request to the catalog's executor against the model at
	// path and returns the recording, the WAV it publishes and what the run
	// measured: its load and the stages it ran.
	serve := func(ctx context.Context, path string, request map[string]any) (speechsynth.Audio, []byte, capabilityruntime.Measured, error) {
		t.Helper()
		raw, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		output, err := capability.Execute(ctx, store, path, modelrecipe.CapabilityEvidenceSelection{Program: program}, string(raw))
		if err != nil {
			return speechsynth.Audio{}, nil, capabilityruntime.Measured{}, err
		}
		measured, enveloped := output.(capabilityruntime.Measured)
		audio, spoken := measured.Output.(speechsynth.Audio)
		if !enveloped || !spoken {
			t.Fatalf("%s was answered with %T", raw, output)
		}
		content, err := OutputContent(audio)
		if err != nil {
			t.Fatal(err)
		}
		return audio, content.Data, measured, nil
	}

	reading := speechsynth.DocumentRequest{Text: wholeTextReading, Voice: "alba", Seed: 7, Temperature: 0.5}
	synth, err := speechsynth.LoadSynthesizer(directory)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := synth.PlanDocument(t.Context(), reading)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Segments) < 2 {
		t.Fatalf("the text is planned as %d segments, and a whole reading needs more than one to differ from a clip", len(plan.Segments))
	}
	direct, err := synth.Speak(t.Context(), reading)
	if err != nil {
		t.Fatal(err)
	}
	want, err := OutputContent(direct)
	if err != nil {
		t.Fatal(err)
	}

	request := map[string]any{"text": reading.Text, "voice": reading.Voice, "seed": reading.Seed, "temperature": reading.Temperature}
	whole, published, measured, err := serve(t.Context(), directory, request)
	if err != nil {
		t.Fatal(err)
	}
	if !whole.Complete || !direct.Complete || !reflect.DeepEqual(whole, direct) || !bytes.Equal(published, want.Data) || measured.Load <= 0 {
		t.Fatalf("a request with no frame limit gave %d samples (ended on its own: %v) after a load of %v; Speak gives %d for its %d segments",
			len(whole.PCM), whole.Complete, measured.Load, len(direct.PCM), len(plan.Segments))
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = nil
	store, err = overgodb.Open(stored)
	if err != nil {
		t.Fatal(err)
	}
	_, replayed, measured, err := serve(t.Context(), t.TempDir(), request)
	if err != nil {
		t.Fatalf("the store had answered the request, and with no model to load it was not answered again: %v", err)
	}
	if !bytes.Equal(replayed, want.Data) || measured.Load != 0 || len(measured.Phases) != 0 {
		t.Fatalf("the stored answer came back as %d bytes after a load of %v and %d stages; the reading published %d", len(replayed), measured.Load, len(measured.Phases), len(want.Data))
	}

	halted := map[string]any{"text": reading.Text, "voice": reading.Voice, "seed": reading.Seed + 1, "temperature": reading.Temperature}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if stopped, _, _, err := serve(ctx, directory, halted); !errors.Is(err, context.Canceled) || len(stopped.PCM) != 0 {
		t.Fatalf("a request stopped before it began returned %d samples: %v", len(stopped.PCM), err)
	}

	const previewFrames = 2
	preview := map[string]any{"text": reading.Text, "voice": reading.Voice, "seed": reading.Seed, "temperature": reading.Temperature, "max_frames": previewFrames}
	clip, _, _, err := serve(t.Context(), directory, preview)
	if err != nil {
		t.Fatal(err)
	}
	if len(clip.PCM) == 0 || len(clip.PCM) >= len(whole.PCM) || clip.Complete {
		t.Fatalf("the same text with a limit of %d frames gave %d samples (ended on its own: %v), and the whole reading holds %d", previewFrames, len(clip.PCM), clip.Complete, len(whole.PCM))
	}
}
