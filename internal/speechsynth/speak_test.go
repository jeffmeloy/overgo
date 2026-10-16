//go:build modeltest

package speechsynth

import (
	"context"
	"errors"
	"overgo/internal/media"
	"slices"
	"strings"
	"testing"
)

// speakOrderText is a short sentence and then one longer than a segment holds,
// so the plan cuts it at a sentence's end and again inside a sentence. Nothing
// in it repeats, so a recording that read the segments in another order, or
// left one out, could not hold the same samples.
const speakOrderText = "The tide went out before dawn. A grey heron stood at the edge of the harbour and watched the fishing boats leave one after another, each with a lamp swinging at its bow and a dog asleep on its nets."

// overLongReadings is how many times speakOrderText is repeated to make a text
// whose segments are budgeted past what one recording may hold.
const overLongReadings = 96

// TestSpeakReadsSegmentsInOrder holds that a text too long for one segment is
// read aloud as its segments in the text's order: the recording is, sample for
// sample, what the three stages of a served request give for each segment's
// own text under the reading's voice, seed and temperature and the plan's
// frame budget, one after another, and it ended on its own. It takes over
// what the retired document recording's test held, on the path that reads a
// whole text aloud.
//
// A stop raised while the second segment is generated ends the reading with
// the stop's own cause and no samples, and a text budgeted past what one
// recording may hold is refused before any noise is drawn.
func TestSpeakReadsSegmentsInOrder(t *testing.T) {
	synth := loadArtifactSynthesizer(t)
	request := DocumentRequest{Text: speakOrderText, Voice: "alba", Seed: 7, Temperature: speakOrderTemperature}
	plan, err := synth.PlanDocument(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Segments) < 2 {
		t.Fatalf("the text is planned as %d segments, and an order needs two", len(plan.Segments))
	}
	seen := map[string]bool{}
	var want []float32
	for _, segment := range plan.Segments {
		text := request.Text[segment.Start:segment.End]
		if seen[text] {
			t.Fatalf("two segments hold the text %q, so their order could not be told", text)
		}
		seen[text] = true
		generation, err := synth.tokenize(SynthesisRequest{Text: text, Voice: request.Voice, Seed: request.Seed, Temperature: request.Temperature, MaxFrames: segment.MaxFrames})
		if err != nil {
			t.Fatalf("%q: %v", text, err)
		}
		latents, err := synth.generate(t.Context(), generation)
		if err != nil {
			t.Fatalf("%q: %v", text, err)
		}
		audio, err := synth.decode(t.Context(), latents)
		if err != nil {
			t.Fatalf("%q: %v", text, err)
		}
		if !audio.Complete || len(audio.PCM) == 0 {
			t.Fatalf("%q read alone gave %d samples (ended on its own: %v)", text, len(audio.PCM), audio.Complete)
		}
		t.Logf("%d tokens, at most %d frames, %d samples: %q", segment.Tokens, segment.MaxFrames, len(audio.PCM), text)
		want = append(want, audio.PCM...)
	}
	var emitted []float32
	whole, err := synth.Speak(media.WithAudioOutput(t.Context(), func(_ context.Context, chunk media.DecodedAudio) error {
		if chunk.Format.Channels != 1 || chunk.Format.SampleRate != uint64(synth.model.Codec.SampleRate) {
			t.Fatalf("invalid provisional format: %+v", chunk.Format)
		}
		emitted = append(emitted, chunk.Samples...)
		return nil
	}), request)
	if err != nil {
		t.Fatal(err)
	}
	if !whole.Complete || !slices.Equal(whole.PCM, want) {
		same := 0
		for same < min(len(whole.PCM), len(want)) && whole.PCM[same] == want[same] {
			same++
		}
		t.Fatalf("the whole text is read as %d samples (ended on its own: %v), and its %d segments read one after another give %d; the first %d are the same", len(whole.PCM), whole.Complete, len(plan.Segments), len(want), same)
	}
	if !slices.Equal(emitted, whole.PCM) {
		t.Fatal("provisional output differs from completed recording")
	}
	consumerFailure := errors.New("speech output consumer closed")
	chunks := 0
	consumerStopped, err := synth.Speak(media.WithAudioOutput(t.Context(), func(context.Context, media.DecodedAudio) error { chunks++; return consumerFailure }), request)
	if !errors.Is(err, consumerFailure) || chunks != 1 || len(consumerStopped.PCM) != 0 {
		t.Fatalf("consumer failure: chunks=%d samples=%d err=%v", chunks, len(consumerStopped.PCM), err)
	}

	ctx, cancel := context.WithCancelCause(t.Context())
	cause := errors.New("stop this reading")
	defer cancel(cause)
	begun := 0
	stopped, err := synth.speakFrom(ctx, request, func(generation generationPlan) frameNoise {
		begun++
		if begun == 1 {
			return seededNoise(generation)
		}
		return func(int, []float32) { cancel(cause) }
	})
	if !errors.Is(err, cause) || len(stopped.PCM) != 0 || begun != 2 {
		t.Fatalf("a reading stopped in its second segment began %d segments and returned %d samples: %v", begun, len(stopped.PCM), err)
	}

	// The reading above was planned, so its length alone is what is refused
	// here. A reading that got past the plan would ask for its noise, and is
	// given none, so that it fails at its first frame and does not run for the
	// length of the text.
	long := request
	long.Text = strings.Repeat(request.Text+" ", overLongReadings)
	drawn := false
	refused, err := synth.speakFrom(t.Context(), long, func(generationPlan) frameNoise {
		drawn = true
		return nil
	})
	if err == nil || drawn || len(refused.PCM) != 0 {
		t.Fatalf("a text of %d readings was not refused before its first segment (noise asked for: %v, %d samples): %v", overLongReadings, drawn, len(refused.PCM), err)
	}
}
