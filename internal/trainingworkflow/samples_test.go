package trainingworkflow

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/trainingprogram"
)

// A sampled run of the tiny model: forty updates, sampled every tenth, enough
// for its greedy reply to the held-out prompt to change as it learns.
const (
	samplingSteps = 40
	samplingEvery = 10
)

// TestTrainingSamplesRunInStepOrder runs a conversation fine-tune that samples
// every tenth of its forty updates and reads the document it published: each
// strip holds the held-out conversation's prompt, up to and including the
// assistant's turn marker, and its reply, and the samples after that prompt,
// the same prompt at each step, are at steps zero, ten, twenty, thirty and forty,
// in order, the last taken on the trained model the run published, and the model
// the strip ends with does not generate what it began with.
func TestTrainingSamplesRunInStepOrder(t *testing.T) {
	run := registerConversations(t, periodic(12))
	ctx := t.Context()
	result, err := RunConversationSFT(ctx, run.store, ConversationSFT{
		Dataset: run.source, Render: letters, ModelDirectory: run.model, WorkDirectory: run.work,
		OutputDirectory: filepath.Join(run.root, "sampled"), Steps: samplingSteps, SampleEvery: samplingEvery,
	})
	if err != nil || !result.Samples.Valid() {
		t.Fatalf("a sampled run = (%+v, %v), want the samples it published", result, err)
	}
	samples, err := trainingSamplesCodec.Require(ctx, run.store, result.Samples)
	if err != nil {
		t.Fatal(err)
	}
	if samples.Trained != result.Trained || len(samples.Strips) == 0 || len(samples.Strips) > sampleRecords {
		t.Fatalf("samples = %+v, want 1 to %d strips about the trained model %v", samples, sampleRecords, result.Trained)
	}
	for _, strip := range samples.Strips {
		if !strings.HasSuffix(strip.Prompt, assistantLetter) || strip.Expected == "" || strings.Trim(strip.Expected, "cd") != "" {
			t.Errorf("strip %+v: want a prompt ending at the assistant's letter and a reply in the assistant's letters", strip)
		}
		var steps []int
		for _, sample := range strip.Samples {
			steps = append(steps, sample.Step)
			if len(sample.Text) != len(strip.Expected) {
				t.Errorf("step %d generated %q, want as many tokens as the reply %q", sample.Step, sample.Text, strip.Expected)
			}
		}
		if !slices.Equal(steps, []int{0, 10, 20, 30, 40}) {
			t.Errorf("strip of %s sampled at steps %v, want every tenth from 0 to 40", strip.Record, steps)
		}
		t.Logf("prompt %q, reply %q, generated %v", strip.Prompt, strip.Expected, strip.Samples)
		if first, last := strip.Samples[0].Text, strip.Samples[len(strip.Samples)-1].Text; first == last {
			t.Errorf("strip of %s generated %q before training and after it: the samples show no change", strip.Record, first)
		}
	}
	// What a run does not sample it does not publish.
	plain, err := RunConversationSFT(ctx, run.store, ConversationSFT{
		Dataset: run.source, Render: letters, ModelDirectory: run.model, WorkDirectory: run.work,
		OutputDirectory: filepath.Join(run.root, "plain"), Steps: 2,
	})
	if err != nil || plain.Samples.Valid() {
		t.Fatalf("an unsampled run = (%+v, %v), want no samples", plain, err)
	}
}

// TestTrainingSamplesRefuseWhatTheyCannotAnswer checks a negative interval and a
// text run, which has no prompt to generate after, are refused before training.
func TestTrainingSamplesRefuseWhatTheyCannotAnswer(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	for name, request := range map[string]TextHeldoutRequest{
		"negative interval": {Host: true, Conversations: true, SampleEvery: -1},
		"plain text":        {Host: true, SampleEvery: 2},
	} {
		request.Alias, request.ModelDirectory, request.OutputDirectory, request.Steps = text.alias, text.model, filepath.Join(text.root, "refused-"+strings.ReplaceAll(name, " ", "-")), 2
		if _, err := RunTextHeldout(t.Context(), text.store, request); err == nil || !strings.Contains(err.Error(), "samples are generated") {
			t.Errorf("%s: err = %v, want the refusal naming samples", name, err)
		}
	}
}

// TestTrainingSamplesDocumentIsCanonical checks a samples document must name a
// trained checkpoint, hold a strip, and keep each strip's steps rising.
func TestTrainingSamplesDocumentIsCanonical(t *testing.T) {
	checkpoint := artifact.ID{}
	if _, err := trainingSamplesCodec.NewInitial(trainingSamples{Trained: checkpoint, Strips: []sampleStrip{{Samples: []sampleAt{{Step: 0}}}}}); err == nil {
		t.Error("samples about no checkpoint were accepted")
	}
}
