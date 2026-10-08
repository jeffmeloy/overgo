package scorecard

import (
	"math"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testutil"
)

// TestSpeechReadingsAreEachModelsOwn scores two recognizers on one set of
// utterances so that their word error rates come out the same to every digit,
// as granite-speech's and nemotron's did on the store: each gets one word
// wrong of the same number, in a different utterance. Equal readings are not
// one record standing for two models. Each recipe's reading is read from the
// report whose plan names that recipe, over that recipe's own runs and
// transcripts; the two reports differ, and so do the transcripts they scored.
// Each states the utterances it rests on and the noise of a rate pooled over
// them, which is what makes an equal reading unremarkable: one word in six
// with an error of a seventh.
func TestSpeechReadingsAreEachModelsOwn(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	set, split := id(artifact.KindDataset, "utterances"), id(artifact.KindDatasetShard, "heldout")
	platform, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "windows", Arch: "amd64", Device: "cpu", Backend: "host", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	commit := strings.Repeat("ab", 20)
	cases := []evaluation.TranscriptionCase{
		{Name: "a", Group: "heldout", Reference: "the cat sat"},
		{Name: "b", Group: "heldout", Reference: "on the mat"},
	}
	authorities := []artifact.Descriptor{{ID: set}, {ID: split}}
	for index := range cases {
		cases[index].Source = recipecontract.AudioReference{Audio: id(artifact.KindFile, cases[index].Name), Profile: id(artifact.KindProfile, cases[index].Name)}
		cases[index].SampleCount, cases[index].SampleRate = 16000, 16000
		authorities = append(authorities, artifact.Descriptor{ID: cases[index].Source.Audio}, artifact.Descriptor{ID: cases[index].Source.Profile})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "speech-pair/authorities", Artifacts: authorities}); err != nil {
		t.Fatal(err)
	}
	environment, err := platform.Batch("speech-pair/environment")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, environment); err != nil {
		t.Fatal(err)
	}
	compiled, err := evaluation.CompileTranscription(evaluation.TranscriptionSuite{
		Kind: evaluation.TranscriptionKind, Schema: "fixture/v1", Source: "pinned fixture", Dataset: set, Split: split,
		Normalization: []evaluation.TranscriptionNormalization{evaluation.TranscriptionLowercase}, Cases: cases,
	})
	if err != nil {
		t.Fatal(err)
	}
	// recognize runs one recognizer over the utterances, hearing texts, and
	// returns its recipe, its report and the transcripts it recorded.
	recognize := func(name string, texts ...string) (artifact.ID, evaluation.TranscriptionReport, []artifact.ID) {
		t.Helper()
		recognizer, definition := id(artifact.KindRecipe, name), id(artifact.KindModelDefinition, name)
		if _, err := store.Commit(ctx, artifact.Batch{Key: "speech-pair/" + name, Artifacts: []artifact.Descriptor{{ID: recognizer}, {ID: definition}}}); err != nil {
			t.Fatal(err)
		}
		plan, err := evaluation.BindTranscription(compiled, evaluation.ExactAuthorities{
			ModelDefinition: definition, RuntimeRecipe: recognizer, CodeCommit: commit, Environment: platform.ID,
			Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
		})
		if err != nil {
			t.Fatal(err)
		}
		var predictions []evaluation.TranscriptionPrediction
		var transcripts []artifact.ID
		for index, text := range texts {
			source := cases[index].Source
			output, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, speechrecognition.TranscriptionSchema),
				recipecontract.Transcription{Source: source, Text: text, Language: "en"})
			if err != nil {
				t.Fatal(err)
			}
			run, err := runrecord.NewBoundRun(recognizer, runrecord.OutcomeSucceeded,
				[]artifact.ID{set, split, source.Audio, source.Profile}, []artifact.ID{output.Descriptor.ID}, "", commit, platform.ID, 1,
				[]runrecord.PhaseMetric{{Phase: runrecord.PhasePrefill, DurationNS: 1}})
			if err != nil {
				t.Fatal(err)
			}
			content, err := run.Content()
			if err != nil {
				t.Fatal(err)
			}
			batch, err := artifact.NewDocumentBatch("speech-pair/"+run.ID.DigestHex(), []artifact.Content{output, content}, run.Lineage(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Commit(ctx, batch); err != nil {
				t.Fatal(err)
			}
			predictions = append(predictions, evaluation.TranscriptionPrediction{Name: cases[index].Name, Run: run.ID})
			transcripts = append(transcripts, output.Descriptor.ID)
		}
		report, err := evaluation.EvaluateTranscription(ctx, store, compiled, plan, predictions)
		if err != nil {
			t.Fatal(err)
		}
		return recognizer, report, transcripts
	}
	// Each misses one word of six, the first in the first utterance and the
	// second in the second.
	first, firstReport, firstTranscripts := recognize("first recognizer", "the cat", "on the mat")
	second, secondReport, secondTranscripts := recognize("second recognizer", "the cat sat", "on mat")

	records, err := heldoutRecords(ctx, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	one, other := records[first].cell(recipe.TaskTranscription), records[second].cell(recipe.TaskTranscription)
	if !one.Measured() || !other.Measured() || one.Value != 1.0/6 || other.Value != one.Value {
		t.Fatalf("the two recognizers read %+v and %+v, want the same word error rate of one word in six", one, other)
	}
	if one.Source != firstReport.ID || other.Source != secondReport.ID || one.Source == other.Source {
		t.Fatalf("equal readings are read from %s and %s, want each recognizer's own report, %s and %s", one.Source, other.Source, firstReport.ID, secondReport.ID)
	}
	for index := range firstTranscripts {
		if firstTranscripts[index] == secondTranscripts[index] {
			t.Fatalf("utterance %d was scored from one transcript for both recognizers", index)
		}
	}
	// One utterance of three words holds the whole error: each utterance's
	// edits less half a word are a half and minus a half, over six words.
	if want := math.Sqrt(2*(0.25+0.25)) / 6; one.Cases != 2 || other.Cases != 2 || math.Abs(one.Noise-want) > 1e-15 || math.Abs(other.Noise-want) > 1e-15 {
		t.Fatalf("the readings state %d and %d cases with noise %g and %g, want two utterances and %g", one.Cases, other.Cases, one.Noise, other.Noise, want)
	}
	if text := one.QualityText(); !strings.HasPrefix(text, "0.1667 ±0.17 on 2 cases (wer, ") {
		t.Fatalf("the reading says %q", text)
	}
}
