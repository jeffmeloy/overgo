package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheReadmesServedModelsAreTheScorecards holds README's served-model table and training line to what writes them.
// The README holds each written part between its marker lines once. -readme writes the newest record's served rows in
// the declaration's order, by the declared words, each quality beside the answers it lost and why, and each model's
// most device memory; a training row is left to the training line, a row no declaration names, or a quality on a
// measure none names, is refused by its name; a figure typed into the table fails -check, naming the record. The
// training line names each model under the strongest tier of its training evidence, with the span its bounded runs
// took, and each held-out verdict against its base.
func TestTheReadmesServedModelsAreTheScorecards(t *testing.T) {
	t.Parallel()
	published := string(testcheck.Must(os.ReadFile(filepath.Join(testutil.RepoRoot(t), readmeFile)))(t))
	for _, part := range readmeParts {
		if strings.Count(published, part.begin) != 1 || strings.Count(published, part.end) != 1 {
			t.Errorf("README.md holds the lines %q and %q other than once each", part.begin, part.end)
		}
	}
	root, store := t.TempDir(), filepath.Join(t.TempDir(), "overgodb-store")
	records := scorecard.Directory(store)
	readme := filepath.Join(root, readmeFile)
	run := func(args ...string) error {
		return runArgs(t.Context(), append([]string{"-root", root, "-repo", store}, args...), io.Discard)
	}
	opening, closing := "### Served models\n\n", "\nThe full scorecard is go run ./cmd/validate -scorecard.\n"
	testcheck.NoError(t, os.WriteFile(readme, []byte(opening+readmeServedBegin+"\n"+readmeServedEnd+closing), readmeMode))
	source := testutil.ArtifactID(t, artifact.KindEvidence, "a reading")
	row := func(name string, task recipe.Task, quality, memory scorecard.Cell) scorecard.Row {
		quality.Source, memory.Source = source, source
		return scorecard.Row{Name: name, Task: task, Cells: map[metric.Name]scorecard.Cell{
			metric.FirstOutput: {Reason: "a fixture measures nothing"}, metric.Throughput: {Reason: "a fixture measures nothing"},
			metric.Failures: {Reason: "a fixture measures nothing"}, metric.Load: {Reason: "a fixture measures nothing"},
			metric.SetupSteps: {Reason: "a fixture measures nothing"}, metric.Quality: quality, metric.PeakMemory: memory,
		}}
	}
	constant := 0.1121
	choices := scorecard.Cell{Value: 0.4921, Noise: 0.014, Cases: 1209, Unread: 145, Measure: "store/mmlu-pro", Constant: &constant,
		Direction: runrecord.DirectionMaximize, UnreadCauses: evaluation.UnreadCauses{Budget: 140, Thinking: 5}}
	speech := scorecard.Cell{Value: 0.0191, Noise: 0.005, Cases: 32, Measure: "dataset:sha256:ad4177fe91ec6ce5ffd1299905164d50672d09d08d5d44fdcb4b36430900b1d7/heldout"}
	unread := scorecard.Cell{Reason: "not measured since the previous scorecard"}
	rows := []scorecard.Row{
		row("Un-0", recipe.TaskImageGen, scorecard.Cell{Reason: "takes no caption", NotApplicable: true}, scorecard.Cell{Value: 0}),
		row("gemma-4-E4B-it", recipe.TaskInference, choices, scorecard.Cell{Value: 13.9 * gibibytes}),
		row("gemma-4-E4B-it", recipe.TaskProjection, speech, scorecard.Cell{Reason: "a projector runs inside the model's request", NotApplicable: true}),
		row("Qwen3.5-9B", recipe.TaskInference, unread, scorecard.Cell{Value: 9.0 * gibibytes}),
		row("Qwen3.5-9B", recipe.TaskTraining, choices, unread),
	}
	publish := func(at int, rows ...scorecard.Row) {
		testcheck.Must(scorecard.Publish(records, time.Date(2026, time.November, 10, at, 0, 0, 0, time.UTC), scorecard.Record{Rows: rows}))(t)
	}
	publish(1, rows...)
	testcheck.NoError(t, run("-readme"))
	newest := testcheck.Must(scorecard.Records(records))(t)[0].Name
	table := "Read from the record `" + strings.TrimSuffix(newest, ".json") + "`.\n\n" +
		"| Model | Modality | Input → output | Held-out quality | Peak device memory |\n| --- | --- | --- | --- | --- |\n" +
		"| Qwen3.5-9B (Q8_0) | text | text → text | — | 9.0 GiB |\n" +
		"| gemma-4-E4B-it | text, vision-language, audio-language | text → text; audio or image → text | MMLU-Pro 0.492 ±0.014 on 1,209 cases, 145 named no option (140 ran out of the answer's tokens, 5 left a thought open); word error rate 0.0191 ±0.005 on 32 cases | 13.9 GiB |\n" +
		"| Un-0 | image generation, video generation | class → 8×8 picture or 12-frame clip (a 120 kB toy model) | no held-out measure applies | host |\n"
	want := opening + readmeServedBegin + "\n" + table + readmeServedEnd + closing
	if got := string(testcheck.Must(os.ReadFile(readme))(t)); got != want {
		t.Fatalf("the README after -readme =\n%s\nwant\n%s", got, want)
	}
	testcheck.NoError(t, run("-readme", "-check"))
	testcheck.NoError(t, os.WriteFile(readme, []byte(strings.Replace(want, "| — | 9.0 GiB |", "| MMLU-Pro 0.49 | 9.0 GiB |", 1)), readmeMode))
	if err := run("-readme", "-check"); err == nil || !strings.Contains(err.Error(), newest) {
		t.Fatalf("-readme -check on a typed quality = %v, want it refused naming %s", err, newest)
	}
	publish(2, append(rows, row("an-undeclared-model", recipe.TaskInference, choices, unread))...)
	if err := run("-readme"); err == nil || !strings.Contains(err.Error(), "an-undeclared-model") {
		t.Fatalf("-readme over a row no declaration names = %v, want it refused naming the row", err)
	}
	unnamed := choices
	unnamed.Measure = "an-unnamed/suite 0123"
	publish(3, row("MiniCPM5-1B", recipe.TaskInference, unnamed, unread))
	if err := run("-readme"); err == nil || !strings.Contains(err.Error(), "an-unnamed/suite") {
		t.Fatalf("-readme over a quality on a measure no declaration names = %v, want it refused naming the measure", err)
	}

	verified := func(name string, claims ...runrecord.CapabilityClaim) runrecord.ModelVerification {
		for index := range claims {
			claims[index].Commit = strings.Repeat("a", 40)
			claims[index].Dataset = testutil.ArtifactID(t, artifact.KindDataset, name+" data")
			claims[index].Evidence = []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, name+" run "+claims[index].Capability)}
		}
		return testcheck.Must(runrecord.NewModelVerificationCorrection(testutil.ArtifactID(t, artifact.KindModel, name), name, claims, nil))(t)
	}
	base := 0.5001
	verdict := scorecard.Cell{Value: 0.5167, Noise: 0.0078, Base: &base, Cases: 24,
		Measure: "profile:sha256:9fe42a327b0dd15877a4597d2591dfbe84a56b763ea644558a0fefab3b711526"}
	line := testcheck.Must(trainingWords([]runrecord.ModelVerification{
		verified("measured", runrecord.CapabilityClaim{Capability: "training", Tier: runrecord.TierRealArtifactSmoke, SpanSteps: 3},
			runrecord.CapabilityClaim{Capability: "full-pipeline-training", Tier: runrecord.TierCapabilityMeasured, SpanSteps: 400}),
		verified("short", runrecord.CapabilityClaim{Capability: "training", Tier: runrecord.TierRealArtifactSmoke, SpanSteps: 1}),
		verified("Bounded", runrecord.CapabilityClaim{Capability: "training", Tier: runrecord.TierRealArtifactSmoke, SpanSteps: 12}),
	}, scorecard.Record{Rows: []scorecard.Row{row("Qwen2.5-0.5B", recipe.TaskTraining, verdict, unread), row("Qwen2.5-0.5B", recipe.TaskInference, choices, unread)}}))(t)
	wantLine := "Training, by the strongest tier of each recipe's evidence in the [training report](docs/TRAINING_COMPATIBILITY.md): " +
		"`capability-measured`, a held-out measure of what training changed, for measured; " +
		"`real-artifact-smoke`, a bounded run on the real model, of 1 to 12 steps, which shows the route runs and not that training converges, for Bounded, short. " +
		"Held-out verdicts in the record: Qwen2.5-0.5B, token accuracy 0.517 ±0.0078 against its base's 0.5 on 24 cases.\n"
	if line != wantLine {
		t.Fatalf("the training line =\n%s\nwant\n%s", line, wantLine)
	}
}

// gibibytes is a gibibyte in the float a memory cell holds.
const gibibytes = 1 << 30
