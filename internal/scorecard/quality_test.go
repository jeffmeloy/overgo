package scorecard

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/hostoptimizer"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/speechrecognition"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestQualityKeepsExecutionAlternatives keeps the evidence index's distinct
// contracts visible, with their actual producer rather than a publishing HEAD.
func TestQualityKeepsExecutionAlternatives(t *testing.T) {
	t.Parallel()
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "quality-environment")
	var summaries []evaluation.EvalSummary
	for _, name := range []string{"resident", "isolated", "other-environment"} {
		data, err := json.Marshal(struct {
			evaluation.EvalSummary
			Comparison  artifact.ID `json:"comparison"`
			Environment artifact.ID `json:"environment"`
			CodeCommit  string      `json:"code_commit"`
		}{evaluation.EvalSummary{Suite: "fixture", Prompting: "raw-completion", Record: testutil.ArtifactID(t, artifact.KindEvaluation, name),
			Metrics: map[string]float64{"accuracy": 0.5}, ExecutionKnown: true}, testutil.ArtifactID(t, artifact.KindProfile, name), environment, "0123456789abcdef0123456789abcdef01234567"})
		var summary evaluation.EvalSummary
		if err == nil {
			err = json.Unmarshal(data, &summary)
		}
		if err != nil {
			t.Fatal(err)
		}
		summary.Directions = map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize}
		summaries = append(summaries, summary)
	}
	cell, others := qualityCell(recipe.TaskInference, summaries, nil)
	if len(others) != len(summaries)-1 {
		t.Fatalf("headline = %+v, alternatives = %+v, want all contracts", cell, others)
	}
	seen := map[artifact.ID]bool{}
	for _, measured := range append(others, cell) {
		data, err := json.Marshal(measured)
		var provenance struct {
			CodeCommit  string      `json:"code_commit"`
			Environment artifact.ID `json:"environment"`
			Contract    struct {
				Comparison artifact.ID `json:"comparison"`
			} `json:"contract"`
		}
		if err == nil {
			err = json.Unmarshal(data, &provenance)
		}
		if err != nil || provenance.CodeCommit != "0123456789abcdef0123456789abcdef01234567" || provenance.Environment != environment ||
			!provenance.Contract.Comparison.Valid() || seen[measured.Source] || measured.Value != 0.5 || measured.Direction != runrecord.DirectionMaximize {
			t.Fatalf("cell = %+v, provenance = %+v: %v", measured, provenance, err)
		}
		seen[measured.Source] = true
	}
	slices.Reverse(summaries)
	reordered, alternatives := qualityCell(recipe.TaskInference, summaries, nil)
	if reordered.Source != cell.Source || !slices.EqualFunc(alternatives, others, func(left, right Cell) bool { return left.Source == right.Source }) {
		t.Fatal("input order changed the displayed headline or alternatives")
	}
}

// TestQualityReadsEachTasksHeldoutEvidence holds held-out quality to the
// evidence each task has. A projection recipe the evaluation circuit scored
// on sets outside the derived catalog reads one of them under a full
// contract: its metric and direction, the evaluation origin, the set's
// identity as suite and its unknown protocol when execution documents are
// absent, and the same set
// whichever ran last. A recipe's wall-time record carries no evaluation
// evidence and is not held-out quality. A task whose held-out evaluation is
// only proposed reads not applicable with the proposal, is proposed as work,
// and withholds no promotion; an inference recipe nothing evaluated is a
// gap that withholds it.
func TestQualityReadsEachTasksHeldoutEvidence(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	projection, timed := id(artifact.KindRecipe, "projection"), id(artifact.KindRecipe, "timed")
	first, second, timing := id(artifact.KindDataset, "caption-matching-a"), id(artifact.KindDataset, "caption-matching-b"), id(artifact.KindDataset, "smoke-timing")
	if second.String() < first.String() {
		first, second = second, first
	}
	// One run per set: a run has one metric record.
	run := func(set artifact.ID) artifact.ID { return id(artifact.KindRun, "run "+set.String()) }
	if _, err := store.Commit(ctx, artifact.Batch{Key: "quality/authorities", Artifacts: []artifact.Descriptor{
		{ID: projection}, {ID: timed}, {ID: first}, {ID: second}, {ID: timing}, {ID: run(first)}, {ID: run(second)}, {ID: run(timing)},
	}}); err != nil {
		t.Fatal(err)
	}
	accuracy := func(value float64) []runrecord.Metric {
		return []runrecord.Metric{{Name: "accuracy", Value: value, Direction: runrecord.DirectionMaximize}}
	}
	record := func(recipeID, set artifact.ID, metrics []runrecord.Metric) runrecord.Evaluation {
		t.Helper()
		evaluated, err := runrecord.NewEvaluation(recipeID, run(set), set, metrics)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := evaluated.Batch("quality/" + set.String())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return evaluated
	}
	// scored records metric evidence naming a plan over the set. This fixture
	// publishes only plan and run descriptors, so it cannot establish a prompt
	// protocol; the original metric, direction and source still remain visible.
	scored := func(recipeID, set artifact.ID, value float64) runrecord.Evaluation {
		t.Helper()
		evaluated := record(recipeID, set, accuracy(value))
		data, err := json.Marshal(evaluation.EvaluationEvidence{
			Version: artifact.InitialDocumentVersion,
			Plan:    id(artifact.KindProfile, "plan "+set.String()), Acceptance: id(artifact.KindProfile, "acceptance"),
			Evaluator: id(artifact.KindEvidence, "evaluator"), Report: id(artifact.KindEvaluation, "report "+set.String()),
			Run: run(set), Evaluation: evaluated.ID, ModelDefinition: id(artifact.KindModelDefinition, "definition"),
			Recipe: recipeID, Dataset: set, Split: id(artifact.KindDatasetShard, "split "+set.String()),
			Environment: id(artifact.KindEvidence, "environment"), CodeCommit: strings.Repeat("ab", 20),
			Phases:  []runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: 1}},
			Metrics: accuracy(value),
		})
		if err != nil {
			t.Fatal(err)
		}
		evidence, err := evaluation.ParseEvaluationEvidence(data)
		if err != nil {
			t.Fatal(err)
		}
		content, err := evidence.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch, err := artifact.NewDocumentBatch("quality/evidence/"+set.String(), []artifact.Content{content}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return evaluated
	}
	// The set named second runs last, so it is the newest.
	read := scored(projection, first, 0.5)
	scored(projection, second, 0.75)
	record(timed, timing, []runrecord.Metric{{Name: "smoke_wall_ns", Value: 1, Direction: runrecord.DirectionMinimize}})

	suites := map[artifact.ID]string{}
	if err := nameEvaluatedSets(ctx, store, suites, nil); err != nil {
		t.Fatal(err)
	}
	index := evaluation.LatestEvidence(ctx, store, suites, 0)
	cell, others := qualityCell(recipe.TaskProjection, index.EvaluationsByRecipe[projection], nil)
	want := Contract{Metric: "accuracy", Origin: originEvaluation, Suite: first.String(), Protocol: "unknown"}
	if !cell.Measured() || cell.Value != 0.5 || cell.Source != read.ID || cell.Direction != runrecord.DirectionMaximize || cell.Contract != want {
		t.Fatalf("projection quality = %+v, want the set named first, %+v, from its evaluation", cell, want)
	}
	if len(others) != 1 || others[0].Value != 0.75 || others[0].Contract.Suite != second.String() {
		t.Fatalf("other held-out qualities = %+v, want the second set's", others)
	}
	if cell, _ := qualityCell(recipe.TaskVQA, index.EvaluationsByRecipe[timed], nil); cell.Measured() || !cell.NotApplicable || !strings.Contains(cell.Reason, "proposed: ") {
		t.Fatalf("a wall-time record reads as held-out quality %+v, want the task's proposed evaluation", cell)
	}

	row := func(name string, task recipe.Task, quality Cell) Row {
		cells := map[metric.Name]Cell{}
		for _, column := range metric.Columns {
			cells[column] = inapplicable("not measured in this fixture")
		}
		cells[metric.Quality] = quality
		return Row{Model: id(artifact.KindModel, name), Name: name, Task: task, Cells: cells}
	}
	proposed, _ := qualityCell(recipe.TaskImageGen, nil, nil)
	published := Record{Commit: "b", Rows: []Row{row("projector", recipe.TaskProjection, cell), row("painter", recipe.TaskImageGen, proposed)}}
	reference := Record{Commit: "a"}
	gaps := published.Gaps(&reference)
	if len(gaps) != 1 || gaps[0].Key != "scorecard-unscored-image-gen" || !strings.Contains(gaps[0].Detail, "painter held-out-quality: "+proposed.Reason) {
		t.Fatalf("gaps = %+v, want one row to define image-gen's held-out evaluation", gaps)
	}
	unevaluated, _ := qualityCell(recipe.TaskInference, nil, nil)
	published.Rows = append(published.Rows, row("reader", recipe.TaskInference, unevaluated))
	gaps = published.Gaps(&reference)
	at := slices.IndexFunc(gaps, func(gap Gap) bool { return gap.Key == "scorecard-unmeasured-inference" })
	if unevaluated.NotApplicable || unevaluated.Measured() || at < 0 || !strings.Contains(gaps[at].Detail, "reader held-out-quality: ") {
		t.Fatalf("an inference recipe nothing evaluated reads %+v and proposes %+v, want a measurement to take", unevaluated, gaps)
	}
}

// TestQualityReadsTranscriptionReports holds a recipe's held-out quality to
// its newest transcription report, re-scored from the persisted runs: the
// word error rate of the report's largest group, to be minimized, under a
// contract naming the report's dataset and group and the prompting and
// normalization it was scored under. A transcription recipe no report
// scored reads its proposal, not applicable.
func TestQualityReadsTranscriptionReports(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	set, split, transcriber := id(artifact.KindDataset, "librispeech"), id(artifact.KindDatasetShard, "test-clean"), id(artifact.KindRecipe, "transcriber")
	platform, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "windows", Arch: "amd64", Device: "cpu", Backend: "host", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	definition, environment, commit := id(artifact.KindModelDefinition, "transcriber"), platform.ID, strings.Repeat("ab", 20)
	// The clean group holds two utterances, the other one.
	cases := []evaluation.TranscriptionCase{
		{Name: "a", Group: "clean", Reference: "the cat sat"},
		{Name: "b", Group: "clean", Reference: "on the mat"},
		{Name: "c", Group: "other", Reference: "hello world"},
	}
	authorities := []artifact.Descriptor{{ID: set}, {ID: split}, {ID: transcriber}, {ID: definition}, {ID: environment}}
	for index := range cases {
		cases[index].Source = recipecontract.AudioReference{Audio: id(artifact.KindFile, cases[index].Name), Profile: id(artifact.KindProfile, cases[index].Name)}
		cases[index].SampleCount, cases[index].SampleRate = 16000, 16000
		authorities = append(authorities, artifact.Descriptor{ID: cases[index].Source.Audio}, artifact.Descriptor{ID: cases[index].Source.Profile})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "quality/transcription/authorities", Artifacts: authorities}); err != nil {
		t.Fatal(err)
	}
	normalization := []evaluation.TranscriptionNormalization{
		evaluation.TranscriptionLowercase, evaluation.TranscriptionStripPunctuation, evaluation.TranscriptionCollapseWhitespace,
	}
	compiled, err := evaluation.CompileTranscription(evaluation.TranscriptionSuite{
		Kind: evaluation.TranscriptionKind, Schema: "fixture/v1", Source: "pinned fixture", Dataset: set, Split: split,
		Normalization: normalization, Cases: cases,
	})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := evaluation.BindTranscription(compiled, evaluation.ExactAuthorities{
		ModelDefinition: definition, RuntimeRecipe: transcriber, CodeCommit: commit, Environment: environment,
		Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	// score records one run per case that heard its text, then the report
	// scoring them.
	score := func(texts ...string) evaluation.TranscriptionReport {
		t.Helper()
		var predictions []evaluation.TranscriptionPrediction
		for index, text := range texts {
			source := cases[index].Source
			output, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, speechrecognition.TranscriptionSchema),
				recipecontract.Transcription{Source: source, Text: text, Language: "en"})
			if err != nil {
				t.Fatal(err)
			}
			run, err := runrecord.NewBoundRun(transcriber, runrecord.OutcomeSucceeded,
				[]artifact.ID{set, split, source.Audio, source.Profile}, []artifact.ID{output.Descriptor.ID}, "", commit, environment, 1,
				[]runrecord.PhaseMetric{{Phase: runrecord.PhasePrefill, DurationNS: 1}})
			if err != nil {
				t.Fatal(err)
			}
			content, err := run.Content()
			if err != nil {
				t.Fatal(err)
			}
			batch, err := artifact.NewDocumentBatch("quality/transcription/"+run.ID.DigestHex(), []artifact.Content{output, content}, run.Lineage(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Commit(ctx, batch); err != nil {
				t.Fatal(err)
			}
			predictions = append(predictions, evaluation.TranscriptionPrediction{Name: cases[index].Name, Run: run.ID})
		}
		report, err := evaluation.EvaluateTranscription(ctx, store, compiled, plan, predictions)
		if err != nil {
			t.Fatal(err)
		}
		return report
	}
	score("the cat", "on mat", "hello")
	newest := score("the cat sat", "on a mat", "hello world")

	records, err := heldoutRecords(ctx, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	cell := records[transcriber].cell(recipe.TaskTranscription)
	if verdict, _ := holds(metric.Quality, cell, cell); verdict != Unknown || cell.Contract.Comparison.Valid() {
		t.Fatal("an absent environment document earned a transcription comparison")
	}
	batch, err := platform.Batch("quality/transcription/environment")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	records, err = heldoutRecords(ctx, store, nil)
	if err != nil {
		t.Fatal(err)
	}
	cell = records[transcriber].cell(recipe.TaskTranscription)
	if verdict, why := holds(metric.Quality, cell, cell); verdict != holding || !cell.Contract.Comparison.Valid() || cell.CodeCommit != commit || cell.Environment != environment {
		t.Fatalf("fully checked transcription cannot compare: %+v (%s)", cell, why)
	}
	protocol := []string{evaluation.PromptingRawCompletion.Label()}
	for _, operation := range normalization {
		protocol = append(protocol, string(operation))
	}
	want := Contract{Metric: "wer", Origin: originEvaluation, Suite: set.String() + "/clean", Protocol: strings.Join(protocol, ", "), Comparison: cell.Contract.Comparison}
	if !cell.Measured() || cell.Value != 1.0/6 || cell.Source != newest.ID || cell.Direction != runrecord.DirectionMinimize || cell.Contract != want {
		t.Fatalf("transcription quality = %+v, want the newest report's clean-group word error rate 1/6 under %+v", cell, want)
	}
	if cell, _ := qualityCell(recipe.TaskTranscription, nil, nil); cell.Measured() || !cell.NotApplicable || !strings.Contains(cell.Reason, "transcription report") {
		t.Fatalf("a transcription recipe no report scored reads %+v, want its proposal", cell)
	}
}

// TestQualityReadsHeldoutVerdicts holds the rows a training session's
// held-out verdict measures to it: the session's training recipe reads the
// trained model's value beside the base model's, and a tabular recipe whose
// model directory holds the weights file the session trained reads the base
// model's alone, each in the
// verdict's metric and direction under a contract naming the objective's
// held-out view and the verdict's target plan. An inference recipe of the
// same model reads no verdict.
func TestQualityReadsHeldoutVerdicts(t *testing.T) {
	for _, test := range []struct {
		name                                            string
		complete, losing, missingCheckpoint, wrongModel bool
		// unreadableObservation adds a newer verdict whose observation cannot be read.
		unreadableObservation bool
	}{
		{name: "legacy-weight-file"},
		{name: "complete-model", complete: true},
		{name: "valid-loss", complete: true, losing: true},
		{name: "missing-checkpoint", complete: true, missingCheckpoint: true},
		{name: "wrong-observed-model", complete: true, wrongModel: true},
		{name: "newer-verdict-with-an-unreadable-observation", complete: true, unreadableObservation: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			testQualityReadsHeldoutVerdict(t, test.complete, test.losing, test.missingCheckpoint, test.wrongModel, test.unreadableObservation)
		})
	}
}

func testQualityReadsHeldoutVerdict(t *testing.T, complete, losing, missingCheckpoint, wrongModel, unreadableObservation bool) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	// The session trained a head's weights file, which the served model
	// directory names as a weights component of the same bytes.
	weights := []byte("classification head weights")
	trainedWeights, component := testutil.ArtifactBytesID(t, artifact.KindModel, weights), testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	components := []artifact.Component{{Role: artifact.ComponentWeights, Name: "classification/weights", Artifact: component}}
	if complete {
		components[0].Role = artifact.ComponentWeightsShard
		components = append(components, artifact.Component{Role: artifact.ComponentWeightsShard, Ordinal: 1, Name: "classification/frozen", Artifact: id(artifact.KindTensorSet, "frozen")})
	}
	directory, err := artifact.NewManifest(artifact.KindModel, components)
	if err != nil {
		t.Fatal(err)
	}
	if complete {
		trainedWeights = directory.ID
	}
	training, tabular, inference := id(artifact.KindRecipe, "training"), id(artifact.KindRecipe, "tabular"), id(artifact.KindRecipe, "inference")
	environment := id(artifact.KindEvidence, "environment")
	authorities := []artifact.Descriptor{{ID: component}, {ID: training}, {ID: environment}}
	if complete {
		authorities = append(authorities, artifact.Descriptor{ID: components[1].Artifact})
	} else {
		authorities = append(authorities, artifact.Descriptor{ID: trainedWeights})
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "quality/verdict/authorities",
		Artifacts: authorities,
		Manifests: []artifact.Manifest{directory},
	}); err != nil {
		t.Fatal(err)
	}
	observedModel := trainedWeights
	if wrongModel {
		observedModel = id(artifact.KindModel, "another observed model")
		if _, err := store.Commit(ctx, artifact.Batch{Key: "quality/verdict/other-model", Artifacts: []artifact.Descriptor{{ID: observedModel}}}); err != nil {
			t.Fatal(err)
		}
	}
	session, err := runrecord.NewServingObservation(runrecord.ServingObservation{
		Model: observedModel, Recipe: training, Environment: environment, Task: recipe.TaskTraining,
		Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 1, MeasuredNS: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	observed, err := session.Batch("quality/verdict/session")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, observed); err != nil {
		t.Fatal(err)
	}

	source := id(artifact.KindDataset, "credit-g")
	var rows []dataset.Record
	for index := range 12 {
		rows = append(rows, dataset.Record{ID: fmt.Sprintf("table/%02d", index), Group: fmt.Sprintf("table-%02d", index)})
	}
	split, err := dataset.BuildGroupSplit(source, rows, 17, []dataset.SplitPartition{{Name: "heldout", Weight: 1}, {Name: "train", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	memberships := map[string]dataset.Membership{}
	for _, membership := range split.Memberships {
		memberships[membership.Partition] = membership
	}
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "credit-g", Kind: trainingprogram.ObjectiveTablePrediction,
		Signature: recipecontract.ModalitySignature{
			Inputs: []recipecontract.Modality{recipecontract.ModalityTable}, Outputs: []recipecontract.Modality{recipecontract.ModalityTable},
		},
		Dataset: source, Split: memberships["train"].ID, Processors: []artifact.ID{id(artifact.KindProfile, "processor")},
		Loss: id(artifact.KindProfile, "loss"), Evaluation: id(artifact.KindProfile, "evaluation"),
		Metric: trainingprogram.MetricTableAccuracy, Evidence: []artifact.ID{session.ID},
		Authority: trainingprogram.ObjectiveAdaptive,
	})
	if err != nil {
		t.Fatal(err)
	}
	view, err := evaluation.CompileSFTEvaluationView(objective, memberships["train"], memberships["heldout"])
	if err != nil {
		t.Fatal(err)
	}
	var cases []evaluation.HeldoutCase
	for _, record := range view.Records {
		cases = append(cases, evaluation.HeldoutCase{Record: record.ID, Label: "good"})
	}
	plan, err := evaluation.CompileHeldoutPlan(objective, view, cases)
	if err != nil {
		t.Fatal(err)
	}
	// answer scores a model that labels every held-out row label.
	answer := func(label string) evaluation.HeldoutReport {
		t.Helper()
		var observations []evaluation.HeldoutObservation
		for _, target := range cases {
			observations = append(observations, evaluation.HeldoutObservation{Record: target.Record, Label: label})
		}
		report, err := plan.Score(observations)
		if err != nil {
			t.Fatal(err)
		}
		return report
	}
	base, trained := answer("bad"), answer("good")
	baseValue, trainedValue := float64(0), float64(1)
	if losing {
		base, trained = trained, base
		baseValue, trainedValue = trainedValue, baseValue
	}
	registration, err := split.PublicationBatch("quality/verdict/registration", nil)
	if err != nil {
		t.Fatal(err)
	}
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	registration.Contents = append(registration.Contents, content)
	for _, declared := range []artifact.ID{source, objective.Processors[0], objective.Loss, objective.Evaluation} {
		registration.Artifacts = append(registration.Artifacts, artifact.Descriptor{ID: declared})
	}
	if _, err := artifact.CommitBatch(ctx, store, registration); err != nil {
		t.Fatal(err)
	}
	checkpoint, checkpointBatch := trainedCheckpoint(t, trainedWeights, source, view.TrainingMembership)
	if missingCheckpoint {
		checkpointBatch = artifact.Batch{Key: checkpointBatch.Key, Artifacts: []artifact.Descriptor{{ID: checkpoint.ID()}}}
	}
	if _, err := artifact.Publish(ctx, store, checkpointBatch); err != nil {
		t.Fatal(err)
	}
	verdict, err := evaluation.JudgeHeldout(objective, view, plan, base, trained, evaluation.HeldoutTraining{
		Observation: session.ID, Model: trainedWeights, Inputs: id(artifact.KindProfile, "inputs"), Trained: checkpoint,
	})
	if err != nil {
		t.Fatal(err)
	}
	judged, err := evaluation.HeldoutBatch(view, plan, base, trained, verdict)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(ctx, store, judged); err != nil {
		t.Fatal(err)
	}
	// A newer verdict of another observation, which the store holds no content of: it can be neither
	// reconstructed nor attributed to a recipe, and the older verdict above must not stand in for it.
	var newer artifact.ID
	if unreadableObservation {
		lost, err := runrecord.NewServingObservation(runrecord.ServingObservation{
			Model: observedModel, Recipe: training, Environment: environment, Task: recipe.TaskTraining,
			Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 2, MeasuredNS: 1,
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: "quality/verdict/lost-session", Artifacts: []artifact.Descriptor{{ID: lost.ID}}}); err != nil {
			t.Fatal(err)
		}
		later, err := evaluation.JudgeHeldout(objective, view, plan, base, trained, evaluation.HeldoutTraining{
			Observation: lost.ID, Model: trainedWeights, Inputs: id(artifact.KindProfile, "inputs"), Trained: checkpoint,
		})
		if err != nil {
			t.Fatal(err)
		}
		again, err := evaluation.HeldoutBatch(view, plan, base, trained, later)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.Publish(ctx, store, again); err != nil {
			t.Fatal(err)
		}
		newer = later.ID
	}

	records, err := heldoutRecords(ctx, store, []discovery.CatalogEntry{
		{Model: trainedWeights, Capabilities: []discovery.Capability{{Task: recipe.TaskTraining, Recipe: training}}},
		{Model: directory.ID, Capabilities: []discovery.Capability{{Task: recipe.TaskTabular, Recipe: tabular}, {Task: recipe.TaskInference, Recipe: inference}}},
	})
	if unreadableObservation {
		if err != nil {
			t.Fatalf("a verdict whose observation cannot be read stopped the read: %v", err)
		}
		for read, task := range map[artifact.ID]recipe.Task{training: recipe.TaskTraining, tabular: recipe.TaskTabular} {
			cell := records[read].cell(task)
			if cell.Measured() || !strings.Contains(cell.Reason, newer.String()) || !strings.Contains(cell.Reason, "older verdict is not known to be this recipe's newest") {
				t.Fatalf("%s's quality = %+v, want a gap naming the newer verdict %s, not the older verdict's value in its place", task, cell, newer)
			}
		}
		if _, found := records[inference]; found {
			t.Fatal("an inference recipe, which no verdict judges, read a held-out record")
		}
		return
	}
	if missingCheckpoint || wrongModel {
		// A verdict that cannot be reconstructed is no quality, and no abort: its
		// recipe's held-out quality is a gap that says why.
		if err != nil {
			t.Fatalf("a verdict that cannot be reconstructed stopped the read: %v", err)
		}
		for recipeID, record := range records {
			if len(record.summary.Metrics) != 0 {
				t.Fatalf("invalid held-out evidence was reported as quality for %s: %+v", recipeID, record)
			}
		}
		held := records[training]
		if cell := held.cell(recipe.TaskTraining); cell.Reason == "" || !strings.Contains(cell.Reason, "cannot be reconstructed") || !strings.Contains(cell.Reason, verdict.ID.String()) {
			t.Fatalf("the training recipe's quality = %+v, want a gap naming the verdict %s that cannot be reconstructed", cell, verdict.ID)
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	want := Contract{Metric: string(trainingprogram.MetricTableAccuracy), Origin: originEvaluation, Suite: view.ID.String(), Protocol: plan.ID().String()}
	var comparison artifact.ID
	for _, read := range []struct {
		task   recipe.Task
		recipe artifact.ID
		value  float64
		based  bool
	}{{recipe.TaskTraining, training, trainedValue, true}, {recipe.TaskTabular, tabular, baseValue, false}} {
		cell := records[read.recipe].cell(read.task)
		if verdict, why := holds(metric.Quality, cell, cell); verdict != holding || !cell.Contract.Comparison.Valid() || comparison.Valid() && cell.Contract.Comparison != comparison || cell.CodeCommit != "" {
			t.Fatalf("fully re-scored held-out contract cannot compare or invented a producer: %+v (%s)", cell, why)
		}
		comparison, want.Comparison = cell.Contract.Comparison, cell.Contract.Comparison
		if !cell.Measured() || cell.Value != read.value || cell.Source != verdict.ID || cell.Direction != runrecord.DirectionMaximize || cell.Contract != want {
			t.Fatalf("%s quality = %+v, want %g from the verdict under %+v", read.task, cell, read.value, want)
		}
		if (cell.Base != nil) != read.based || read.based && *cell.Base != baseValue {
			t.Fatalf("%s quality carries base %v, want the base model's %g beside it: %t", read.task, cell.Base, baseValue, read.based)
		}
	}
	if record, read := records[inference]; read {
		t.Fatalf("an inference recipe reads the verdict %+v, want its own suites", record)
	}
}

// trainedCheckpoint publishes, into a scratch directory, a one-step
// checkpoint that trained model on split of source: the trained output a
// held-out verdict names.
func trainedCheckpoint(t *testing.T, model, source, split artifact.ID) (trainingprogram.Checkpoint, artifact.Batch) {
	t.Helper()
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	const parameters = 4
	plan, err := hostoptimizer.CompilePlan(parameters, []hostoptimizer.GroupSpec{{Name: "weight", End: parameters, Rows: 2, Cols: 2}})
	if err != nil {
		t.Fatal(err)
	}
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(parameters)
	if err != nil {
		t.Fatal(err)
	}
	weights := make([]float32, parameters)
	muon, err := hostoptimizer.New(weights, slices.Repeat([]float32{1}, parameters), plan, config)
	if err != nil {
		t.Fatal(err)
	}
	muon.Step()
	state := muon.Snapshot()
	rng := id(artifact.KindProfile, "rng")
	directory := filepath.Join(t.TempDir(), "checkpoint")
	checkpoint, err := trainingprogram.PublishCheckpoint(directory, trainingprogram.CheckpointSpec{
		RunPlan: id(artifact.KindRecipe, "run plan"), Program: id(artifact.KindRecipe, "program"), Model: model,
		Dataset: source, Split: split, Stream: trainingprogram.DatasetState{Identity: id(artifact.KindProfile, "stream"), Position: 1},
		Optimizer: state, ParameterCount: state.Parameters(),
		RNG:        []trainingprogram.RNGState{{Name: "augmentation", Algorithm: rng}, {Name: "data", Algorithm: rng}},
		Processors: []artifact.ID{id(artifact.KindProfile, "processor")},
		Lineage:    []trainingprogram.LineageParent{{Artifact: model, Relation: artifact.RelationTrainedFrom}},
	}, func(stage string) error {
		return optimizer.SavePlanWeights(filepath.Join(stage, trainingprogram.CheckpointWeights), plan, weights)
	})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := checkpoint.Batch("quality/verdict/checkpoint", directory)
	if err != nil {
		t.Fatal(err)
	}
	return checkpoint, batch
}

// TestCarriedQualityNamesItsSource holds a carried held-out quality to the
// run that scored it: a result carried forward reads as carried and names the
// commit of the run its chain began at, not the commit of the run that
// replayed a sample of it, while a quality a run scored itself keeps its own.
func TestCarriedQualityNamesItsSource(t *testing.T) {
	t.Parallel()
	scored, carrying := strings.Repeat("a", 40), strings.Repeat("b", 40)
	carried := evaluation.EvalSummary{
		Suite: "fixture", Prompting: "raw-completion", Record: testutil.ArtifactID(t, artifact.KindEvaluation, "carrier"),
		Metrics: map[string]float64{"accuracy": 0.5}, Directions: map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize},
		CodeCommit: carrying, ScoredCommit: scored, Carried: testutil.ArtifactID(t, artifact.KindEvaluation, "receipt"), ExecutionKnown: true,
	}
	cell, _ := qualityCell(recipe.TaskInference, []evaluation.EvalSummary{carried}, nil)
	if cell.CodeCommit != scored || !strings.HasSuffix(cell.Label, ", carried") {
		t.Fatalf("a carried quality reads %q at %q, want carried at the scoring run's %q", cell.Label, cell.CodeCommit, scored)
	}
	own := carried
	own.Carried, own.ScoredCommit = artifact.ID{}, ""
	cell, _ = qualityCell(recipe.TaskInference, []evaluation.EvalSummary{own}, nil)
	if cell.CodeCommit != carrying || strings.Contains(cell.Label, "carried") {
		t.Fatalf("a quality its run scored reads %q at %q, want its own commit %q and no carried scope", cell.Label, cell.CodeCommit, carrying)
	}
}
