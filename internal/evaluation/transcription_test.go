package evaluation

import (
	"bytes"
	"encoding/json"
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testutil"
)

const transcriptionTestCommit = "0123456789abcdef0123456789abcdef01234567"

type transcriptionEvaluationFixture struct {
	store       artifact.Repository
	compiled    TranscriptionPlan
	plan        Plan
	recipe      artifact.ID
	environment artifact.ID
	cases       map[string]TranscriptionCase
}

func TestTranscriptionEvaluationUsesStoredPredictionsAndFailureDenominators(t *testing.T) {
	fixture := newTranscriptionEvaluationFixture(t)
	defer fixture.store.Close()
	predictions := []TranscriptionPrediction{
		fixture.failedPrediction(t, "admission", speechrecognition.AudioAdmissionFailure),
		fixture.successfulPrediction(t, "substitution", "the bat sat"),
		fixture.failedPrediction(t, "silence", speechrecognition.AudioAdmissionFailure),
		fixture.successfulPrediction(t, "exact", "hello world"),
	}
	report, err := EvaluateTranscription(t.Context(), fixture.store, fixture.compiled, fixture.plan, predictions)
	if err != nil {
		t.Fatal(err)
	}
	if report.Overall.Utterances != 3 || report.Overall.AdmissionFailures != 1 ||
		report.Overall.InferenceFailures != 0 || report.Overall.SilentControls != 1 ||
		report.Overall.SilentControlFailures != 0 || report.Overall.SampleCount != 36_000 ||
		math.Abs(report.Overall.DurationSeconds-2.25) > 1e-12 {
		t.Fatalf("overall accounting = %+v", report.Overall)
	}
	if math.Abs(report.Overall.WordErrorRate-3.0/7.0) > 1e-12 ||
		math.Abs(report.Overall.CharacterErrorRate-11.0/32.0) > 1e-12 {
		t.Fatalf("quality = WER %.12f CER %.12f", report.Overall.WordErrorRate, report.Overall.CharacterErrorRate)
	}
	if len(report.Groups) != 2 || report.Groups[0].Name != "clean" || report.Groups[1].Name != "control" {
		t.Fatalf("groups = %+v", report.Groups)
	}
	exactIndex := slices.IndexFunc(report.Observations, func(value TranscriptionObservation) bool { return value.Name == "exact" })
	if exactIndex < 0 || report.Observations[exactIndex].CodeCommit != transcriptionTestCommit ||
		report.Observations[exactIndex].Environment != fixture.environment ||
		report.Observations[exactIndex].Output.Kind() != artifact.KindOutput {
		t.Fatalf("prediction lineage = %+v", report.Observations)
	}
	stored, found, err := artifact.ReadContent(t.Context(), fixture.store, report.ID)
	if err != nil || !found || stored.Descriptor.Schema != transcriptionReportSchema {
		t.Fatalf("stored report found=%v schema=%q err=%v", found, stored.Descriptor.Schema, err)
	}
}

func TestTranscriptionPredictionIsolationAndAuthorityRefusal(t *testing.T) {
	fixture := singleTranscriptionEvaluationFixture(t, TranscriptionCase{
		Name: "exact", Group: "clean", Source: audioReference(t, "isolated-exact"),
		Reference: "Hello,   world!", SampleCount: 16_000, SampleRate: 16_000,
	})
	defer fixture.store.Close()
	prediction := fixture.successfulPrediction(t, "exact", "hello world")
	encoded, err := json.Marshal(prediction)
	if err != nil {
		t.Fatal(err)
	}
	if slices.ContainsFunc([]string{"Hello,   world!", "hello world"}, func(reference string) bool {
		return bytes.Contains(encoded, []byte(reference))
	}) {
		t.Fatalf("target leaked into prediction handoff: %s", encoded)
	}
	foreign := fixture
	foreign.plan.body.CodeCommit = "fedcba9876543210fedcba9876543210fedcba98"
	foreign.plan.identity, err = artifact.JSONID(artifact.KindProfile, foreign.plan.body)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := EvaluateTranscription(t.Context(), fixture.store, fixture.compiled, foreign.plan,
		[]TranscriptionPrediction{prediction}); err == nil {
		t.Fatal("prediction from a foreign code authority was accepted")
	}
	reordered := fixture.compiled.suite
	reordered.Normalization = []TranscriptionNormalization{
		TranscriptionCollapseWhitespace, TranscriptionLowercase, TranscriptionStripPunctuation,
	}
	other, err := CompileTranscription(reordered)
	if err != nil {
		t.Fatal(err)
	}
	otherPlan, err := BindTranscription(other, ExactAuthorities{
		ModelDefinition: fixture.plan.body.ModelDefinition, RuntimeRecipe: fixture.recipe,
		CodeCommit: transcriptionTestCommit, Environment: fixture.environment,
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	if otherPlan.Identity() == fixture.plan.Identity() {
		t.Fatal("normalization order did not change scoring provenance")
	}
}

func TestTranscriptionSilentSuccessIsNegativeControlFailure(t *testing.T) {
	fixture := singleTranscriptionEvaluationFixture(t, TranscriptionCase{
		Name: "silence", Group: "control", Source: audioReference(t, "silent-success"),
		SampleCount: 4_000, SampleRate: 16_000, SilentControl: true,
	})
	defer fixture.store.Close()
	prediction := fixture.successfulPrediction(t, "silence", "hallucinated speech")
	report, err := EvaluateTranscription(t.Context(), fixture.store, fixture.compiled, fixture.plan,
		[]TranscriptionPrediction{prediction})
	if err != nil {
		t.Fatal(err)
	}
	if report.Overall.Utterances != 0 || report.Overall.SilentControls != 1 ||
		report.Overall.SilentControlFailures != 1 || !report.Observations[0].ControlViolation {
		t.Fatalf("silent control counted as success: %+v", report.Overall)
	}
}

func TestWordErrorRateUsesOneSequenceEditPrimitive(t *testing.T) {
	reference := stringsForTranscriptionScore("The, quick brown fox")
	hypothesis := stringsForTranscriptionScore("the quick fox")
	if distance := sequenceEditDistance(reference, hypothesis); distance != 1 {
		t.Fatalf("word edit distance = %d, want 1", distance)
	}
}

func TestCharacterErrorRateUsesUnicodeRunes(t *testing.T) {
	reference := []rune(normalizeTranscription("Café", []TranscriptionNormalization{TranscriptionLowercase}))
	hypothesis := []rune(normalizeTranscription("Cafe", []TranscriptionNormalization{TranscriptionLowercase}))
	if distance := sequenceEditDistance(reference, hypothesis); distance != 1 {
		t.Fatalf("character edit distance = %d, want 1", distance)
	}
}

func stringsForTranscriptionScore(value string) []string {
	return strings.Fields(normalizeTranscription(value, []TranscriptionNormalization{
		TranscriptionLowercase, TranscriptionStripPunctuation, TranscriptionCollapseWhitespace,
	}))
}

func newTranscriptionEvaluationFixture(t *testing.T) transcriptionEvaluationFixture {
	t.Helper()
	cases := []TranscriptionCase{
		{Name: "exact", Group: "clean", Source: audioReference(t, "exact"), Reference: "Hello,   world!", SampleCount: 16_000, SampleRate: 16_000},
		{Name: "substitution", Group: "clean", Source: audioReference(t, "substitution"), Reference: "The cat sat", SampleCount: 8_000, SampleRate: 16_000},
		{Name: "admission", Group: "clean", Source: audioReference(t, "admission"), Reference: "lost words", SampleCount: 8_000, SampleRate: 16_000},
		{Name: "silence", Group: "control", Source: audioReference(t, "silence"), SampleCount: 4_000, SampleRate: 16_000, SilentControl: true},
	}
	return transcriptionEvaluationFixtureFromCases(t, cases)
}

func singleTranscriptionEvaluationFixture(t *testing.T, testCase TranscriptionCase) transcriptionEvaluationFixture {
	t.Helper()
	return transcriptionEvaluationFixtureFromCases(t, []TranscriptionCase{testCase})
}

func transcriptionEvaluationFixtureFromCases(t *testing.T, cases []TranscriptionCase) transcriptionEvaluationFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	dataset := testutil.ArtifactID(t, artifact.KindDataset, "transcription dataset")
	split := testutil.ArtifactID(t, artifact.KindDatasetShard, "transcription split")
	recipe := testutil.ArtifactID(t, artifact.KindRecipe, "transcription recipe")
	modelDefinition := testutil.ArtifactID(t, artifact.KindModelDefinition, "transcription model definition")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "transcription environment")
	descriptors := []artifact.Descriptor{{ID: dataset}, {ID: split}, {ID: recipe}, {ID: modelDefinition}, {ID: environment}}
	caseMap := make(map[string]TranscriptionCase, len(cases))
	for _, testCase := range cases {
		descriptors = append(descriptors, artifact.Descriptor{ID: testCase.Source.Audio}, artifact.Descriptor{ID: testCase.Source.Profile})
		caseMap[testCase.Name] = testCase
	}
	if _, err = artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "evaluation/transcription/fixture-authorities", Artifacts: descriptors,
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	compiled, err := CompileTranscription(TranscriptionSuite{
		Kind: TranscriptionKind, Schema: "fixture/v1", Source: "pinned fixture",
		Dataset: dataset, Split: split,
		Normalization: []TranscriptionNormalization{
			TranscriptionLowercase, TranscriptionStripPunctuation, TranscriptionCollapseWhitespace,
		},
		Cases: cases,
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	plan, err := BindTranscription(compiled, ExactAuthorities{
		ModelDefinition: modelDefinition, RuntimeRecipe: recipe, CodeCommit: transcriptionTestCommit,
		Environment: environment, Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	return transcriptionEvaluationFixture{
		store: store, compiled: compiled, plan: plan, recipe: recipe, environment: environment, cases: caseMap,
	}
}

func audioReference(t *testing.T, name string) recipecontract.AudioReference {
	t.Helper()
	return recipecontract.AudioReference{
		Audio:   testutil.ArtifactID(t, artifact.KindFile, name+" audio"),
		Profile: testutil.ArtifactID(t, artifact.KindProfile, name+" audio profile"),
	}
}

func (fixture transcriptionEvaluationFixture) successfulPrediction(t *testing.T, name, text string) TranscriptionPrediction {
	t.Helper()
	testCase := fixture.cases[name]
	transcription := recipecontract.Transcription{Source: testCase.Source, Text: text, Language: "en"}
	output, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindOutput, speechrecognition.TranscriptionSchema), transcription,
	)
	if err != nil {
		t.Fatal(err)
	}
	run, err := runrecord.NewBoundRun(
		fixture.recipe, runrecord.OutcomeSucceeded,
		[]artifact.ID{fixture.compiled.dataset, fixture.compiled.split, testCase.Source.Audio, testCase.Source.Profile},
		[]artifact.ID{output.Descriptor.ID}, "",
		transcriptionTestCommit, fixture.environment, 100,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhasePrefill, DurationNS: 100}},
	)
	if err != nil {
		t.Fatal(err)
	}
	runContent, err := run.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"evaluation/transcription/prediction/"+name,
		[]artifact.Content{output, runContent}, run.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = artifact.CommitBatch(t.Context(), fixture.store, batch); err != nil {
		t.Fatal(err)
	}
	return TranscriptionPrediction{Name: name, Run: run.ID}
}

func (fixture transcriptionEvaluationFixture) failedPrediction(t *testing.T, name, failure string) TranscriptionPrediction {
	t.Helper()
	testCase := fixture.cases[name]
	run, err := runrecord.NewBoundRun(
		fixture.recipe, runrecord.OutcomeFailed,
		[]artifact.ID{fixture.compiled.dataset, fixture.compiled.split, testCase.Source.Audio, testCase.Source.Profile}, nil, failure,
		transcriptionTestCommit, fixture.environment, 100,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseMediaDecode, DurationNS: 100}},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := run.Batch("evaluation/transcription/prediction/" + name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = artifact.CommitBatch(t.Context(), fixture.store, batch); err != nil {
		t.Fatal(err)
	}
	return TranscriptionPrediction{Name: name, Run: run.ID}
}
