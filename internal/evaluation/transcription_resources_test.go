package evaluation

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testutil"
)

type transcriptionResourceFixture struct {
	store    artifact.Repository
	compiled TranscriptionPlan
	plan     Plan
	executor *recordingTranscriptionExecutor
	inputs   []TranscriptionResourceInput
}

type recordingTranscriptionExecutor struct {
	repository artifact.Repository
	model      artifact.ID
	recipe     artifact.ID
	cases      map[artifact.ID]TranscriptionCase
	texts      map[string]string
	failures   map[string]string
	calls      map[string]uint32
	totalCalls uint64
	change     bool
}

func (executor *recordingTranscriptionExecutor) Transcribe(
	ctx context.Context,
	data []byte,
	origin dataset.AudioPayloadOrigin,
	_ dataset.AudioInspectionPolicy,
	_ *speechrecognition.TranscriptionWorkspace,
	binding speechrecognition.RunBinding,
) (recipecontract.Transcription, runrecord.Run, error) {
	if err := ctx.Err(); err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	testCase, found := executor.cases[origin.Container]
	if !found {
		return recipecontract.Transcription{}, runrecord.Run{}, errors.New("fixture: unknown audio source")
	}
	source, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil || source != testCase.Source.Audio {
		return recipecontract.Transcription{}, runrecord.Run{}, errors.New("fixture: audio identity differs")
	}
	executor.calls[testCase.Name]++
	executor.totalCalls++
	measured := 100 + executor.totalCalls
	inputs := []artifact.ID{executor.model, binding.Dataset, binding.Split, testCase.Source.Audio, testCase.Source.Profile}
	if failure := executor.failures[testCase.Name]; failure != "" {
		run, runErr := runrecord.NewBoundRun(
			executor.recipe, runrecord.OutcomeFailed, inputs, nil, failure,
			binding.CodeCommit, binding.Environment, measured,
			[]runrecord.PhaseMetric{{Phase: runrecord.PhasePrefill, DurationNS: measured}},
		)
		if runErr != nil {
			return recipecontract.Transcription{}, runrecord.Run{}, runErr
		}
		batch, runErr := run.Batch(binding.Key)
		if runErr == nil {
			_, runErr = artifact.CommitBatch(ctx, executor.repository, batch)
		}
		return recipecontract.Transcription{}, run, errors.Join(errors.New("fixture: transcription failed"), runErr)
	}
	text := executor.texts[testCase.Name]
	if executor.change && executor.calls[testCase.Name] > 1 {
		text += " changed"
	}
	transcription := recipecontract.Transcription{Source: testCase.Source, Text: text, Language: "en"}
	output, err := artifact.JSONContent(
		artifact.JSONContract(artifact.KindOutput, speechrecognition.TranscriptionSchema), transcription,
	)
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	run, err := runrecord.NewBoundRun(
		executor.recipe, runrecord.OutcomeSucceeded, inputs, []artifact.ID{output.Descriptor.ID}, "",
		binding.CodeCommit, binding.Environment, measured,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhasePrefill, DurationNS: measured}},
	)
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	runContent, err := run.Content()
	if err != nil {
		return recipecontract.Transcription{}, runrecord.Run{}, err
	}
	batch, err := artifact.NewDocumentBatch(binding.Key, []artifact.Content{output, runContent}, run.Lineage(), nil)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, executor.repository, batch)
	}
	return transcription, run, err
}

func TestTranscriptionResourcesExecuteEveryRunAndPublishFitness(t *testing.T) {
	fixture := newTranscriptionResourceFixture(t, []string{"first", "second"})
	defer fixture.store.Close()
	report, err := EvaluateTranscriptionResources(
		t.Context(), fixture.store, fixture.compiled, fixture.plan, fixture.executor.model, fixture.inputs,
		TranscriptionResourceOptions{WarmupRuns: 1, TimedRuns: 2},
		func(context.Context) (TranscriptionResourceExecutor, error) { return fixture.executor, nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if fixture.executor.totalCalls != 6 || fixture.executor.calls["first"] != 3 || fixture.executor.calls["second"] != 3 {
		t.Fatalf("recipe executions = total %d cases %+v", fixture.executor.totalCalls, fixture.executor.calls)
	}
	if report.Summary.Runs != 4 || report.Summary.AudioSeconds != 4 || report.Summary.WallSeconds <= 0 ||
		report.Summary.RealTimeFactor <= 0 || report.Summary.PeakHostBytes == 0 || report.QualityScore.WordErrorRate != 0 ||
		report.QualityScore.CharacterErrorRate != 0 || len(report.Executions) != 6 || report.Load.Observation.Kind() != artifact.KindEvidence {
		t.Fatalf("resource report = %+v", report)
	}
	for _, execution := range report.Executions {
		summary, summaryErr := runrecord.RequireObservationChunkSummary(t.Context(), fixture.store, execution.Observation)
		wall, wallKnown := summary.Aggregate.Measure(runrecord.ResourceWallNS)
		peak, peakKnown := summary.Aggregate.Measure(runrecord.ResourcePeakHostBytes)
		inputBytes, inputKnown := summary.Aggregate.Measure(runrecord.ResourceInputBytes)
		if summaryErr != nil || !wallKnown || wall != execution.WallNS || !peakKnown || peak != execution.PeakHostBytes ||
			!inputKnown || inputBytes == 0 || summary.Aggregate.Interactions == nil {
			t.Fatalf("execution resource evidence = %+v, err %v", summary, summaryErr)
		}
	}
	stored, found, err := artifact.ReadContent(t.Context(), fixture.store, report.ID)
	if err != nil || !found || stored.Descriptor.Schema != transcriptionResourceReportSchema {
		t.Fatalf("stored resource report = found %t schema %q err %v", found, stored.Descriptor.Schema, err)
	}
}

func TestAudioResourceFitnessRejectsChangedTranscriptionOutput(t *testing.T) {
	fixture := newTranscriptionResourceFixture(t, []string{"unstable"})
	defer fixture.store.Close()
	fixture.executor.change = true
	_, err := EvaluateTranscriptionResources(
		t.Context(), fixture.store, fixture.compiled, fixture.plan, fixture.executor.model, fixture.inputs,
		TranscriptionResourceOptions{TimedRuns: 2},
		func(context.Context) (TranscriptionResourceExecutor, error) { return fixture.executor, nil },
	)
	if err == nil || fixture.executor.totalCalls != 2 {
		t.Fatalf("unstable output err=%v calls=%d", err, fixture.executor.totalCalls)
	}
}

func TestTranscriptionResourcesCountPersistedFailures(t *testing.T) {
	fixture := newTranscriptionResourceFixture(t, []string{"failed"})
	defer fixture.store.Close()
	fixture.executor.failures["failed"] = speechrecognition.AudioAdmissionFailure
	report, err := EvaluateTranscriptionResources(
		t.Context(), fixture.store, fixture.compiled, fixture.plan, fixture.executor.model, fixture.inputs,
		TranscriptionResourceOptions{TimedRuns: 1},
		func(context.Context) (TranscriptionResourceExecutor, error) { return fixture.executor, nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if report.Summary.AdmissionFailures != 1 || report.Summary.InferenceFailures != 0 ||
		report.QualityScore.AdmissionFailures != 1 || math.Abs(report.QualityScore.WordErrorRate-1) > 1e-12 {
		t.Fatalf("failure accounting = resources %+v quality %+v", report.Summary, report.QualityScore)
	}
}

func newTranscriptionResourceFixture(t *testing.T, names []string) transcriptionResourceFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "resource dataset")
	splitID := testutil.ArtifactID(t, artifact.KindDatasetShard, "resource split")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "resource recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "resource model")
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "resource model definition")
	environmentID := testutil.ArtifactID(t, artifact.KindEvidence, "resource environment")
	descriptors := []artifact.Descriptor{{ID: datasetID}, {ID: splitID}, {ID: recipeID}, {ID: modelID}, {ID: definitionID}, {ID: environmentID}}
	cases := make([]TranscriptionCase, len(names))
	inputs := make([]TranscriptionResourceInput, len(names))
	executorCases := make(map[artifact.ID]TranscriptionCase, len(names))
	texts := make(map[string]string, len(names))
	for index, name := range names {
		data := []byte("fixture audio payload " + name)
		audioID, identifyErr := artifact.IdentifyBytes(artifact.KindFile, data)
		if identifyErr != nil {
			store.Close()
			t.Fatal(identifyErr)
		}
		profileID := testutil.ArtifactID(t, artifact.KindProfile, "resource profile "+name)
		cases[index] = TranscriptionCase{
			Name: name, Group: "clean", Source: recipecontract.AudioReference{Audio: audioID, Profile: profileID},
			Reference: "expected " + name, SampleCount: 16_000, SampleRate: 16_000,
		}
		policy := dataset.AudioInspectionPolicy{
			MaximumEncodedBytes: 1 << 20, MaximumSamples: 1 << 20, ClipThreshold: 1,
			Admission: recipecontract.AudioAdmissionPolicy{
				MinimumChannels: 1, MaximumChannels: 1, SilenceRMSThreshold: 1.0 / 32_768,
				MaximumAbsoluteDCOffset: 1,
			},
		}
		inputs[index] = TranscriptionResourceInput{
			Name: name, Data: data, Origin: dataset.AudioPayloadOrigin{Container: audioID}, Policy: policy,
		}
		descriptors = append(descriptors, artifact.Descriptor{ID: audioID, Size: uint64(len(data))}, artifact.Descriptor{ID: profileID})
		executorCases[audioID] = cases[index]
		texts[name] = cases[index].Reference
	}
	if _, err = artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "evaluation/transcription-resource/fixture", Artifacts: descriptors}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	compiled, err := CompileTranscription(TranscriptionSuite{
		Kind: TranscriptionKind, Schema: "fixture/resource/v1", Source: "pinned resource fixture",
		Dataset: datasetID, Split: splitID,
		Normalization: []TranscriptionNormalization{TranscriptionLowercase, TranscriptionStripPunctuation, TranscriptionCollapseWhitespace},
		Cases:         cases,
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	plan, err := BindTranscription(compiled, ExactAuthorities{
		ModelDefinition: definitionID, RuntimeRecipe: recipeID, CodeCommit: transcriptionTestCommit,
		Environment: environmentID, Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	executor := &recordingTranscriptionExecutor{
		repository: store, model: modelID, recipe: recipeID, cases: executorCases, texts: texts,
		failures: make(map[string]string), calls: make(map[string]uint32),
	}
	return transcriptionResourceFixture{
		store: store, compiled: compiled, plan: plan, executor: executor,
		inputs: slices.Clone(inputs),
	}
}

var _ TranscriptionResourceExecutor = (*recordingTranscriptionExecutor)(nil)
