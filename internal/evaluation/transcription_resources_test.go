package evaluation

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/speechrecognitiontest"
	"overgo/internal/testutil"
)

type transcriptionResourceFixture struct {
	store    artifact.Repository
	compiled TranscriptionPlan
	plan     Plan
	model    artifact.ID
	inputs   []TranscriptionResourceInput
}

func TestTranscriptionResourceEvidenceContract(t *testing.T) {
	t.Run("native-execution-and-measurement-scope", testTranscriptionResourcesNativeExecution)
	t.Run("refused-input-denominators", testTranscriptionResourcesFailureDenominators)
	t.Run("stable-output-required", testTranscriptionResourcesStableOutput)
	t.Run("unknown-not-zero", testTranscriptionResourcesUnknownCounters)
	t.Run("repeated-and-mismatched-attempts", testTranscriptionResourcesRunAuthority)
	t.Run("model-definition-mismatch", testTranscriptionResourcesModelMismatch)
	t.Run("profiling-refused-before-side-effects", testTranscriptionResourcesProfileRefusal)
}

func testTranscriptionResourcesNativeExecution(t *testing.T) {
	fixture := newTranscriptionResourceFixture(t)
	report, err := EvaluateTranscriptionResources(t.Context(), fixture.store, fixture.compiled, fixture.plan,
		fixture.model, fixture.inputs, TranscriptionResourceOptions{WarmupRuns: 1, TimedRuns: 2}, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if report.Summary.Runs != 4 || len(report.Executions) != 6 ||
		report.Summary.AdmissionFailures != 0 || report.Summary.InferenceFailures != 0 ||
		report.QualityScore.SilentControls != 1 || report.QualityScore.SilentControlFailures != 0 || report.QualityScore.WordErrorRate != 0 ||
		report.QualityScore.CharacterErrorRate != 0 {
		t.Fatalf("native report = %+v", report)
	}
	if report.Load.WallNS == 0 || report.Summary.WallSeconds <= 0 || report.Summary.RealTimeFactor <= 0 {
		t.Fatalf("missing measured time: %+v", report)
	}
	attempts := make(map[artifact.ID]bool)
	var wall, recorded, tail, allocations uint64
	for _, measurement := range report.Executions {
		if attempts[measurement.Attempt] {
			t.Fatalf("repeated attempt %s", measurement.Attempt)
		}
		attempts[measurement.Attempt] = true
		run, err := runrecord.RequireExactRun(t.Context(), fixture.store, measurement.Run)
		if err != nil {
			t.Fatal(err)
		}
		if measurement.WallNS < run.MeasuredNS || measurement.RecordedWorkNS != run.MeasuredNS ||
			measurement.RecordedWorkNS+measurement.UnrecordedNS != measurement.WallNS {
			t.Fatalf("different timing scopes: %+v, run %+v", measurement, run)
		}
		if !measurement.Warmup {
			wall += measurement.WallNS
			recorded += measurement.RecordedWorkNS
			tail += measurement.UnrecordedNS
			allocations += measurement.ProcessAllocations
		}
		observation, err := runrecord.RequireObservationChunkSummary(t.Context(), fixture.store, measurement.Observation)
		if err != nil {
			t.Fatal(err)
		}
		value, known := observation.Aggregate.Measure(runrecord.ResourceWallNS)
		inputBytes, inputKnown := observation.Aggregate.Measure(runrecord.ResourceInputBytes)
		_, peakKnown := observation.Aggregate.Measure(runrecord.ResourcePeakHostBytes)
		if !known || value != measurement.WallNS || !inputKnown || inputBytes == 0 ||
			peakKnown || observation.Aggregate.Interactions != nil {
			t.Fatalf("misclassified observation: %+v", observation)
		}
		if run.Outcome == runrecord.OutcomeSucceeded {
			output, err := speechrecognition.RequireTranscription(t.Context(), fixture.store, measurement.Output)
			if err != nil || output.Text != "x" {
				t.Fatalf("native output %+v: %v", output, err)
			}
		}
	}
	for _, pair := range [][2]float64{
		{report.Summary.WallSeconds, float64(wall) / 1e9},
		{report.Summary.RecordedWorkSeconds, float64(recorded) / 1e9},
		{report.Summary.UnrecordedSeconds, float64(tail) / 1e9},
	} {
		if math.Abs(pair[0]-pair[1]) > 1e-12 {
			t.Fatalf("summary span mismatch %v", pair)
		}
	}
	if report.Summary.ProcessAllocations != allocations {
		t.Fatal("allocation denominator differs")
	}
	for _, metric := range report.Metrics {
		if strings.Contains(metric.Name, "peak") {
			t.Fatalf("unobserved peak metric %+v", metric)
		}
	}
	content, found, err := artifact.ReadContent(t.Context(), fixture.store, report.ID)
	if err != nil || !found || content.Descriptor.Schema != transcriptionResourceReportSchema {
		t.Fatalf("report persistence: found=%t err=%v", found, err)
	}
	if strings.Contains(string(content.Data), "peak_host_bytes") || strings.Contains(string(content.Data), "\"interactions\"") {
		t.Fatal("report invents unavailable evidence")
	}
	repeated, err := EvaluateTranscriptionResources(t.Context(), fixture.store, fixture.compiled, fixture.plan,
		fixture.model, fixture.inputs, TranscriptionResourceOptions{TimedRuns: 1}, 1<<20)
	if err != nil || repeated.Summary.Runs != 2 {
		t.Fatalf("second evaluation must publish new measurements: %v", err)
	}
}

func testTranscriptionResourcesStableOutput(t *testing.T) {
	stable := transcriptionExecutionSignature{outcome: runrecord.OutcomeSucceeded, output: testutil.ArtifactID(t, artifact.KindOutput, "stable")}
	for _, changed := range []transcriptionExecutionSignature{
		{outcome: runrecord.OutcomeSucceeded, output: testutil.ArtifactID(t, artifact.KindOutput, "changed")},
		{outcome: runrecord.OutcomeFailed, failure: speechrecognition.AudioAdmissionFailure},
	} {
		signatures := make(map[string]transcriptionExecutionSignature)
		if err := recordTranscriptionResourceSignature(signatures, "case", stable); err != nil {
			t.Fatal(err)
		}
		if err := recordTranscriptionResourceSignature(signatures, "case", stable); err != nil {
			t.Fatal(err)
		}
		if err := recordTranscriptionResourceSignature(signatures, "case", changed); err == nil {
			t.Fatal("unstable execution accepted")
		}
	}
}

func testTranscriptionResourcesFailureDenominators(t *testing.T) {
	fixture := newTranscriptionResourceFixture(t)
	suite := fixture.compiled.suite
	suite.Cases = slices.Clone(suite.Cases)
	for index := range suite.Cases {
		if suite.Cases[index].SilentControl {
			suite.Cases[index].SilentControl = false
			suite.Cases[index].Reference = "x"
		}
	}
	compiled, err := CompileTranscription(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindTranscription(compiled, ExactAuthorities{
		ModelDefinition: fixture.plan.body.ModelDefinition, RuntimeRecipe: fixture.plan.body.RuntimeRecipe,
		CodeCommit: fixture.plan.body.CodeCommit, Environment: fixture.plan.body.Environment,
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := EvaluateTranscriptionResources(t.Context(), fixture.store, compiled, plan,
		fixture.model, fixture.inputs, TranscriptionResourceOptions{TimedRuns: 1}, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if report.Summary.Runs != 2 || report.Summary.AdmissionFailures != 1 || report.QualityScore.AdmissionFailures != 1 ||
		report.QualityScore.WordErrorRate != .5 || report.QualityScore.CharacterErrorRate != .5 {
		t.Fatalf("refused sample disappeared from denominator: %+v", report)
	}
}

func testTranscriptionResourcesUnknownCounters(t *testing.T) {
	fixture := newTranscriptionResourceFixture(t)
	attempt := testutil.ArtifactID(t, artifact.KindRun, "observed zero wall")
	fitness, err := transcriptionResourceFitness(fixture.plan, fixture.model, attempt, 0)
	if err != nil {
		t.Fatal(err)
	}
	wall, known := fitness.Measure(runrecord.ResourceWallNS)
	_, peakKnown := fitness.Measure(runrecord.ResourcePeakHostBytes)
	if !known || wall != 0 || peakKnown || fitness.Interactions != nil {
		t.Fatalf("unknown differs from observed zero: %+v", fitness)
	}
}

func testTranscriptionResourcesRunAuthority(t *testing.T) {
	fixture := newTranscriptionResourceFixture(t)
	testCase := fixture.compiled.suite.Cases[0]
	run, err := runrecord.NewBoundRun(fixture.plan.body.RuntimeRecipe, runrecord.OutcomeFailed,
		[]artifact.ID{fixture.model, fixture.plan.body.Dataset, fixture.plan.body.Split, testCase.Source.Audio, testCase.Source.Profile},
		nil, speechrecognition.AudioAdmissionFailure, fixture.plan.body.CodeCommit, fixture.plan.body.Environment, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateTranscriptionResourceRun(run, fixture.plan, fixture.model, testCase); err != nil {
		t.Fatal(err)
	}
	batch, err := run.Batch("resource-fixture/run")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), fixture.store, batch); err != nil {
		t.Fatal(err)
	}
	if err := publishPlanAuthorities(t.Context(), fixture.store, fixture.plan, nil); err != nil {
		t.Fatal(err)
	}
	load, err := publishTranscriptionLoadMeasurement(t.Context(), fixture.store, fixture.plan, fixture.model, 1,
		transcriptionMemoryBoundary{}, transcriptionMemoryBoundary{})
	if err != nil {
		t.Fatal(err)
	}
	measurement := TranscriptionResourceMeasurement{Name: testCase.Name, Run: run.ID, WallNS: 1, Outcome: run.Outcome, Failure: run.Failure}
	seen := make(map[string]bool)
	first, _, err := publishTranscriptionExecutionMeasurement(t.Context(), fixture.store, fixture.plan, fixture.model, load.Attempt, 1, measurement, seen)
	if err != nil {
		t.Fatal(err)
	}
	measurement.WallNS++
	if _, _, err := publishTranscriptionExecutionMeasurement(t.Context(), fixture.store, fixture.plan, fixture.model, load.Attempt, 1, measurement, seen); err == nil {
		t.Fatal("duplicate repetition accepted")
	}
	measurement.Repetition++
	second, _, err := publishTranscriptionExecutionMeasurement(t.Context(), fixture.store, fixture.plan, fixture.model, load.Attempt, 1, measurement, seen)
	if err != nil || first == second {
		t.Fatalf("equal run content must allow distinct measured repetitions: %v", err)
	}
	for _, mutate := range []func(*runrecord.Run){
		func(r *runrecord.Run) { r.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "foreign") },
		func(r *runrecord.Run) { r.CodeCommit = strings.Repeat("a", len(transcriptionTestCommit)) },
		func(r *runrecord.Run) { r.Environment = testutil.ArtifactID(t, artifact.KindEvidence, "foreign") },
		func(r *runrecord.Run) {
			r.Inputs = slices.DeleteFunc(slices.Clone(r.Inputs), func(id artifact.ID) bool { return id == fixture.model })
		},
		func(r *runrecord.Run) { r.ID = artifact.ID{} },
	} {
		changed := run
		mutate(&changed)
		if err := validateTranscriptionResourceRun(changed, fixture.plan, fixture.model, testCase); err == nil {
			t.Fatalf("foreign run accepted %+v", changed)
		}
	}
}

func testTranscriptionResourcesModelMismatch(t *testing.T) {
	fixture := newTranscriptionResourceFixture(t)
	other := testutil.ArtifactID(t, artifact.KindModel, "foreign model")
	if _, err := EvaluateTranscriptionResources(t.Context(), fixture.store, fixture.compiled, fixture.plan,
		other, fixture.inputs, TranscriptionResourceOptions{TimedRuns: 1}, 1<<20); err == nil ||
		!strings.Contains(err.Error(), "model definition differs") {
		t.Fatalf("model mismatch: %v", err)
	}
	if _, err := EvaluateTranscriptionResources(t.Context(), fixture.store, fixture.compiled, fixture.plan,
		fixture.model, fixture.inputs, TranscriptionResourceOptions{TimedRuns: 1}, 0); err == nil {
		t.Fatal("absent memory bound accepted")
	}
}

func testTranscriptionResourcesProfileRefusal(t *testing.T) {
	profile := filepath.Join(t.TempDir(), "profile.pprof")
	command := exec.CommandContext(t.Context(), "go", "run", "../../cmd/evaluate",
		"-transcription-resource-manifest", filepath.Join(t.TempDir(), "absent.json"), "-cpuprofile", profile)
	output, err := command.CombinedOutput()
	if err == nil || !strings.Contains(string(output), "CPU profiling is not allowed during timed transcription evaluation") {
		t.Fatalf("profile refusal: %v\n%s", err, output)
	}
	if _, err := os.Stat(profile); !os.IsNotExist(err) {
		t.Fatalf("profile created before refusal: %v", err)
	}
}

func newTranscriptionResourceFixture(t *testing.T) transcriptionResourceFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	fixture := speechrecognitiontest.Publish(t, store, "../speechrecognition/testdata/encoder.json")
	metadata, err := modelrecipetest.PublishModelDefinition(t.Context(), store, "resource-fixture/metadata", fixture.Definition.Model)
	if err != nil {
		t.Fatal(err)
	}
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "resource corpus")
	splitID := testutil.ArtifactID(t, artifact.KindDatasetShard, "resource split")
	environmentID := testutil.ArtifactID(t, artifact.KindEvidence, "resource CPU")
	batch := artifact.Batch{Key: "resource-fixture/authorities", Artifacts: []artifact.Descriptor{{ID: datasetID}, {ID: splitID}, {ID: environmentID}}}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	policy := fixture.Policy
	var cases []TranscriptionCase
	var inputs []TranscriptionResourceInput
	for _, name := range []string{"speech", "silence"} {
		wave, sampleCount := speechrecognitiontest.Wave(t, name == "silence")
		id, err := artifact.IdentifyBytes(artifact.KindFile, wave)
		if err != nil {
			t.Fatal(err)
		}
		batch := artifact.Batch{Key: "resource-fixture/" + name, Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(wave))}}}
		if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
		origin := dataset.AudioPayloadOrigin{Container: id}
		inspection, err := dataset.InspectAudio(t.Context(), store, wave, origin, policy)
		if err != nil {
			t.Fatal(err)
		}
		reference := "x"
		if name == "silence" {
			reference = ""
		}
		cases = append(cases, TranscriptionCase{Name: name, Group: "fixture", Source: inspection.Signal.Source,
			Reference: reference, SampleCount: sampleCount, SampleRate: 16000, SilentControl: name == "silence"})
		path := filepath.Join(t.TempDir(), name+".wav")
		if err := os.WriteFile(path, wave, 0600); err != nil {
			t.Fatal(err)
		}
		inputs = append(inputs, TranscriptionResourceInput{Name: name, Policy: policy,
			Reference: dataset.AudioPayloadReference{Path: path, Audio: id, Origin: origin}})
	}
	compiled, err := CompileTranscription(TranscriptionSuite{
		Kind: TranscriptionKind, Schema: "fixture/resource/v2", Source: "small encoder oracle and deterministic PCM controls",
		Dataset: datasetID, Split: splitID, Normalization: []TranscriptionNormalization{TranscriptionCollapseWhitespace}, Cases: cases,
	})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindTranscription(compiled, ExactAuthorities{ModelDefinition: metadata.Document.ID, RuntimeRecipe: fixture.Definition.ID,
		CodeCommit: transcriptionTestCommit, Environment: environmentID, Execution: ExecutionPolicy{Lifecycle: LifecycleResident}})
	if err != nil {
		t.Fatal(err)
	}
	return transcriptionResourceFixture{store: store, compiled: compiled, plan: plan, model: fixture.Definition.Model, inputs: inputs}
}
