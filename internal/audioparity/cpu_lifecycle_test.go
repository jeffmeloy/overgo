package audioparity

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testevidence"
)

type lifecycleCorpus struct {
	compiled evaluation.TranscriptionPlan
	inputs   []evaluation.TranscriptionResourceInput
}

// lifecycleHeldout reads ground truth only from the immutable test shard. Its
// selection was fixed before prediction and is disjoint from both adapter
// training and the validation rows used by the native implementation oracle.
func lifecycleHeldout(t *testing.T, l *adapterLifecycle) lifecycleCorpus {
	t.Helper()
	data, err := os.ReadFile("testdata/lifecycle_heldout.json")
	if err != nil {
		t.Fatal(err)
	}
	var selection struct {
		Schema, Selection, Path, SHA256 string
		Bytes                           uint64
		Rows                            []uint64
	}
	if err := json.Unmarshal(data, &selection); err != nil {
		t.Fatal(err)
	}
	if selection.Schema != "overgo/audio-lifecycle-heldout/v1" || selection.Path != "clean/test/0000.parquet" || len(selection.Rows) == 0 {
		t.Fatal("held-out selection differs")
	}
	shard, err := artifact.ParseID("file:sha256:" + selection.SHA256)
	if err != nil {
		t.Fatal(err)
	}
	if shard == l.origin.Container || shard == l.fixture.election.Dataset.Artifact {
		t.Fatal("held-out source overlaps training or oracle selection")
	}
	path := filepath.Join(filepath.Dir(l.fixture.storeRoot), "datasets", "librispeech_asr-clean-xet", filepath.FromSlash(selection.Path))
	verifyASRFile(t, path, shard, selection.Bytes)
	location, err := artifact.CanonicalLocalLocation(shard, artifact.LocationFile, path)
	if err != nil {
		t.Fatal(err)
	}
	l.commit(t, artifact.Batch{Key: "lifecycle/heldout-source", Artifacts: []artifact.Descriptor{{ID: shard, Size: selection.Bytes}}, Locations: []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}}}, nil)
	content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindDataset, selection.Schema), selection)
	datasetID := l.content(t, content, err)
	content, err = artifact.JSONContent(artifact.JSONContract(artifact.KindDatasetShard, selection.Schema), selection)
	splitID := l.content(t, content, err)
	l.commit(t, artifact.Batch{Key: "lifecycle/heldout-lineage", Lineage: append(artifact.DependencyLineage(datasetID, shard), artifact.DependencyLineage(splitID, datasetID, shard)...)}, nil)
	rows, err := dataset.OpenParquetRows(t.Context(), path, []string{"id", "text", "audio.bytes"}, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	suite := evaluation.TranscriptionSuite{Kind: evaluation.TranscriptionKind, Schema: selection.Schema, Source: selection.Selection, Dataset: datasetID, Split: splitID,
		Normalization: []evaluation.TranscriptionNormalization{evaluation.TranscriptionLowercase, evaluation.TranscriptionStripPunctuation, evaluation.TranscriptionCollapseWhitespace}}
	var corpus lifecycleCorpus
	for i, row := range selection.Rows {
		if slices.Contains(selection.Rows[:i], row) {
			t.Fatal("duplicate held-out row")
		}
		values, err := rows.Read(t.Context(), row)
		if err != nil {
			t.Fatal(err)
		}
		if values["id"] == nil || values["text"] == nil || values["audio.bytes"] == nil || *values["id"] == l.fixture.record.RowID {
			t.Fatal("held-out row absent or overlaps training")
		}
		payload := []byte(*values["audio.bytes"])
		// Admission is derived from this decoded record, within the same explicit
		// fixture numeric ceiling. It does not change silence/clipping policy.
		audio, _, err := media.DecodeAudio(t.Context(), payload, adapterAcceptanceMemory/4)
		if err != nil {
			t.Fatal(err)
		}
		policy := l.policy
		policy.MaximumEncodedBytes, policy.MaximumSamples = uint64(len(payload)), uint64(len(audio.Samples))
		origin := dataset.AudioPayloadOrigin{Container: shard, Column: "audio.bytes", Row: &row}
		inspection, err := dataset.InspectAudio(t.Context(), l.store, payload, origin, policy)
		if err != nil {
			t.Fatal(err)
		}
		name := *values["id"]
		suite.Cases = append(suite.Cases, evaluation.TranscriptionCase{Name: name, Group: "held-out-test", Source: inspection.Signal.Source,
			Reference: *values["text"], SampleCount: uint64(len(audio.Samples)), SampleRate: audio.Format.SampleRate})
		corpus.inputs = append(corpus.inputs, evaluation.TranscriptionResourceInput{Name: name, Reference: dataset.AudioPayloadReference{Path: path, Audio: inspection.Signal.Source.Audio, Origin: origin}, Policy: policy})
	}
	corpus.compiled, err = evaluation.CompileTranscription(suite)
	if err != nil {
		t.Fatal(err)
	}
	return corpus
}

func (corpus lifecycleCorpus) evaluate(t *testing.T, l *adapterLifecycle, definition recipe.Definition, binding speechrecognition.RunBinding) evaluation.TranscriptionResourceReport {
	t.Helper()
	metadata, err := modelrecipe.PublishTaskModelDefinition(t.Context(), l.store, definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := evaluation.BindTranscription(corpus.compiled, evaluation.ExactAuthorities{ModelDefinition: metadata, RuntimeRecipe: definition.ID,
		CodeCommit: binding.CodeCommit, Environment: binding.Environment, Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident}})
	if err != nil {
		t.Fatal(err)
	}
	report, err := evaluation.EvaluateTranscriptionResources(t.Context(), l.store, corpus.compiled, plan, definition.Model, corpus.inputs,
		evaluation.TranscriptionResourceOptions{TimedRuns: 1}, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	if report.Summary.Runs != uint64(len(corpus.inputs)) || report.QualityScore.Utterances == 0 || report.Summary.InferenceFailures != 0 {
		t.Fatalf("held-out execution denominator: %+v", report.Summary)
	}
	quality, err := evaluation.RequireTranscriptionReport(t.Context(), l.store, report.Quality)
	if err != nil || quality.Overall != report.QualityScore {
		t.Fatalf("persisted quality replay: %v", err)
	}
	for _, measurement := range report.Executions {
		if !measurement.Attempt.Valid() || !measurement.Observation.Valid() || measurement.WallNS == 0 {
			t.Fatal("unmeasured execution")
		}
		run, err := runrecord.RequireExactRun(t.Context(), l.store, measurement.Run)
		if err != nil || run.Recipe != definition.ID || !slices.Contains(run.Inputs, report.Dataset) || !slices.Contains(run.Inputs, report.Split) {
			t.Fatalf("held-out run lineage: %v", err)
		}
	}
	return report
}

func TestASRCPUClosedLoop(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": full CPU lifecycle runs as its exact mandatory plan acceptance")
	}
	l := newAdapterLifecycle(t)
	reference, referenceStore, _ := loadVADReference(t)
	vad := publishVADLifecycle(t, referenceStore, reference.Models[0], false, l.store)
	activity, err := recipe.RequireDefinition(t.Context(), l.store, vad.recipe)
	if err != nil {
		t.Fatal(err)
	}
	compose := func(base recipe.Definition) recipe.Definition {
		definition, err := modelrecipe.SegmentedTranscriptionDefinition(base, activity)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := modelrecipe.PublishCandidate(t.Context(), l.store, "lifecycle/recipe/"+definition.ID.String(), definition); err != nil {
			t.Fatal(err)
		}
		return definition
	}
	base := compose(l.base)
	corpus := lifecycleHeldout(t, l)
	environment, err := runrecord.CurrentEnvironment("cpu", "go-host")
	if err != nil {
		t.Fatal(err)
	}
	content, err := environment.Content()
	l.content(t, content, err)
	binding := speechrecognition.RunBinding{CodeCommit: strings.TrimSpace(baselineCommand(t, l.fixture.root, "git", "rev-parse", "HEAD")), Environment: environment.ID}
	before := corpus.evaluate(t, l, base, binding)
	initial := l.adapter.WeightSnapshot()
	checkpoint, directory := verifyAdapterResume(t, l)
	if slices.Equal(initial, l.adapter.WeightSnapshot()) || checkpoint.Stream.Position == 0 {
		t.Fatal("training performed no update")
	}
	batch, err := checkpoint.Batch("lifecycle/checkpoint", directory)
	l.commit(t, batch, err)
	adapted, err := modelrecipe.AdaptedTranscriptionDefinition(l.base, checkpoint.ID())
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), l.store, "lifecycle/adapted", adapted); err != nil {
		t.Fatal(err)
	}
	trained := compose(adapted)
	if err := l.store.Close(); err != nil {
		t.Fatal(err)
	}
	l.store, err = overgodb.Open(l.storePath)
	if err != nil {
		t.Fatal(err)
	}
	after := corpus.evaluate(t, l, trained, binding)
	if before.Dataset != after.Dataset || before.Split != after.Split || before.Model != after.Model {
		t.Fatal("comparison authorities differ")
	}
	comparison, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/audio-cpu-lifecycle-comparison/v1"), struct {
		Baseline, Candidate, Checkpoint artifact.ID
		Promoted                        bool
	}{before.ID, after.ID, checkpoint.ID(), false})
	if err != nil {
		t.Fatal(err)
	}
	l.commit(t, artifact.Batch{Key: "lifecycle/comparison", Contents: []artifact.Content{comparison}, Lineage: artifact.DependencyLineage(comparison.Descriptor.ID, before.ID, after.ID, checkpoint.ID(), l.spec.Dataset, l.spec.Split)}, nil)
	// The lifecycle's HTTP and recovery assertions follow this same reloaded
	// candidate; no quality observation elects it in the canonical user store.
	verifyLifecycleHTTP(t, l, corpus, trained, after, binding)
	t.Logf("CPU lifecycle: training rows=1 exact resumed updates=3 held-out rows=%d; WER before=%g after=%g RTF before=%g after=%g; fresh inference, dataset/adapter/segmentation/evaluation/HTTP lineage; no promotion, full-benchmark, process-peak, speedup or GPU claim", len(corpus.inputs), before.QualityScore.WordErrorRate, after.QualityScore.WordErrorRate, before.Summary.RealTimeFactor, after.Summary.RealTimeFactor)
}

func verifyLifecycleHTTP(t *testing.T, l *adapterLifecycle, corpus lifecycleCorpus, definition recipe.Definition, report evaluation.TranscriptionResourceReport, binding speechrecognition.RunBinding) {
	t.Helper()
	input := corpus.inputs[0]
	source, err := dataset.NewAudioPayloadReader(adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	payload, err := source.Read(t.Context(), input.Reference, input.Policy.MaximumEncodedBytes)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := speechrecognition.RequireTranscription(t.Context(), l.store, report.Executions[0].Output)
	if err != nil {
		t.Fatal(err)
	}
	handler, _ := newAudioHTTPFixture(t, l.store, definition, input.Policy, binding.CodeCommit)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, audioHTTPRequest(t, definition, payload, true))
	var actual struct {
		Text string `json:"text"`
	}
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &actual) != nil || actual.Text != expected.Text {
		t.Fatalf("reloaded trained HTTP response %d: %s", response.Code, response.Body)
	}
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, audioHTTPRequest(t, definition, payload, false))
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated trained route: %d", response.Code)
	}
	statuses := audioHTTPOperations(t, handler)
	if len(statuses) != 1 || statuses[0].State != operation.StateCompleted || statuses[0].Recipe != definition.ID || statuses[0].Run == nil {
		t.Fatal("HTTP operation did not complete under the adapted recipe")
	}
	run, err := runrecord.RequireExactRun(t.Context(), l.store, *statuses[0].Run)
	if err != nil || run.Recipe != definition.ID || !slices.Contains(run.Inputs, expected.Source.Audio) || len(run.Outputs) != 1 {
		t.Fatalf("HTTP run lost source/recipe lineage: %v", err)
	}
}
