package audioparity

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

type audioMeasurementFixture struct {
	l                               *adapterLifecycle
	corpus                          lifecycleCorpus
	directory, executable, manifest string
	binding                         speechrecognition.RunBinding
	controls                        []dataset.AudioInspection
}

func newAudioMeasurementFixture(t *testing.T) *audioMeasurementFixture {
	t.Helper()
	l := &adapterLifecycle{fixture: loadCTCTrainingFixture(t)}
	l.storePath = filepath.Join(t.TempDir(), "store")
	var err error
	l.store, err = overgodb.Open(l.storePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { l.store.Close() })
	l.base, _ = l.publishTranscriptionBase(t, l.fixture)
	l.origin.Container, err = artifact.ParseID("file:sha256:" + l.fixture.record.ShardSHA256)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("../dataset/testdata/audio_inspection_policy.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &l.policy); err != nil {
		t.Fatal(err)
	}
	f := &audioMeasurementFixture{l: l, corpus: lifecycleHeldout(t, l, "testdata/resource_heldout.json"), directory: t.TempDir()}
	sourceIdentity := audioSources(t, l.fixture.root).Identity()
	environment, err := runrecord.CurrentEnvironment("cpu", fmt.Sprintf("go-host-reference/source=%s/gomaxprocs=%d", sourceIdentity, runtime.GOMAXPROCS(0)))
	if err != nil {
		t.Fatal(err)
	}
	batch, err := environment.Batch("audio-measurement/environment/" + environment.ID.String())
	l.commit(t, batch, err)
	f.binding = speechrecognition.RunBinding{CodeCommit: strings.TrimSpace(baselineCommand(t, l.fixture.root, "git", "rev-parse", "HEAD")), Environment: environment.ID, Dataset: f.corpus.suite.Dataset, Split: f.corpus.suite.Split}
	// Silence has known geometry. Corrupt bytes have no decoded duration or
	// transcript denominator; preserve their admission receipt separately.
	silence := testutil.MonoPCM16WAV(uint32(l.fixture.frontend.SampleRate), make([]int16, l.fixture.frontend.Geometry.WindowSamples))
	for _, control := range []struct {
		name string
		data []byte
	}{
		{"silence", silence}, {"corrupt", []byte("fLaC")},
	} {
		id, err := artifact.IdentifyBytes(artifact.KindFile, control.data)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(f.directory, control.name)
		if err := os.WriteFile(path, control.data, 0600); err != nil {
			t.Fatal(err)
		}
		location, err := artifact.CanonicalLocalLocation(id, artifact.LocationFile, path)
		if err != nil {
			t.Fatal(err)
		}
		l.commit(t, artifact.Batch{Key: "audio-measurement/control/" + id.String(), Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(control.data))}}, Locations: []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}}}, nil)
		origin := dataset.AudioPayloadOrigin{Container: id}
		inspection, err := dataset.InspectAudio(t.Context(), l.store, control.data, origin, l.policy)
		if err != nil || inspection.Decision.Outcome == recipecontract.AudioAdmissionAccepted || len(inspection.Samples) != 0 {
			t.Fatalf("negative control %s was not quarantined: %v", control.name, err)
		}
		f.controls = append(f.controls, inspection)
		if control.name == "silence" {
			f.corpus.suite.Cases = append(f.corpus.suite.Cases, evaluation.TranscriptionCase{Name: control.name, Group: "negative-controls", Source: inspection.Signal.Source, SampleCount: uint64(l.fixture.frontend.Geometry.WindowSamples), SampleRate: uint64(l.fixture.frontend.SampleRate), SilentControl: true})
			f.corpus.inputs = append(f.corpus.inputs, evaluation.TranscriptionResourceInput{Name: control.name, Reference: dataset.AudioPayloadReference{Path: path, Audio: id, Origin: origin}, Policy: l.policy})
		} else if inspection.DecodeError == "" || inspection.Signal.SampleCount != 0 {
			t.Fatal("corrupt control acquired decoded geometry or lost its decode error")
		}
	}
	f.corpus.compiled, err = evaluation.CompileTranscription(f.corpus.suite)
	if err != nil {
		t.Fatal(err)
	}
	suitePath := filepath.Join(f.directory, "suite.json")
	writeAudioMeasurementJSON(t, suitePath, f.corpus.suite)
	var inputs []map[string]any
	for _, input := range f.corpus.inputs {
		inputs = append(inputs, map[string]any{"name": input.Name, "path": input.Reference.Path, "origin": input.Reference.Origin, "policy": input.Policy})
	}
	f.manifest = filepath.Join(f.directory, "resources.json")
	writeAudioMeasurementJSON(t, f.manifest, map[string]any{"suite": suitePath, "model": l.base.Model, "runtime_recipe": l.base.ID, "code_commit": f.binding.CodeCommit, "environment": f.binding.Environment, "memory_bytes": adapterAcceptanceMemory, "options": evaluation.TranscriptionResourceOptions{WarmupRuns: 1, TimedRuns: 1}, "inputs": inputs})
	f.executable = filepath.Join(f.directory, "evaluate.exe")
	baselineCommand(t, l.fixture.root, "go", "build", "-o", f.executable, "./cmd/evaluate")
	return f
}

func writeAudioMeasurementJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}

type audioMeasuredProcess struct {
	Report evaluation.TranscriptionResourceReport
	Stream runrecord.ObservationStream
}

func (f *audioMeasurementFixture) measure(t *testing.T) audioMeasuredProcess {
	t.Helper()
	if err := f.l.store.Close(); err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(t.Context(), f.executable, "-repo", f.l.storePath, "-transcription-resource-manifest", f.manifest)
	command.Dir = f.l.fixture.root
	measured, err := processmeasure.Measure(command)
	if err != nil {
		t.Fatalf("measured evaluator: %v\n%s", err, measured.Output)
	}
	if measured.Wall <= 0 || measured.PeakWorkingSetByte == 0 {
		t.Fatal("process measurements unavailable")
	}
	f.l.store, err = overgodb.Open(f.l.storePath)
	if err != nil {
		t.Fatal(err)
	}
	var reportID artifact.ID
	for field := range strings.FieldsSeq(string(measured.Output)) {
		id, err := artifact.ParseID(field)
		if err == nil && id.Kind() == artifact.KindEvaluation {
			reportID = id
			break
		}
	}
	content, err := artifact.RequireTypedContent(t.Context(), f.l.store, reportID)
	if err != nil {
		t.Fatal(err)
	}
	var result audioMeasuredProcess
	if err := json.Unmarshal(content.Data, &result.Report); err != nil {
		t.Fatal(err)
	}
	result.Report.ID = reportID
	report := result.Report
	quality, err := evaluation.RequireTranscriptionReport(t.Context(), f.l.store, report.Quality)
	if err != nil || quality.Overall != report.QualityScore || report.Dataset != f.corpus.suite.Dataset || report.Split != f.corpus.suite.Split || report.Model != f.l.base.Model || report.Summary.Runs != uint64(len(f.corpus.inputs)) || len(report.Executions) != int(report.Options.WarmupRuns+report.Options.TimedRuns)*len(f.corpus.inputs) {
		t.Fatalf("quality, authority or denominator mismatch: %v", err)
	}
	if quality.Overall.Utterances != uint64(len(f.corpus.inputs)-1) || quality.Overall.AdmissionFailures != 0 || quality.Overall.InferenceFailures != 0 || quality.Overall.SilentControls != 1 || quality.Overall.SilentControlFailures != 0 {
		t.Fatalf("held-out coverage failed: %+v", quality.Overall)
	}
	speakers := make(map[string]bool)
	for _, group := range quality.Groups {
		if strings.HasPrefix(group.Name, "speaker-") {
			speakers[group.Name] = true
		}
	}
	if len(speakers) != 3 {
		t.Fatalf("speaker denominator: %d", len(speakers))
	}
	seen := make(map[artifact.ID]bool)
	for _, execution := range report.Executions {
		if seen[execution.Attempt] || !execution.Attempt.Valid() || execution.WallNS == 0 {
			t.Fatal("execution was not independently measured")
		}
		seen[execution.Attempt] = true
		run, err := runrecord.RequireExactRun(t.Context(), f.l.store, execution.Run)
		if err != nil || run.Recipe != f.l.base.ID || run.Environment != f.binding.Environment || run.CodeCommit != f.binding.CodeCommit || !slices.Contains(run.Inputs, report.Dataset) || !slices.Contains(run.Inputs, report.Split) {
			t.Fatalf("source-bound run: %v", err)
		}
	}
	// The outer measurement includes process startup, source I/O, cold load,
	// warmup, timed inference, scoring and publication. It is not inference RSS.
	receipt, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/audio-process-measurement/v1"), struct {
		Report        artifact.ID `json:"report"`
		WallNS        uint64      `json:"wall_ns"`
		PeakHostBytes uint64      `json:"peak_host_bytes"`
	}{reportID, uint64(measured.Wall), measured.PeakWorkingSetByte})
	if err != nil {
		t.Fatal(err)
	}
	fitness, err := runrecord.NewResourceFitness(runrecord.ResourceFitness{Scope: runrecord.ResourceScope{Surface: runrecord.SurfaceEvaluation, Model: report.Model, Hardware: f.binding.Environment, Workload: report.Plan, Attempt: receipt.Descriptor.ID}, Measures: []runrecord.ResourceMeasure{{Metric: runrecord.ResourceWallNS, Value: uint64(measured.Wall)}, {Metric: runrecord.ResourcePeakHostBytes, Value: measured.PeakWorkingSetByte}}})
	if err != nil {
		t.Fatal(err)
	}
	chunk, err := runrecord.NewInitialObservationChunk(fitness, runrecord.ObservationSampleExecution, uint64(measured.Wall))
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("audio-process/"+receipt.Descriptor.ID.DigestHex(), []artifact.Content{receipt}, artifact.DependencyLineage(receipt.Descriptor.ID, reportID, report.Plan, report.Model, f.binding.Environment), nil)
	if err != nil {
		t.Fatal(err)
	}
	summary, err := runrecord.BindObservationChunk(t.Context(), f.l.store, &batch, chunk)
	if err != nil {
		t.Fatal(err)
	}
	f.l.commit(t, batch, nil)
	result.Stream, err = runrecord.RequireObservationStream(t.Context(), f.l.store, receipt.Descriptor.ID, summary.ID, runrecord.ObservationStreamBounds{MaxChunks: 1, MaxRawBytes: summary.Stats.Bytes})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("held-out=%s speakers=%d utterances=%d WER=%.6f CER=%.6f cold_load_ns=%d warm_timed_seconds=%.6f process_wall=%s process_peak_bytes=%d; corrupt admission=%s outside decoded-duration denominator; no GPU, training, full-corpus or promotion claim", reportID, len(speakers), quality.Overall.Utterances, quality.Overall.WordErrorRate, quality.Overall.CharacterErrorRate, report.Load.WallNS, report.Summary.WallSeconds, measured.Wall, measured.PeakWorkingSetByte, f.controls[len(f.controls)-1].DecisionID)
	return result
}

func TestAudioBenchmarkCoverageAcceptance(t *testing.T) {
	t.Parallel()
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": held-out measurement runs as its exact mandatory batch acceptance")
	}
	f := newAudioMeasurementFixture(t)
	f.measure(t)
}
