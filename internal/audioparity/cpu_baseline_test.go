package audioparity

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testutil"
)

// TestASRCPUBaselineAcceptance runs the production evaluator in a separate
// process on the pinned real checkpoint. It does not inject oracle transcripts.
// The optional store is restricted to this worktree's catalog; normal test runs
// use temporary storage. This same acceptance can retain a measured baseline.
func TestASRCPUBaselineAcceptance(t *testing.T) {
	root := testutil.RepoRoot(t)
	referenceRoot := resolveReferenceRoots(t).store
	dataRoot := filepath.Dir(referenceRoot)
	storeRoot, fixtureRoot := t.TempDir(), t.TempDir()
	if retained := os.Getenv("OVERGO_AUDIO_BASELINE_STORE"); retained != "" {
		absolute, err := filepath.Abs(retained)
		if err != nil || !strings.EqualFold(filepath.Clean(absolute), filepath.Join(root, "overgodb-store")) {
			t.Fatal("retained baseline must use this worktree's catalog")
		}
		storeRoot, fixtureRoot = absolute, filepath.Join(root, "tmp", "asr-cpu-baseline")
		if err := os.MkdirAll(fixtureRoot, 0700); err != nil {
			t.Fatal(err)
		}
	}
	declarationPath := filepath.Join(root, "internal", "audioparity", "testdata", "registered_models.json")
	baselineCommand(t, root, "go", "run", "./cmd/recipe", "register", "-repo", storeRoot,
		"-root", filepath.Join(dataRoot, "models"), "-spec", declarationPath)
	store, err := overgodb.Open(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	commitBatch := func(batch artifact.Batch, err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
			t.Fatal(err)
		}
	}
	declarationBytes, err := os.ReadFile(declarationPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyRegisteredAudioModels(t.Context(), store, dataRoot, declarationBytes); err != nil {
		t.Fatal(err)
	}
	electionBytes, err := os.ReadFile("testdata/granite_speech_5_election.json")
	if err != nil {
		t.Fatal(err)
	}
	election, err := NormalizeElection(electionBytes)
	if err != nil || len(election.Observations) == 0 {
		t.Fatalf("pinned election: %v", err)
	}
	var encoder speechrecognition.Declaration
	var frontend audiodsp.FrontendConfig
	var policy dataset.AudioInspectionPolicy
	for path, destination := range map[string]any{
		"recipes/granite5asr_execution.json":               &encoder,
		"../audiodsp/testdata/power_logmel_frontend.json":  &frontend,
		"../dataset/testdata/audio_inspection_policy.json": &policy,
	} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, destination); err != nil {
			t.Fatal(err)
		}
	}
	architecture, _, err := graniteRecipe()
	if err != nil {
		t.Fatal(err)
	}
	grouping := audiodsp.GroupedFeatureConfig{StackFrames: int(architecture.Preprocessor.StackFactor),
		DeltaRadius: int(architecture.Preprocessor.DeltaWinLength-1) / 2, FinalFrameSamples: 1}
	profile, err := speechrecognition.NewExecutionProfile(speechrecognition.ExecutionProfile{Frontend: frontend, Grouping: grouping, Encoder: encoder, BlankToken: int(architecture.Config.PadTokenID), Language: "en"})
	if err != nil {
		t.Fatal(err)
	}
	contract, err := modelrecipe.NewAudioContract(recipecontract.AudioFormat{SampleRate: uint64(frontend.SampleRate), Channels: 1, Encoding: "pcm-f32le"}, frontend.Geometry, artifact.ID{}, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	commitBatch(profile.Batch("baseline/profile/" + profile.ID.String()))
	commitBatch(contract.Batch("baseline/contract/" + contract.ID.String()))
	var registered []struct {
		Model     artifact.ID `json:"model"`
		Inventory artifact.ID `json:"tensor_inventory"`
	}
	if err := json.Unmarshal(declarationBytes, &registered); err != nil {
		t.Fatal(err)
	}
	var inventory artifact.ID
	for _, entry := range registered {
		if entry.Model == election.Model.ID {
			inventory = entry.Inventory
		}
	}
	if !inventory.Valid() {
		t.Fatal("elected model is not in the exact registration")
	}
	var tokenizer artifact.ID
	for _, component := range election.Model.Components {
		if component.Role == artifact.ComponentTokenizer && component.Name == "tokenizer.json" {
			tokenizer = component.Artifact
		}
	}
	definition, err := modelrecipe.TranscriptionDefinition(election.Model.ID, contract.ID, profile.ID, tokenizer, inventory)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), store, "baseline/recipe/"+definition.ID.String(), definition); err != nil {
		t.Fatal(err)
	}
	corpusPath := filepath.Join(dataRoot, "datasets", "librispeech_asr-clean-xet", filepath.FromSlash(election.Dataset.Path))
	registration, err := dataset.RegisterDirectoryDataset(t.Context(), store, "librispeech-clean-validation", filepath.Dir(corpusPath))
	if err != nil || registration.Dataset.String() != "dataset:sha256:39b4aa4872600afff4be0633dc066adfe5ae36ac17ade4a6ac4a16e26852b2e3" {
		t.Fatalf("validation inventory differs: %v", err)
	}
	batch := artifact.Batch{Key: "baseline/source/" + election.Dataset.Artifact.String(),
		Artifacts: []artifact.Descriptor{{ID: election.Dataset.Artifact, Size: election.Dataset.Bytes, MediaType: election.Dataset.MediaType}}}
	location, err := artifact.CanonicalLocalLocation(election.Dataset.Artifact, artifact.LocationFile, corpusPath)
	if err != nil {
		t.Fatal(err)
	}
	batch.Locations = []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}}
	if prior, found, err := store.Artifact(t.Context(), election.Dataset.Artifact); err != nil {
		t.Fatal(err)
	} else if found {
		batch.Artifacts[0] = prior
	}
	commitBatch(batch, nil)
	// Fixed integration workspace, not an observed process peak or runtime default.
	const memoryBytes = 4 << 30
	source, err := dataset.NewAudioPayloadReader(memoryBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	suite := evaluation.TranscriptionSuite{Kind: evaluation.TranscriptionKind, Schema: "overgo/asr-cpu-baseline/v1",
		Source:  "Pinned validation rows from the CPU oracle election plus synthetic negative controls; no held-out or full-corpus claim",
		Dataset: registration.Dataset, Split: election.Dataset.Artifact,
		Normalization: []evaluation.TranscriptionNormalization{evaluation.TranscriptionLowercase, evaluation.TranscriptionStripPunctuation, evaluation.TranscriptionCollapseWhitespace}}
	var inputs []map[string]any
	add := func(name, group, path, target string, origin dataset.AudioPayloadOrigin, encoded []byte, samples, rate uint64, silent bool) {
		t.Helper()
		inspection, err := dataset.InspectAudio(t.Context(), store, encoded, origin, policy)
		if err != nil {
			t.Fatal(err)
		}
		suite.Cases = append(suite.Cases, evaluation.TranscriptionCase{Name: name, Group: group, Source: inspection.Signal.Source,
			Reference: target, SampleCount: samples, SampleRate: rate, SilentControl: silent})
		inputs = append(inputs, map[string]any{"name": name, "path": path, "origin": origin, "policy": policy})
	}
	for index, observation := range election.Observations {
		origin := dataset.AudioPayloadOrigin{Container: election.Dataset.Artifact, Column: "audio.bytes", Row: new(uint64(index))}
		encoded, err := source.Read(t.Context(), dataset.AudioPayloadReference{Path: corpusPath, Audio: observation.Audio, Origin: origin}, policy.MaximumEncodedBytes)
		if err != nil {
			t.Fatal(err)
		}
		add(observation.Fixture, "librispeech-validation", corpusPath, observation.Expected, origin, encoded, observation.Samples, uint64(observation.SampleRate), false)
	}
	for _, name := range []string{"silence", "clipped", "wrong-rate"} {
		samples := make([]int16, frontend.Geometry.WindowSamples)
		rate := uint64(frontend.SampleRate)
		target := "CONTROL"
		for index := range samples {
			switch name {
			case "clipped":
				samples[index] = math.MinInt16
			case "wrong-rate":
				samples[index] = math.MaxInt16 / 4
				if index%2 != 0 {
					samples[index] = -samples[index]
				}
			}
		}
		if name == "silence" {
			target = ""
		}
		if name == "wrong-rate" {
			rate /= 2
		}
		encoded := testutil.MonoPCM16WAV(uint32(rate), samples)
		id, err := artifact.IdentifyBytes(artifact.KindFile, encoded)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(fixtureRoot, name+".wav")
		if err := os.WriteFile(path, encoded, 0600); err != nil {
			t.Fatal(err)
		}
		location, err := artifact.CanonicalLocalLocation(id, artifact.LocationFile, path)
		if err != nil {
			t.Fatal(err)
		}
		commitBatch(artifact.Batch{Key: "baseline/control/" + id.String(), Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(encoded))}},
			Locations: []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}}}, nil)
		add(name, "negative-controls", path, target, dataset.AudioPayloadOrigin{Container: id}, encoded, uint64(len(samples)), rate, name == "silence")
	}
	sourceIdentity := audioSources(t, root).Identity()
	environment, err := runrecord.CurrentEnvironment("cpu", fmt.Sprintf("go-host-reference/source=%s/gomaxprocs=%d", sourceIdentity, runtime.GOMAXPROCS(0)))
	if err != nil {
		t.Fatal(err)
	}
	commitBatch(environment.Batch("baseline/environment/" + environment.ID.String()))
	commit := strings.TrimSpace(baselineCommand(t, root, "git", "rev-parse", "HEAD"))
	suitePath := filepath.Join(fixtureRoot, "suite.json")
	manifestPath := filepath.Join(fixtureRoot, "resources.json")
	// One warmup and two measured rounds are a bounded acceptance protocol,
	// not a calibrated latency distribution or a throughput promotion threshold.
	options := evaluation.TranscriptionResourceOptions{WarmupRuns: 1, TimedRuns: 2}
	for path, body := range map[string]any{
		suitePath: suite,
		manifestPath: map[string]any{"suite": suitePath, "model": election.Model.ID, "runtime_recipe": definition.ID,
			"code_commit": commit, "environment": environment.ID, "memory_bytes": memoryBytes, "options": options, "inputs": inputs},
	} {
		encoded, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, encoded, 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	output := baselineCommand(t, root, "go", "run", "./cmd/evaluate", "-repo", storeRoot, "-transcription-resource-manifest", manifestPath)
	var reportID artifact.ID
	for field := range strings.FieldsSeq(output) {
		id, err := artifact.ParseID(field)
		if err == nil && id.Kind() == artifact.KindEvaluation {
			reportID = id
			break
		}
	}
	store, err = overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	content, err := artifact.RequireTypedContent(t.Context(), store, reportID)
	if err != nil {
		t.Fatal(err)
	}
	var report evaluation.TranscriptionResourceReport
	if err := json.Unmarshal(content.Data, &report); err != nil {
		t.Fatal(err)
	}
	if report.Model != election.Model.ID || report.Dataset != suite.Dataset || report.Split != suite.Split ||
		report.Options != options || report.Summary.Runs != uint64(options.TimedRuns)*uint64(len(suite.Cases)) ||
		len(report.Executions) != int(options.WarmupRuns+options.TimedRuns)*len(suite.Cases) ||
		report.Summary.AdmissionFailures != uint64(options.TimedRuns) || report.Summary.InferenceFailures != uint64(options.TimedRuns) ||
		report.Load.WallNS == 0 || report.QualityScore.SilentControls != 1 || report.QualityScore.SilentControlFailures != 0 ||
		report.QualityScore.Utterances != uint64(len(election.Observations)+2) ||
		report.QualityScore.AdmissionFailures != 1 || report.QualityScore.InferenceFailures != 1 {
		t.Fatalf("baseline denominator or authority differs: %+v", report)
	}
	seen := make(map[artifact.ID]bool)
	want := make(map[string]string)
	for _, observation := range election.Observations {
		want[observation.Fixture] = observation.Observed
	}
	for _, measured := range report.Executions {
		if seen[measured.Attempt] || !measured.Attempt.Valid() || measured.WallNS == 0 {
			t.Fatal("missing or repeated fresh attempt")
		}
		seen[measured.Attempt] = true
		run, err := runrecord.RequireExactRun(t.Context(), store, measured.Run)
		if err != nil || run.Recipe != definition.ID || run.Environment != environment.ID || run.CodeCommit != commit {
			t.Fatalf("run authority: %v", err)
		}
		if expected, speech := want[measured.Name]; speech {
			transcript, err := speechrecognition.RequireTranscription(t.Context(), store, measured.Output)
			if err != nil || run.Outcome != runrecord.OutcomeSucceeded || transcript.Text != expected {
				t.Fatalf("fresh transcript %s differs: %q, %v", measured.Name, transcript.Text, err)
			}
		} else {
			failure := speechrecognition.AudioAdmissionFailure
			if measured.Name == "wrong-rate" {
				failure = speechrecognition.AudioFormatFailure
			}
			if run.Outcome != runrecord.OutcomeFailed || run.Failure != failure {
				t.Fatalf("negative control %s outcome=%s failure=%s", measured.Name, run.Outcome, run.Failure)
			}
		}
		observation, err := runrecord.RequireObservationChunkSummary(t.Context(), store, measured.Observation)
		if err != nil {
			t.Fatal(err)
		}
		if _, known := observation.Aggregate.Measure(runrecord.ResourcePeakHostBytes); known || observation.Aggregate.Interactions != nil {
			t.Fatal("unmeasured resources became available")
		}
	}
	qualityContent, err := artifact.RequireTypedContent(t.Context(), store, report.Quality)
	if err != nil {
		t.Fatal(err)
	}
	var quality evaluation.TranscriptionReport
	if err := json.Unmarshal(qualityContent.Data, &quality); err != nil {
		t.Fatal(err)
	}
	for _, group := range quality.Groups {
		t.Logf("group=%s utterances=%d WER=%.6f CER=%.6f admission_failures=%d inference_failures=%d", group.Name, group.Utterances, group.WordErrorRate, group.CharacterErrorRate, group.AdmissionFailures, group.InferenceFailures)
	}
	t.Logf("CPU baseline report=%s model=%s recipe=%s source=%s cold_load=%s measured_runs=%d audio_seconds=%.6f wall_seconds=%.6f RTF=%.6f; source I/O excluded, peak memory and interaction counters unavailable; no GPU, training, held-out scoring or promotion", reportID, report.Model, definition.ID, sourceIdentity, time.Duration(report.Load.WallNS), report.Summary.Runs, report.Summary.AudioSeconds, report.Summary.WallSeconds, report.Summary.RealTimeFactor)
}

func baselineCommand(t *testing.T, root, name string, arguments ...string) string {
	t.Helper()
	command := exec.CommandContext(t.Context(), name, arguments...)
	command.Dir = root
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("%s %v: %v\n%s", name, arguments, err, output)
	}
	return string(output)
}
