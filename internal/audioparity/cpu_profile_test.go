package audioparity

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/strictjson"
	"overgo/internal/testskip"
)

// Only target-free inputs cross this child boundary. Profiling measurements
// never enter the timed resource comparison or its calibration repeats.
type audioProfileRequest struct {
	Store, Profile, Predictions string
	Recipe                      artifact.ID
	Inputs                      []evaluation.TranscriptionResourceInput
	Binding                     speechrecognition.RunBinding
}

type audioProfilePrediction struct {
	Name, Text, Failure string
	Outcome             runrecord.Outcome
}

func runAudioProfile(t *testing.T, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var request audioProfileRequest
	if err := strictjson.DecodeBytes(data, &request); err != nil {
		t.Fatal(err)
	}
	if request.Profile == "" || request.Predictions == "" || len(request.Inputs) == 0 {
		t.Fatal("incomplete profile request")
	}
	store, err := overgodb.Open(request.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	session, err := speechrecognition.LoadSession(t.Context(), store, request.Recipe, adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(context.WithoutCancel(t.Context()))
	source, err := dataset.NewAudioPayloadReader(adapterAcceptanceMemory)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	profile, err := os.Create(request.Profile)
	if err != nil {
		t.Fatal(err)
	}
	if err := pprof.StartCPUProfile(profile); err != nil {
		profile.Close()
		t.Fatal(err)
	}
	defer func() {
		pprof.StopCPUProfile()
		if err := profile.Close(); err != nil {
			t.Error(err)
		}
	}()
	var predictions []audioProfilePrediction
	for _, input := range request.Inputs {
		payload, err := source.Read(t.Context(), input.Reference, input.Policy.MaximumEncodedBytes)
		if err != nil {
			t.Fatal(err)
		}
		lease, err := session.Lease(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		binding := request.Binding
		binding.Key = "audio-profile/" + input.Name
		transcript, run, executeErr := lease.Transcribe(t.Context(), payload, input.Reference.Origin, input.Policy, binding)
		if err := lease.Release(); err != nil {
			t.Fatal(err)
		}
		if !run.ID.Valid() || executeErr != nil && run.Outcome != runrecord.OutcomeFailed {
			t.Fatalf("profile execution: %v", executeErr)
		}
		predictions = append(predictions, audioProfilePrediction{Name: input.Name, Text: transcript.Text, Failure: run.Failure, Outcome: run.Outcome})
	}
	writeAudioMeasurementJSON(t, request.Predictions, predictions)
}

func (f *audioMeasurementFixture) profile(t *testing.T, baseline audioMeasuredProcess) {
	t.Helper()
	profilePath, predictionsPath := filepath.Join(f.directory, "inference.pprof"), filepath.Join(f.directory, "profile-predictions.json")
	requestPath := filepath.Join(f.directory, "profile-request.json")
	writeAudioMeasurementJSON(t, requestPath, audioProfileRequest{Store: f.l.storePath, Profile: profilePath, Predictions: predictionsPath, Recipe: f.l.base.ID, Inputs: f.corpus.inputs, Binding: f.binding})
	if err := f.l.store.Close(); err != nil {
		t.Fatal(err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(t.Context(), executable, "-test.run=^TestASRCPUSimplificationAcceptance$", "-test.v")
	command.Env = append(os.Environ(), "OVERGO_AUDIO_PROFILE_REQUEST="+requestPath)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("profile child: %v\n%s", err, output)
	}
	f.l.store, err = overgodb.Open(f.l.storePath)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(predictionsPath)
	if err != nil {
		t.Fatal(err)
	}
	var predictions []audioProfilePrediction
	if err := strictjson.DecodeBytes(data, &predictions); err != nil {
		t.Fatal(err)
	}
	if len(predictions) != len(f.corpus.inputs) {
		t.Fatal("profile skipped inputs")
	}
	seen := make(map[string]bool)
	for _, prediction := range predictions {
		if seen[prediction.Name] {
			t.Fatal("profile repeated a source instead of covering every input")
		}
		seen[prediction.Name] = true
		matched := false
		for _, execution := range baseline.Report.Executions {
			if execution.Warmup || execution.Name != prediction.Name {
				continue
			}
			matched = true
			if prediction.Outcome != execution.Outcome || prediction.Failure != execution.Failure {
				t.Fatal("profile changed execution outcome")
			}
			if execution.Output.Valid() {
				transcript, err := speechrecognition.RequireTranscription(t.Context(), f.l.store, execution.Output)
				if err != nil || transcript.Text != prediction.Text {
					t.Fatalf("profile changed raw transcript: %v", err)
				}
			}
		}
		if !matched {
			t.Fatal("profile prediction has no measured source")
		}
	}
	top := baselineCommand(t, f.l.fixture.root, "go", "tool", "pprof", "-top", "-nodecount=12", executable, profilePath)
	if !strings.Contains(top, "Total samples") || !strings.Contains(top, "overgo/internal/") {
		t.Fatalf("profile has no native execution samples:\n%s", top)
	}
	t.Logf("separate CPU profile; exact transcript/outcome parity; no timed evidence credited:\n%s", top)
}

func TestASRCPUSimplificationAcceptance(t *testing.T) {
	t.Parallel()
	if path := os.Getenv("OVERGO_AUDIO_PROFILE_REQUEST"); path != "" {
		runAudioProfile(t, path)
		return
	}
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": exact CPU simplification acceptance profiles a separate real-model child")
	}
	f := newAudioMeasurementFixture(t)
	f.profile(t, f.measure(t))
}
