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
	"overgo/internal/testcheck"
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
	data := testcheck.Must(os.ReadFile(path))(t)
	var request audioProfileRequest
	testcheck.NoError(t, strictjson.DecodeBytes(data, &request))
	if request.Profile == "" || request.Predictions == "" || len(request.Inputs) == 0 {
		t.Fatal("incomplete profile request")
	}
	store := testcheck.Must(overgodb.Create(request.Store))(t)
	defer store.Close()
	session := testcheck.Must(speechrecognition.LoadSession(t.Context(), store, request.Recipe, adapterAcceptanceMemory))(t)
	defer session.Close(context.WithoutCancel(t.Context()))
	source := testcheck.Must(dataset.NewAudioPayloadReader(adapterAcceptanceMemory))(t)
	defer source.Close()
	profile := testcheck.Must(os.Create(request.Profile))(t)
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
		payload := testcheck.Must(source.Read(t.Context(), input.Reference, input.Policy.MaximumEncodedBytes))(t)
		lease := testcheck.Must(session.Lease(t.Context()))(t)
		binding := request.Binding
		binding.Key = "audio-profile/" + input.Name
		transcript, run, executeErr := lease.Transcribe(t.Context(), payload, input.Reference.Origin, input.Policy, binding)
		testcheck.NoError(t, lease.Release())
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
	testcheck.NoError(t, f.l.store.Close())
	executable := testcheck.Must(os.Executable())(t)
	command := exec.CommandContext(t.Context(), executable, "-test.run=^"+audioProfileOwner+"$", "-test.v")
	command.Env = append(os.Environ(), audioProfileRequestEnvironment+"="+requestPath)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("profile child: %v\n%s", err, output)
	}
	f.l.store = testcheck.Must(overgodb.Create(f.l.storePath))(t)
	data := testcheck.Must(os.ReadFile(predictionsPath))(t)
	var predictions []audioProfilePrediction
	testcheck.NoError(t, strictjson.DecodeBytes(data, &predictions))
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

// audioProfileRequestEnvironment carries the profile request to the child
// process the fitness acceptance re-executes as itself.
const audioProfileRequestEnvironment = "OVERGO_AUDIO_PROFILE_REQUEST"

// audioProfileOwner is the test the profile child runs as; the fitness
// acceptance owns every measured child run.
const audioProfileOwner = "TestAudioResourceFitnessAcceptance"
