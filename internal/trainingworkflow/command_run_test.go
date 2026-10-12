package trainingworkflow

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/trainingprogram"
)

// TestRewardRunHandsItsStepsToItsStarter holds the two ends of the file a
// training command leaves a reward run's steps in for the process that
// started it, with no child process.
//
// A run whose starter named a file writes there the steps it returns, with
// its checkpoint published and its session recorded. The file must be new: a
// second run that names the same file is refused, and the file still holds
// the first run's steps. A run that fails writes none.
//
// The starter's side, ExecuteInCommand, refuses before it starts anything a
// request the command's flags cannot carry: one held to the host, with a
// device trainer, a reference, a sequence limit, an objective scale, a frozen
// lexicon, a session that samples its own answers, a preference mode or
// margin, an observer of preference steps, a bound on projected time or an
// audio specification, and one that records no session. The command path
// here names no file, so the request that is carried fails at the start of
// the command, and each refused one does not get that far. And a context
// that has already ended returns its own error, with nothing started.
func TestRewardRunHandsItsStepsToItsStarter(t *testing.T) {
	// The run names a code revision, as one from a clean tree does, so that
	// it verifies its recipe whatever state this tree is in.
	stubExecutableCodeCommit(t, strings.Repeat("d4", 20))
	root, request := orderedRewardFixture(t, "c", "d")
	handoff := filepath.Join(root, "steps.json")
	t.Setenv(rewardStepsEnvironment, handoff)
	request.Host, request.Device, request.Observations = false, hostWeightedStep, request.Repository.(artifact.Repository)
	request.Steps, request.OutputDirectory = 1, filepath.Join(root, "first")
	result, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	var handed []trainingprogram.GRPOObservation
	if err := jsonfile.DecodeStrict(handoff, &handed); err != nil || len(handed) != 1 || !reflect.DeepEqual(handed, result.GRPO) {
		t.Fatalf("the file the run's starter named holds %+v (%v); want the one step the run returned, %+v", handed, err, result.GRPO)
	}
	if !result.Observation.Valid() || result.Checkpoint.ID().Kind() != artifact.KindCheckpoint {
		t.Fatalf("the run that handed its steps over = observation %s, checkpoint %s; want its session recorded and its checkpoint published", result.Observation, result.Checkpoint.ID())
	}
	request.OutputDirectory = filepath.Join(root, "second")
	if _, err := Execute(t.Context(), request); err == nil || !strings.Contains(err.Error(), "reward steps handoff") {
		t.Fatalf("a second run naming the same file = %v; want it refused, the file being another run's", err)
	}
	var kept []trainingprogram.GRPOObservation
	if err := jsonfile.DecodeStrict(handoff, &kept); err != nil || !reflect.DeepEqual(kept, handed) {
		t.Fatalf("after the refused run the file holds %+v (%v); want the first run's steps still", kept, err)
	}
	// A run that fails hands nothing over: here one refused for the objective
	// scale a reward session does not take.
	unwritten := filepath.Join(root, "unwritten.json")
	t.Setenv(rewardStepsEnvironment, unwritten)
	failed := request
	failed.ObjectiveScale, failed.OutputDirectory = 1, filepath.Join(root, "failed")
	if _, err := Execute(t.Context(), failed); err == nil {
		t.Fatal("a reward session with an objective scale ran; want it refused")
	}
	if _, err := os.Stat(unwritten); !os.IsNotExist(err) {
		t.Fatalf("a run that failed left a file for its starter (%v); want none", err)
	}

	// The starter's side. The carried request is the workbench's: off the
	// host, with no device trainer of its own, its session recorded.
	command := filepath.Join(root, "absent-train.exe")
	carried := request
	carried.Device, carried.OutputDirectory = nil, filepath.Join(root, "carried")
	if _, err := ExecuteInCommand(t.Context(), command, filepath.Join(root, "store"), carried); err == nil || !strings.Contains(err.Error(), "start "+command) {
		t.Fatalf("a request the flags carry, with no command at %s = %v; want it to reach the start of the command", command, err)
	}
	for name, change := range map[string]func(*Request){
		"held to the host":               func(request *Request) { request.Host = true },
		"with a device trainer":          func(request *Request) { request.Device = hostWeightedStep },
		"with a reference":               func(request *Request) { request.ReferenceDirectory = request.ModelDirectory },
		"with a sequence limit":          func(request *Request) { request.MaximumSequence = 8 },
		"with an objective scale":        func(request *Request) { request.ObjectiveScale = 0.1 },
		"with a frozen lexicon":          func(request *Request) { request.FreezeLexical = true },
		"sampling its own answers":       func(request *Request) { request.Rewards = &RewardSampling{} },
		"with a preference mode":         func(request *Request) { request.PreferenceMode = trainingprogram.PreferenceSimPO },
		"with a preference margin":       func(request *Request) { request.PreferenceMargin = 0.5 },
		"observing preference steps":     func(request *Request) { request.ObserveDPO = func(trainingprogram.DPOObservation) {} },
		"with a bound on projected time": func(request *Request) { request.MaxProjectedWall = 1 },
		"with an audio specification":    func(request *Request) { request.Audio = &AudioTrainingSpec{} },
		"recording no session":           func(request *Request) { request.Observations = nil },
	} {
		refused := carried
		change(&refused)
		if _, err := ExecuteInCommand(t.Context(), command, filepath.Join(root, "store"), refused); err == nil ||
			!strings.Contains(err.Error(), "this request asks for something else") || strings.Contains(err.Error(), command) {
			t.Errorf("a request %s = %v; want it refused as one the training command's flags cannot carry, before the command is started", name, err)
		}
	}
	ended, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := ExecuteInCommand(ended, command, filepath.Join(root, "store"), carried); !errors.Is(err, context.Canceled) || strings.Contains(err.Error(), command) {
		t.Fatalf("a request under a context that has ended = %v; want the context's error and nothing started", err)
	}
}
