package trainingworkflow

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/evaluation"
)

// authorityPilot is the pilot whose acceptance the tests of a pilot run's
// inputs fix over a base, read on the host.
func authorityPilot(model string) InstructionPilot {
	return InstructionPilot{
		Model: model, Host: true, Gain: 0.6, Weights: [len(evaluation.PoolParts)]uint64{2, 1, 4, 1},
		Template: rewardPlain, Window: 24, MaxTokens: 6, Seed: 11, Code: "pilot fixture code",
		Training: InstructionTraining{Draws: 4, Passes: 1, DeviceBytes: 1 << 30},
	}
}

// TestPilotRefusesAProtocolItCannotHonourAndAReplacedCatalog holds a run of
// the instruction pilot to what it refuses before it draws an answer or
// trains an arm, on the fixture model and the host. What it refuses once its
// arms have checkpoints needs those trained, which the device does
// (TestPilotRefusesChangedInputsAndForeignCheckpoints).
//
// A protocol the sampler does not honour is refused before a draw: thinking
// asked for, or another stop.
//
// And a catalog replaced under the same names, with other prompts, is refused
// before anything is written, in a fresh directory.
func TestPilotRefusesAProtocolItCannotHonourAndAReplacedCatalog(t *testing.T) {
	ctx := t.Context()
	store, model := pilotFixture(t)
	pilot := authorityPilot(model)
	outcome, err := FixInstructionPilot(ctx, store, pilot)
	if err != nil || outcome.Refused != nil {
		t.Fatal(err, outcome.Refused)
	}
	acceptance := outcome.Acceptance
	specification, err := armSpecification(acceptance, pilotArms[0])
	if err != nil {
		t.Fatal(err)
	}
	sampling := RewardSampling{Protocol: acceptance.Protocol, Training: specification, Seed: int64(acceptance.Seed), Draws: outcome.Training.Draws, Code: pilot.Code}

	// A protocol the sampler does not honour is refused before a draw.
	if _, err := sampling.tokenizer(model, store, rewardDraws); err != nil {
		t.Fatalf("the acceptance's own protocol is refused: %v", err)
	}
	for _, changed := range []func(*evaluation.InstructionProtocol){
		func(protocol *evaluation.InstructionProtocol) { protocol.Thinking = "on" },
		func(protocol *evaluation.InstructionProtocol) { protocol.Stop = "a blank line" },
	} {
		other := sampling
		changed(&other.Protocol)
		if _, err := other.tokenizer(model, store, rewardDraws); err == nil {
			t.Fatalf("a session under thinking %q and stop %q was given a tokenizer to draw with, want it refused", other.Protocol.Thinking, other.Protocol.Stop)
		}
	}

	// A catalog replaced under its names is refused before anything is written.
	catalogPilotCases(t, store, true)
	fresh := PilotRun{Acceptance: acceptance.ID, Model: model, Out: filepath.Join(t.TempDir(), "pilot"), Code: pilot.Code}
	if _, err := RunInstructionPilot(ctx, store, fresh); err == nil || !strings.Contains(err.Error(), "divides the instruction cases of dataset") {
		t.Fatalf("under a catalog whose prompts were replaced the run ended with %v, want it refused for its cases", err)
	}
	if _, err := os.Stat(fresh.Out); !os.IsNotExist(err) {
		t.Fatalf("the refused run wrote %s (%v)", fresh.Out, err)
	}
}
