package trainingworkflow

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/trainingprogram"
)

// TestPilotRefusesChangedInputsAndForeignCheckpoints holds a run of the
// instruction pilot to the inputs and the checkpoints its acceptance fixed,
// on the fixture model and the device, where its arms train: the host has no
// reward trainer, so no arm has a checkpoint there. The run trains both arms
// to every checkpoint and answers its comparisons.
//
// Run again over its own directory under its unchanged catalog, the prompts
// it trains on and its arms' checkpoints are kept as they are, not written
// again. Run over that directory by other code, or under another acceptance
// that divides the same groups, the first checkpoint found is refused as not
// the arm's, by what it is not; under an acceptance that divides them
// otherwise, the prompts the directory holds are another run's and it is
// refused for them. Nothing there is written over by either.
//
// What makes a found checkpoint the arm's is held one condition at a time on
// the checkpoints that run published: it is of the base, sampled by the
// arm's protocol, specification, seed, draws, code and control, continues
// the checkpoint before it, and stands after its count of groups; each
// changed alone refuses it.
//
// A session continues a checkpoint only as it was sampled: resumed under
// another seed it is refused, and resumed as it was it publishes the weights
// the uninterrupted run reached. A session that fails after its draws are
// kept and before its checkpoint is published leaves nothing counted twice:
// started again it reaches the uninterrupted run's weights and cursor.
//
// And a catalog replaced under the same names, with other prompts, is refused
// before anything is written, in a fresh directory and in the run's own.
func TestPilotRefusesChangedInputsAndForeignCheckpoints(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	store, model := pilotFixture(t)
	pilot := authorityPilot(model)
	outcome, err := FixInstructionPilot(ctx, store, pilot)
	if err != nil || outcome.Refused != nil {
		t.Fatal(err, outcome.Refused)
	}
	acceptance := outcome.Acceptance
	base, err := identifyModel(model)
	if err != nil {
		t.Fatal(err)
	}
	specification, err := armSpecification(acceptance, pilotArms[0])
	if err != nil {
		t.Fatal(err)
	}
	sampling := RewardSampling{Protocol: acceptance.Protocol, Training: specification, Seed: int64(acceptance.Seed), Draws: outcome.Training.Draws, Code: pilot.Code}

	// The run trains both arms and answers its comparisons.
	const foreign = "holds a checkpoint that is not the reward arm's"
	run := PilotRun{Acceptance: acceptance.ID, Model: model, Out: filepath.Join(t.TempDir(), "pilot"), Code: pilot.Code, Device: residentTrainer}
	if _, err := RunInstructionPilot(ctx, store, run); err != nil {
		t.Fatal(err)
	}
	arm := filepath.Join(run.Out, pilotArms[0])
	at := func(groups int) string { return filepath.Join(arm, fmt.Sprintf("groups-%03d", groups)) }
	with := func(change func(*RewardSampling)) RewardSampling {
		other := sampling
		change(&other)
		return other
	}
	// held is every file under the run's directory, by its bytes: the
	// prompts it trains on and its arms' checkpoints.
	held := func() map[string][sha256.Size]byte {
		t.Helper()
		files := map[string][sha256.Size]byte{}
		err := filepath.WalkDir(run.Out, func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() {
				return err
			}
			data, err := os.ReadFile(path)
			files[path] = sha256.Sum256(data)
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
		return files
	}
	published := held()
	if len(published) <= len(pilotArms)*len(outcome.Training.Checkpoints) {
		t.Fatalf("the run left %d files for its prompts and %d checkpoints an arm of %d arms", len(published), len(outcome.Training.Checkpoints), len(pilotArms))
	}
	if _, err := RunInstructionPilot(ctx, store, run); err != nil || !maps.Equal(held(), published) {
		t.Fatalf("run again as it was: the run ended with %v and left its directory as the first run did %t; want it to end well with nothing written again", err, maps.Equal(held(), published))
	}
	refused := func(label string, other PilotRun, want string) {
		t.Helper()
		_, err := RunInstructionPilot(ctx, store, other)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("%s: the run ended with %v, want %q", label, err, want)
		}
		if !maps.Equal(held(), published) {
			t.Fatalf("%s: what the first run left in its directory was written again", label)
		}
	}
	changed := run
	changed.Code = "other code"
	refused("run by other code", changed, foreign)
	// An acceptance of another gain divides the same groups and so trains on
	// the same prompts; one of another seed divides them otherwise.
	for _, other := range []struct {
		label, want string
		change      func(*InstructionPilot)
	}{
		{"run under an acceptance of another gain", foreign, func(pilot *InstructionPilot) { pilot.Gain = 0.61 }},
		{"run under an acceptance of another seed", "holds the development prompts of another run", func(pilot *InstructionPilot) { pilot.Seed++ }},
	} {
		moved := pilot
		other.change(&moved)
		another, err := FixInstructionPilot(ctx, store, moved)
		if err != nil || another.Refused != nil || another.Acceptance.ID == acceptance.ID {
			t.Fatalf("%s: the acceptance is %s (%v, %v), want another acceptance", other.label, another.Acceptance.ID, err, another.Refused)
		}
		changed = run
		changed.Acceptance = another.Acceptance.ID
		refused(other.label, changed, other.want)
	}

	// What makes a found checkpoint the arm's, one condition at a time.
	load := func(groups int) trainingprogram.Checkpoint {
		t.Helper()
		checkpoint, err := trainingprogram.LoadCheckpoint(at(groups))
		if err != nil {
			t.Fatal(err)
		}
		return checkpoint
	}
	first, second, fourth := load(1), load(2), load(4)
	var none trainingprogram.Checkpoint
	if err := errors.Join(sampling.published(first, base, none, 1), sampling.published(second, base, first, 2), sampling.published(fourth, base, second, 4)); err != nil {
		t.Fatalf("the run's own checkpoints are not held to be its arm's: %v", err)
	}
	otherArm, err := armSpecification(acceptance, pilotArms[1])
	if err != nil {
		t.Fatal(err)
	}
	for label, verdict := range map[string]error{
		"another base":                  sampling.published(second, specification, first, 2),
		"no checkpoint before it":       sampling.published(second, base, none, 2),
		"another checkpoint before it":  sampling.published(fourth, base, first, 4),
		"a checkpoint before the first": sampling.published(first, base, second, 1),
		"another count of groups":       sampling.published(second, base, first, 4),
		"another seed":                  with(func(other *RewardSampling) { other.Seed++ }).published(second, base, first, 2),
		"other draws":                   with(func(other *RewardSampling) { other.Draws++ }).published(second, base, first, 2),
		"other code":                    with(func(other *RewardSampling) { other.Code += " changed" }).published(second, base, first, 2),
		"the control":                   with(func(other *RewardSampling) { other.Control = true }).published(second, base, first, 2),
		"another output limit":          with(func(other *RewardSampling) { other.Protocol.MaxTokens++ }).published(second, base, first, 2),
		"another arm's specification":   with(func(other *RewardSampling) { other.Training = otherArm }).published(second, base, first, 2),
	} {
		if verdict == nil {
			t.Fatalf("a checkpoint of %s is held to be the arm's", label)
		}
	}

	// A session continues a checkpoint only as it was sampled, and an
	// interrupted one is started again without counting anything twice.
	asked := filepath.Join(run.Out, "development.json")
	suite, err := os.ReadFile(asked)
	if err != nil {
		t.Fatal(err)
	}
	recipeID, err := bootstrapRewardRecipe(ctx, store, model, suite)
	if err != nil {
		t.Fatal(err)
	}
	continued := func(from int, out string, steps int, rewards RewardSampling) (Result, error) {
		return Execute(ctx, Request{
			Repository: store, Observations: store, Recipe: recipeID, DatasetPath: asked, OutputDirectory: out,
			ResumeDirectory: at(from), Steps: steps, Device: residentTrainer, Rewards: &rewards,
		})
	}
	scratch := t.TempDir()
	if _, err := continued(1, filepath.Join(scratch, "reseeded"), 1, with(func(other *RewardSampling) { other.Seed++ })); err == nil || !strings.Contains(err.Error(), "resume authority differs") {
		t.Fatalf("a session resumed under another seed ended with %v, want its resume refused", err)
	}
	again, err := continued(1, filepath.Join(scratch, "again"), 1, sampling)
	if err != nil || again.Checkpoint.Weights != second.Weights || again.Checkpoint.Stream.Position != second.Stream.Position {
		t.Fatalf("resumed as it was sampled, a session reached weights %s at %d (%v), want the uninterrupted run's %s at %d", again.Checkpoint.Weights, again.Checkpoint.Stream.Position, err, second.Weights, second.Stream.Position)
	}
	// The output is a file, so the session draws, steps and then cannot publish.
	blocked := filepath.Join(scratch, "blocked")
	if err := os.WriteFile(blocked, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := continued(2, blocked, 2, sampling); err == nil {
		t.Fatal("a session whose output is a file published a checkpoint")
	}
	retried, err := continued(2, filepath.Join(scratch, "retried"), 2, sampling)
	if err != nil || retried.Checkpoint.Weights != fourth.Weights || retried.Checkpoint.Stream.Position != fourth.Stream.Position {
		t.Fatalf("started again after it failed to publish, a session reached weights %s at %d (%v), want the uninterrupted run's %s at %d", retried.Checkpoint.Weights, retried.Checkpoint.Stream.Position, err, fourth.Weights, fourth.Stream.Position)
	}

	// A catalog replaced under its names is refused before anything is written.
	catalogPilotCases(t, store, true)
	fresh := run
	fresh.Out = filepath.Join(t.TempDir(), "pilot")
	if _, err := RunInstructionPilot(ctx, store, fresh); err == nil || !strings.Contains(err.Error(), "divides the instruction cases of dataset") {
		t.Fatalf("under a catalog whose prompts were replaced the run ended with %v, want it refused for its cases", err)
	}
	if _, err := os.Stat(fresh.Out); !os.IsNotExist(err) {
		t.Fatalf("the refused run wrote %s (%v)", fresh.Out, err)
	}
	refused("run over its own directory under the replaced catalog", run, "divides the instruction cases of dataset")
}
