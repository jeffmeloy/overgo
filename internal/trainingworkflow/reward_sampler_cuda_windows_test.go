package trainingworkflow

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/densecausal"
	"overgo/internal/densecausaltrain"
	"overgo/internal/evaluation"
	"overgo/internal/gguf"
	"overgo/internal/tokenizer"
)

// TestDeviceSamplerDrawsWhatTheHostSamplerDraws holds the sampler a reward
// session on the device draws a GGUF model's answers by to the model in
// training.
//
// The weights are opened in the inference runtime on the device, and what it
// draws is what the host sampler draws from the model in training: the same
// ids from the same seeds, each at the host's probability to within
// servedAgreement; and moved and released, the weights opened next are the
// moved ones (heldToHostSampler).
//
// A session that trains on the device, through the resident trainer's one
// weighted step, samples each group there from the weights as they stand: the
// runtime's model and the trainer's take the device in turn within one
// process, group after group, and the session stopped and resumed ends at the
// bytes the uninterrupted one ends at (rewardSessionOverGGUF).
func TestDeviceSamplerDrawsWhatTheHostSamplerDraws(t *testing.T) {
	cudatest.Require(t)
	request := rewardGGUFFixture(t)
	heldToHostSampler(t, request, false, servedAgreement)
	request.Device = residentTrainer
	rewardSessionOverGGUF(t, request)

	// The control of a reward experiment, on the device: a pass over the
	// prompts steps on the groups with an answer that passed every rule and
	// passes the rest, the prompt no answer can pass among them.
	control := rewardGGUFFixture(t)
	passed := *control.Rewards
	passed.Control = true
	control.Rewards, control.Device = &passed, residentTrainer
	const prompts = 5
	control.Steps, control.OutputDirectory = prompts, filepath.Join(filepath.Dir(control.DatasetPath), "control")
	result, err := Execute(t.Context(), control)
	if trace := result.Rewards; err != nil || trace.Sampled != prompts || trace.Updated+trace.Skipped != prompts || trace.Skipped == 0 || result.Checkpoint.Optimizer.Step != trace.Updated {
		t.Fatalf("the control session = %+v at step %d (%v), want %d groups sampled, those with a passed answer stepped and the rest passed", trace, result.Checkpoint.Optimizer.Step, err, prompts)
	}
}

// rewardSessionOverGGUF runs a reward session over the fixture model, by the
// device trainer the request carries, and holds it to sampling by the
// inference runtime from the weights as they stand: its groups are sampled
// and some stepped; the text of an answer it recorded is its own ids' text,
// with none of the prompt's, which the runtime's returned text begins with;
// and stopped at its checkpoint and resumed in a store that held none of its
// draws, it ends at the bytes it ends at uninterrupted, so each group was
// drawn from the weights as they stood: the resumed session opens the
// checkpoint's weights where the uninterrupted one opened its own after a
// step. It leaves nothing it wrote to open its weights.
func rewardSessionOverGGUF(t *testing.T, request Request) {
	t.Helper()
	ctx := t.Context()
	root := filepath.Dir(request.ModelDirectory)
	tokens, err := loadTrainableTokenizer(request.ModelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	run := func(request Request, name, resume string, steps int) Result {
		t.Helper()
		request.Steps, request.OutputDirectory = steps, filepath.Join(filepath.Dir(request.DatasetPath), name)
		if resume != "" {
			request.ModelDirectory, request.ResumeDirectory = "", filepath.Join(filepath.Dir(request.DatasetPath), resume)
		}
		result, err := Execute(ctx, request)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	const groups = 4
	whole := run(request, "whole", "", groups)
	if trace := whole.Rewards; trace.Sampled != groups || trace.Updated == 0 || trace.Updated+trace.Skipped != groups || trace.SampledTokens == 0 || whole.Checkpoint.Optimizer.Step != trace.Updated ||
		trace.Shift == 0 || math.IsNaN(trace.Shift) || math.IsInf(trace.Shift, 0) {
		t.Fatalf("the session = %+v at step %d, want %d groups sampled, some stepped, and their answers moved under the weights after", trace, whole.Checkpoint.Optimizer.Step, groups)
	}
	file, err := gguf.Open(request.ModelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	vocabulary, err := tokenizer.Load(file)
	if closeErr := file.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	base, err := identifyModel(request.ModelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	suite, err := evaluation.ParseInstructionPrompts([]byte(rewardPrompts))
	if err != nil {
		t.Fatal(err)
	}
	name := suite.Names()[0]
	asked, limit, _ := suite.Asked(name)
	first := evaluation.SampledGroup{
		Group: name, Code: request.Rewards.Code, Weights: base, Tokenizer: base, Template: "none",
		Ends: []tokenizer.TokenID{tokenizer.TokenID(tokens.end)}, Temperature: 1, Seed: request.Rewards.Seed, MaxTokens: limit,
	}
	encoded, err := tokens.encode(asked)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range encoded {
		first.Prompt = append(first.Prompt, tokenizer.TokenID(id))
	}
	answers, err := evaluation.SampleGroup(ctx, request.Observations, neverSampled{}, first, request.Rewards.Draws)
	if err != nil {
		t.Fatalf("the session's first group is not in the store under the base's weights: %v", err)
	}
	for _, answer := range answers {
		var text strings.Builder
		for _, id := range answer.Completion {
			piece, err := vocabulary.DecodePiece(id, false)
			if err != nil {
				t.Fatal(err)
			}
			text.WriteString(piece)
		}
		if answer.Text != text.String() || len(answer.Completion) == 0 {
			t.Fatalf("a recorded answer's text is %q, want its ids' own %q", answer.Text, text.String())
		}
	}
	other := rewardGGUFFixture(t)
	other.Device = request.Device
	if stopped := run(other, "stopped", "", groups/2); stopped.Rewards.Updated == 0 {
		t.Fatalf("the stopped session = %+v, want a step before its checkpoint", stopped.Rewards)
	}
	run(other, "resumed", "stopped", groups-groups/2)
	trained := func(request Request, name string) map[string][]float32 {
		t.Helper()
		model, err := loadTrainableModel(filepath.Join(filepath.Dir(request.DatasetPath), name))
		if err != nil {
			t.Fatal(err)
		}
		return model.Weights
	}
	resumed := trained(other, "resumed")
	for name, want := range trained(request, "whole") {
		if !slices.Equal(resumed[name], want) {
			t.Fatalf("the session stopped and resumed ends with other bytes in %s than the uninterrupted one", name)
		}
	}
	for _, directory := range []string{root, filepath.Dir(other.DatasetPath)} {
		if left, err := filepath.Glob(filepath.Join(directory, "reward-weights-*")); err != nil || len(left) != 0 {
			t.Fatalf("a finished session left %v behind (%v)", left, err)
		}
	}
}

// TestBootstrappedRewardRecipeRunsARewardSessionOnTheDevice holds the recipe
// a training command publishes for a reward session over a GGUF, as a pilot
// publishes it, to being one a reward session runs under on the device: a
// session over that model and the suite of prompts under it samples its
// groups in the device's runtime and steps by the device's trainer, as one
// under an authority a test assembles does. What the recipe is, published
// once and again, is held with it (bootstrappedRewardSession).
func TestBootstrappedRewardRecipeRunsARewardSessionOnTheDevice(t *testing.T) {
	cudatest.Require(t)
	request := bootstrappedRewardSession(t, writeTrainingGGUF(t))
	request.Device = residentTrainer
	heldToARewardSession(t, request)
}

// residentTrainer is the device trainer a training command gives a session:
// the resident trainer's one weighted step, on the first device.
func residentTrainer(model *densecausal.Model, groups [][]densecausal.WeightedSequence, learningRate, momentum float64, freezeLexical bool, resume *densecausal.TrainState, observe densecausal.TrainObserver) ([]float64, int, densecausal.TrainState, error) {
	worker, err := device.New(0)
	if err != nil {
		return nil, 0, densecausal.TrainState{}, err
	}
	defer worker.Close()
	result, err := densecausaltrain.TrainDeviceResidentWeighted(model, worker, groups, learningRate, momentum, densecausaltrain.Options{FrozenLexical: freezeLexical, Resume: resume, Observe: observe})
	return result.Losses, result.Skipped, result.State, err
}

// TestPilotRunTrainsFreezesAndJudgesItsArms holds a run of the instruction
// pilot to its acceptance, on the fixture model and the device.
//
// Both arms are trained from the base to each checkpoint the training
// specification names, the reward arm and its control; each checkpoint is
// read on the selection groups, and the one frozen for an arm passes as many
// of them as any of the arm's, the earliest of equals. Each comparison the
// acceptance named is charged under it and answered once, the promotion part
// left with one confirmation and then none, and the arm is adopted exactly
// when the adoption rule says so of those answers. Run again over the same
// directory, nothing is trained again, the same checkpoints are frozen, and
// each comparison returns the verdict it recorded. And with the reward arm's
// first checkpoint put where the control's belongs, the run is refused: a
// checkpoint is credited to the arm it was sampled as.
func TestPilotRunTrainsFreezesAndJudgesItsArms(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	store, model := pilotFixture(t)
	outcome, err := FixInstructionPilot(ctx, store, InstructionPilot{
		Model: model, Gain: 0.6, Weights: [len(evaluation.PoolParts)]uint64{2, 1, 4, 1},
		Template: rewardPlain, Window: 24, MaxTokens: 6, Seed: 11, Code: "pilot fixture code",
		Training: InstructionTraining{Draws: 4, Passes: 1, DeviceBytes: 1 << 30},
	})
	if err != nil || outcome.Refused != nil {
		t.Fatal(err, outcome.Refused)
	}
	run := PilotRun{Acceptance: outcome.Acceptance.ID, Model: model, Out: filepath.Join(filepath.Dir(model), "pilot"), Code: "pilot fixture code", Device: residentTrainer}
	result, err := RunInstructionPilot(ctx, store, run)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Arms) != len(pilotArms) || result.Selection != outcome.Parts[1] || result.BaseSelected > result.Selection {
		t.Fatalf("the run = %d arms, the base passing %d of %d selection groups; want both arms and the selection part's %d", len(result.Arms), result.BaseSelected, result.Selection, outcome.Parts[1])
	}
	for index, arm := range result.Arms {
		if arm.Name != pilotArms[index] || len(arm.Checkpoints) != len(outcome.Training.Checkpoints) {
			t.Fatalf("arm %d is %s with %d checkpoints, want %s with %d", index, arm.Name, len(arm.Checkpoints), pilotArms[index], len(outcome.Training.Checkpoints))
		}
		sampled := 0
		for at, checkpoint := range arm.Checkpoints {
			sampled += checkpoint.Trace.Sampled
			if checkpoint.Groups != outcome.Training.Checkpoints[at] || sampled != checkpoint.Groups || checkpoint.Selected > arm.Checkpoints[arm.Frozen].Selected || at < arm.Frozen && checkpoint.Selected == arm.Checkpoints[arm.Frozen].Selected {
				t.Fatalf("the %s arm's checkpoint %d = %+v after %d groups sampled, frozen at %d: want the specification's count reached and the first of the best frozen", arm.Name, at, checkpoint, sampled, arm.Frozen)
			}
		}
	}
	if len(result.Comparisons) != len(pilotComparisons) {
		t.Fatalf("%d comparisons were answered, want the acceptance's %d", len(result.Comparisons), len(pilotComparisons))
	}
	for index, answered := range result.Comparisons {
		if answered.Name != pilotComparisons[index].Name || !answered.Confirmation.Charge.Valid() || answered.Confirmation.Acceptance != run.Acceptance || answered.Left != uint64(len(pilotComparisons)-index-1) {
			t.Fatalf("comparison %d = %+v, want %s charged under the acceptance and %d left", index, answered, pilotComparisons[index].Name, len(pilotComparisons)-index-1)
		}
	}
	if adopted, reason := pilotAdopted(outcome.Acceptance, result.Comparisons); adopted != result.Adopted || reason != result.Reason {
		t.Fatalf("the run adopted %t (%q) and the rule says %t (%q)", result.Adopted, result.Reason, adopted, reason)
	}
	again, err := RunInstructionPilot(ctx, store, run)
	if err != nil {
		t.Fatal(err)
	}
	for index, arm := range again.Arms {
		for _, checkpoint := range arm.Checkpoints {
			if checkpoint.Trace.Sampled != 0 {
				t.Fatalf("run again, the %s arm sampled %d groups to reach %d, want its published checkpoint kept", arm.Name, checkpoint.Trace.Sampled, checkpoint.Groups)
			}
		}
		if arm.Frozen != result.Arms[index].Frozen {
			t.Fatalf("run again, the %s arm froze checkpoint %d, want %d", arm.Name, arm.Frozen, result.Arms[index].Frozen)
		}
	}
	for index, answered := range again.Comparisons {
		if answered.Confirmation.ID != result.Comparisons[index].Confirmation.ID || again.Adopted != result.Adopted {
			t.Fatalf("run again, comparison %s is %s, want the recorded %s", answered.Name, answered.Confirmation.ID, result.Comparisons[index].Confirmation.ID)
		}
	}
	// The reward arm's first checkpoint where the control's belongs is another arm's.
	const first = "groups-001"
	control := filepath.Join(run.Out, pilotArms[1], first)
	if err := errors.Join(os.Rename(control, control+".aside"), os.CopyFS(control, os.DirFS(filepath.Join(run.Out, pilotArms[0], first)))); err != nil {
		t.Fatal(err)
	}
	if _, err := RunInstructionPilot(ctx, store, run); err == nil || !strings.Contains(err.Error(), "holds a checkpoint that is not the "+pilotArms[1]+" arm's") {
		t.Fatalf("with the reward arm's checkpoint in the control's place the run ended with %v, want it refused as not the control's", err)
	}
}
