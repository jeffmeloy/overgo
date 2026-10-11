package trainingworkflow

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// rewardPrompts is a suite of prompts over the fixture model's letters, each
// asking for a rule or two of an answer: one no answer can pass, so its
// answers always stand together.
const rewardPrompts = `{"kind":"instruction-rules","schema":"overgo/reward-fixture/v1","source":"reward fixture","cases":[
{"name":"ends-in-c","prompt":"ab","max_tokens":4,"rules":[{"name":"contains:c","kind":"contains-all","values":["c"]}]},
{"name":"without-d","prompt":"ba","max_tokens":4,"rules":[{"name":"excludes:d","kind":"excludes-all","values":["d"]}]},
{"name":"c-without-d","prompt":"aab","max_tokens":4,"rules":[{"name":"contains:c","kind":"contains-all","values":["c"]},{"name":"excludes:d","kind":"excludes-all","values":["d"]}]},
{"name":"b-and-c","prompt":"a","max_tokens":4,"rules":[{"name":"contains:b","kind":"contains-all","values":["b"]},{"name":"contains:c","kind":"contains-all","values":["c"]}]},
{"name":"impossible","prompt":"bb","max_tokens":4,"rules":[{"name":"contains:zz","kind":"contains-all","values":["zz"]}]}
]}`

// rewardPromptEvaluators names the scorer of a dataset that is a suite of
// instruction prompts, and says whether it is one.
func rewardPromptEvaluators(raw []byte) ([]artifact.ID, bool) {
	prompts, err := evaluation.ParseInstructionPrompts(raw)
	if err != nil {
		return nil, false
	}
	evaluator, err := prompts.Evaluator()
	return []artifact.ID{evaluator}, err == nil
}

// rewardFixture writes the tiny model, which declares token zero its end,
// and the suite of prompts, and makes the authority of a reward session over
// them. Two fixtures are the same model and prompts in two stores.
func rewardFixture(t *testing.T) (string, Request) {
	t.Helper()
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 17,
	})
	root := t.TempDir()
	model, prompts := filepath.Join(root, "model"), filepath.Join(root, "prompts.json")
	writeModel(t, model, weights, shapes)
	// The end token is declared in config.json, which a checkpoint carries with it.
	config, err := os.ReadFile(filepath.Join(model, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	for path, content := range map[string]string{filepath.Join(model, "config.json"): strings.Replace(string(config), "{", `{"eos_token_id":0,`, 1), prompts: rewardPrompts} {
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	store, recipeID := trainingAuthority(t, model, "", prompts, trainingprogram.ObjectiveGRPO)
	return root, Request{
		Repository: store, Observations: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: prompts, Host: true,
		Rewards: &RewardSampling{
			Protocol: evaluation.InstructionProtocol{Scorer: evaluation.InstructionScorer, Template: "none", Thinking: "off", Sampling: rewardSampling, Stop: "end token", Window: 16, MaxTokens: 4},
			Training: testutil.ArtifactID(t, artifact.KindProfile, "reward fixture training"), Seed: 5, Draws: 4, Code: "reward fixture code",
		},
	}
}

// neverSampled is a generator a test hands to a group whose draws the store
// is to hold already: asked to sample, it fails.
type neverSampled struct{}

func (neverSampled) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("a draw the store was to hold was sampled")
}

// TestRewardRunSamplesAndScoresItsOwnGroups holds a reward session that is
// given prompts and no prepared answers to the loop it closes.
//
// For each prompt its stream gives it, the session draws its answers from the
// weights as they stand, scores them by the prompts' rules, and takes at most
// one step: over two passes of five prompts it samples ten groups, steps on
// some, and passes the prompt no answer can pass both times, the second after
// the weights have moved. What it did is counted: groups sampled, stepped and
// passed, tokens sampled and trained, answers that ran to the limit, and time
// in each stage. Every group it sampled is in the store under the identity of
// the weights it was drawn from, the base's after so many steps, with each of
// its draws: asked for again there, none is sampled; and a prompt visited
// twice between steps is two groups, under two identities.
//
// Interrupted and run again, a session reuses what it recorded. A session
// that drew only two answers to its first prompt and was lost, and one that
// ran three prompts and was lost before its checkpoint, leave their draws in
// the store; the whole session run after them ends at the bytes an
// uninterrupted session ends at in a store that held nothing, with as many
// steps. And a session stopped at its checkpoint and resumed ends where one
// session does.
//
// A session asked for a cut or sharpened distribution, for fewer than two
// draws, for another scorer, or given no store to keep its draws in is
// refused. Off the host, each group goes to the device trainer as one step.
func TestRewardRunSamplesAndScoresItsOwnGroups(t *testing.T) {
	ctx := t.Context()
	root, request := rewardFixture(t)
	run := func(request Request, name string, steps int) Result {
		t.Helper()
		request.Steps, request.OutputDirectory = steps, filepath.Join(filepath.Dir(request.ModelDirectory), name)
		result, err := Execute(ctx, request)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	trained := func(request Request, name string) map[string][]float32 {
		t.Helper()
		model, err := loadTrainableModel(filepath.Join(filepath.Dir(request.ModelDirectory), name))
		if err != nil {
			t.Fatal(err)
		}
		return model.Weights
	}
	const prompts, passes = 5, 2
	whole := run(request, "whole", prompts*passes)
	trace := whole.Rewards
	if trace.Sampled != prompts*passes || trace.Updated+trace.Skipped != trace.Sampled || trace.Updated == 0 || trace.Skipped < passes || len(whole.GRPO) != trace.Updated ||
		whole.Checkpoint.Optimizer.Step != trace.Updated || whole.StreamPosition != prompts*passes || whole.Skipped != uint64(trace.Skipped) || trace.Stopped != "" {
		t.Fatalf("the session = %+v with %d observations, step %d and position %d; want ten groups sampled, some stepped, the impossible prompt passed twice, and the checkpoint at as many steps", trace, len(whole.GRPO), whole.Checkpoint.Optimizer.Step, whole.StreamPosition)
	}
	if trace.SampledTokens == 0 || trace.TrainedTokens == 0 || trace.TrainedTokens > trace.SampledTokens || trace.Generation <= 0 || trace.Scoring < 0 || trace.Update < 0 {
		t.Fatalf("the session counts %d tokens sampled and %d trained, and %v, %v and %v sampling, scoring and stepping", trace.SampledTokens, trace.TrainedTokens, trace.Generation, trace.Scoring, trace.Update)
	}
	for _, observation := range whole.GRPO {
		if !observation.Valid() || observation.GroupSize != uint64(request.Rewards.Draws) {
			t.Fatalf("a stepped group is observed as %+v", observation)
		}
	}

	// Every group is in the store under the weights it was drawn from.
	base, err := identifyModel(request.ModelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	suite, err := evaluation.ParseInstructionPrompts([]byte(rewardPrompts))
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := loadTrainableTokenizer(request.ModelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	held := map[string]int{}
	for _, name := range suite.Names() {
		prompt, limit, _ := suite.Asked(name)
		encoded, err := tokens.encode(prompt)
		if err != nil {
			t.Fatal(err)
		}
		for updates := 0; updates <= trace.Updated; updates++ {
			weights, err := evaluation.UpdatedWeights(base, request.Rewards.Training, uint64(updates))
			if err != nil {
				t.Fatal(err)
			}
			group := evaluation.SampledGroup{
				Group: name, Code: request.Rewards.Code, Weights: weights, Tokenizer: base, Template: "none",
				Ends: []tokenizer.TokenID{0}, Temperature: 1, Seed: request.Rewards.Seed, MaxTokens: limit,
			}
			for _, id := range encoded {
				group.Prompt = append(group.Prompt, tokenizer.TokenID(id))
			}
			answers, err := evaluation.SampleGroup(ctx, request.Observations, neverSampled{}, group, request.Rewards.Draws)
			if err != nil {
				continue
			}
			held[name]++
			for _, answer := range answers {
				if len(answer.Completion) == 0 || len(answer.Completion) != len(answer.LogProbabilities) || len(answer.Completion) > limit {
					t.Fatalf("a recorded draw of %s is %+v", name, answer)
				}
			}
		}
	}
	sampled := 0
	for _, groups := range held {
		sampled += groups
	}
	if sampled != trace.Sampled || held["impossible"] != passes {
		t.Fatalf("the store holds %d groups by prompt %v, want the %d sampled, the impossible prompt's under %d identities", sampled, held, trace.Sampled, passes)
	}

	// Interrupted twice and run again, against the uninterrupted session's bytes.
	_, again := rewardFixture(t)
	few := again
	rewards := *again.Rewards
	rewards.Draws = 2
	few.Rewards = &rewards
	run(few, "lost-in-a-group", 1)
	run(again, "lost-before-its-checkpoint", 3)
	rerun := run(again, "whole", prompts*passes)
	for name, want := range trained(request, "whole") {
		if !slices.Equal(trained(again, "whole")[name], want) {
			t.Fatalf("the session run again after two lost ones ends with other bytes in %s than an uninterrupted one", name)
		}
	}
	if rerun.Rewards.Updated != trace.Updated || rerun.Rewards.Skipped != trace.Skipped || rerun.Checkpoint.Optimizer.Step != whole.Checkpoint.Optimizer.Step || rerun.Rewards.SampledTokens != trace.SampledTokens {
		t.Fatalf("the session run again = %+v at step %d, want the uninterrupted one's %+v at step %d", rerun.Rewards, rerun.Checkpoint.Optimizer.Step, trace, whole.Checkpoint.Optimizer.Step)
	}
	resumeRoot, resumed := rewardFixture(t)
	verifyWorkflowResume(t, resumeRoot, resumed)

	// What a session is refused for.
	for name, change := range map[string]func(*Request, *RewardSampling){
		"a cut distribution":     func(_ *Request, rewards *RewardSampling) { rewards.Protocol.Sampling = "top-p 0.9" },
		"one draw":               func(_ *Request, rewards *RewardSampling) { rewards.Draws = 1 },
		"another scorer":         func(_ *Request, rewards *RewardSampling) { rewards.Protocol.Scorer = "instruction-rules/strict/0" },
		"no store for its draws": func(request *Request, _ *RewardSampling) { request.Observations = nil },
	} {
		refused, sampling := request, *request.Rewards
		refused.Rewards = &sampling
		change(&refused, &sampling)
		refused.Steps, refused.OutputDirectory = 1, filepath.Join(root, "refused")
		if _, err := Execute(ctx, refused); err == nil || !strings.Contains(err.Error(), "a reward session draws") {
			t.Errorf("a session asked for %s = %v, want it refused", name, err)
		}
	}

	// Off the host each group is one step of the device trainer, here a stand-in.
	calls := 0
	device := request
	device.Host, device.Device = false, func(model *densecausal.Model, groups [][]densecausal.WeightedSequence, learningRate, momentum float64, _ bool, resume *densecausal.TrainState, _ densecausal.TrainObserver) ([]float64, int, densecausal.TrainState, error) {
		calls++
		update, err := model.NewDenseUpdate(learningRate, momentum, resume)
		if err != nil || len(groups) != 1 || len(groups[0]) != request.Rewards.Draws {
			return nil, 0, densecausal.TrainState{}, errors.Join(errors.New("the device was not handed one group of the session's draws"), err)
		}
		if !slices.ContainsFunc(groups[0], func(sequence densecausal.WeightedSequence) bool {
			return slices.ContainsFunc(sequence.Weights, func(weight float32) bool { return weight != 0 })
		}) {
			return nil, 1, update.Snapshot(), nil
		}
		update.Step()
		return []float64{0.5}, 0, update.Snapshot(), nil
	}
	if onDevice := run(device, "device", prompts); onDevice.Backend != "cuda" || calls != prompts || onDevice.Rewards.Sampled != prompts || onDevice.Rewards.Updated+onDevice.Rewards.Skipped != prompts || len(onDevice.GRPO) != onDevice.Rewards.Updated {
		t.Fatalf("the session off the host = backend %q, %d device calls, %+v, %d observations; want each of %d groups one call", onDevice.Backend, calls, onDevice.Rewards, len(onDevice.GRPO), prompts)
	}
}

// TestControlSessionStepsOnPassedAnswersAlone holds the control of a reward
// experiment to what it trains on. An answer that passed every rule is one
// sequence whose own tokens share a weight of one, and nothing of its prompt
// weighs; an answer that did not pass weighs nothing, so a group none of whose
// answers passed is all zeros, which the weighted step passes without an
// update. And a control session asked to train on the host is refused: it
// trains through the weighted step, which is the device's.
func TestControlSessionStepsOnPassedAnswersAlone(t *testing.T) {
	sequence := func(prompt, answer int) trainingdata.PreferenceSequence {
		made := trainingdata.PreferenceSequence{Tokens: make([]int, prompt+answer), Completion: make([]bool, prompt+answer)}
		for index := prompt; index < prompt+answer; index++ {
			made.Completion[index] = true
		}
		return made
	}
	group := trainingdata.RolloutGroup{Rollouts: []trainingdata.Rollout{{Sequence: sequence(2, 3)}, {Sequence: sequence(2, 1)}, {Sequence: sequence(1, 2)}}}
	weighed := passedSequences(group, []bool{true, false, true})
	third := float32(1) / 3
	if len(weighed) != 3 || !slices.Equal(weighed[0].Weights, []float32{0, third, third, third}) || !slices.Equal(weighed[1].Weights, []float32{0, 0}) || !slices.Equal(weighed[2].Weights, []float32{0.5, 0.5}) {
		t.Fatalf("the passed answers are weighed %v, %v and %v", weighed[0].Weights, weighed[1].Weights, weighed[2].Weights)
	}
	for _, unpassed := range passedSequences(group, make([]bool, len(group.Rollouts))) {
		if slices.ContainsFunc(unpassed.Weights, func(weight float32) bool { return weight != 0 }) {
			t.Fatalf("a group none of whose answers passed weighs %v", unpassed.Weights)
		}
	}
	root, request := rewardFixture(t)
	control := *request.Rewards
	control.Control = true
	request.Rewards, request.Steps, request.OutputDirectory = &control, 1, filepath.Join(root, "control")
	if _, err := Execute(t.Context(), request); err == nil || !strings.Contains(err.Error(), "weighted step") {
		t.Fatalf("a control session on the host = %v, want it refused", err)
	}
}
