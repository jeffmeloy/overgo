package trainingworkflow

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/sampling"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// servedAgreement bounds how far the probability the inference runtime draws
// an id with stands from the one the model in training gives it on the host,
// as a share of it: the two are one forward in two implementations of f32
// arithmetic, measured on the fixture to differ by at most four parts in ten
// million with the runtime on the host and five with it on the device.
const servedAgreement = 1e-5

// rewardGGUFFixture is rewardFixture over a model that is a GGUF: the fixture
// model, whose vocabulary ends an answer at one token, the suite of prompts
// over its letters, and the authority of a reward session over them in a
// store that knows the architecture, as a store a model is served from does.
// Two fixtures are the same model and prompts in two stores.
func rewardGGUFFixture(t *testing.T) Request {
	t.Helper()
	model := writeTrainingGGUF(t)
	prompts := filepath.Join(filepath.Dir(model), "prompts.json")
	if err := os.WriteFile(prompts, []byte(rewardPrompts), 0o600); err != nil {
		t.Fatal(err)
	}
	store, recipeID := trainingAuthority(t, model, "", prompts, trainingprogram.ObjectiveGRPO)
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	return Request{
		Repository: store, Observations: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: prompts, Host: true,
		Rewards: &RewardSampling{
			Protocol: evaluation.InstructionProtocol{Scorer: evaluation.InstructionScorer, Template: "none", Thinking: pilotThinking, Sampling: rewardSampling, Stop: pilotStop, Window: 16, MaxTokens: 4},
			Training: testutil.ArtifactID(t, artifact.KindProfile, "reward fixture training"), Seed: 5, Draws: 4, Code: "reward fixture code",
		},
	}
}

// TestRewardSamplerDrawsARealModelsCurrentWeights holds the sampler a reward
// session draws a GGUF model's answers by to the model in training, on the
// host.
//
// The model's weights are opened in the inference runtime, and what it draws
// is what the host sampler draws from the model in training: the same ids
// from the same seeds, each with the probability the host gives it to within
// servedAgreement, and ended at the vocabulary's ending token. Moved, and
// released as a session releases them after a step, the weights are opened
// again as they then stand: the draws are the moved model's, which are not the
// base's. Nothing written to open them is left when they are released
// (heldToHostSampler).
//
// A session over the model samples by that runtime and scores what it
// sampled: the text of a recorded answer is its own ids' text, with none of
// the prompt's, which the runtime's returned text begins with. And a session
// stopped at its checkpoint and resumed in a store that held none of its
// draws ends at the bytes an uninterrupted one ends at, so each group was
// drawn from the weights as they stood: the resumed session opens the
// checkpoint's weights where the uninterrupted one opened its own after a
// step (rewardSessionOverGGUF).
func TestRewardSamplerDrawsARealModelsCurrentWeights(t *testing.T) {
	request := rewardGGUFFixture(t)
	heldToHostSampler(t, request, true, servedAgreement)
	rewardSessionOverGGUF(t, request)
}

// heldToHostSampler opens the fixture model's weights in the inference
// runtime, on the host or on the device, and holds what it draws to what the
// host sampler draws from the model in training: the same ids from the same
// seeds, each at the host's probability to within agreement as a share of
// it, ended at the end token or the limit, some each way. It then moves the
// weights, releases them as a session does after a step, and holds the
// weights opened next to the moved model the same way, and to drawing other
// answers than the base did. Nothing written to open them is left.
func heldToHostSampler(t *testing.T, request Request, onHost bool, agreement float64) {
	t.Helper()
	ctx := t.Context()
	root := filepath.Dir(request.ModelDirectory)
	model, err := loadTrainableModel(request.ModelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := loadTrainableTokenizer(request.ModelDirectory)
	if err != nil || !slices.Equal(tokens.ends, []int{tokens.end}) {
		t.Fatalf("the fixture's ending tokens = (%v, %v), want its one end token %d", tokens.ends, err, tokens.end)
	}
	encoded, err := tokens.encode("ab")
	if err != nil {
		t.Fatal(err)
	}
	prompt := make([]tokenizer.TokenID, len(encoded))
	for index, id := range encoded {
		prompt[index] = tokenizer.TokenID(id)
	}
	draw := func(generator evaluation.Generator, seed int64) (ids []tokenizer.TokenID, chances []float64) {
		t.Helper()
		sampler, err := sampling.New(sampling.Config{Temperature: 1, Seed: seed})
		if err != nil {
			t.Fatal(err)
		}
		returned, _, err := generator.Generate(ctx, "", inference.GenerateOptions{
			MaxNewTokens: 8, Sampler: sampler, PromptTokenIDs: prompt, PostSamplingProbabilities: 1,
			OnToken: func(event inference.TokenEvent) error {
				ids, chances = append(ids, event.ID), append(chances, event.SelectedProbability)
				return nil
			},
		})
		if err != nil || !slices.Equal(returned, slices.Concat(prompt, ids)) {
			t.Fatalf("seed %d returned %v after sampling %v (%v)", seed, returned, ids, err)
		}
		return ids, chances
	}
	host := modelSampler{model: model, tokens: tokens}
	served := &rewardSampler{
		session: denseSession{ctx: ctx, request: Request{Repository: request.Repository, OutputDirectory: filepath.Join(root, "sampled"), Host: onHost}, inputDirectory: request.ModelDirectory, model: model},
		tokens:  tokens,
	}
	const seeds = 24
	agree := func(weights string) (drawn [][]tokenizer.TokenID, ended int) {
		t.Helper()
		generator, err := served.generator()
		if err != nil {
			t.Fatal(err)
		}
		if again, err := served.generator(); err != nil || again != generator {
			t.Fatalf("%s were opened twice with no step between (%v)", weights, err)
		}
		var answers []evaluation.SampledAnswer
		var rollouts []trainingdata.Rollout
		var sampled []float64
		for seed := range int64(seeds) {
			wantIDs, wantChances := draw(host, seed)
			gotIDs, gotChances := draw(generator, seed)
			if !slices.Equal(gotIDs, wantIDs) {
				t.Fatalf("%s, seed %d: the runtime drew %v where the host draws %v", weights, seed, gotIDs, wantIDs)
			}
			logarithm := 0.0
			for index, want := range wantChances {
				if math.Abs(gotChances[index]-want) > agreement*want {
					t.Fatalf("%s, seed %d: id %d was drawn at %v where the host gives it %v", weights, seed, index, gotChances[index], want)
				}
				logarithm += math.Log(gotChances[index])
			}
			sequence := trainingdata.PreferenceSequence{Tokens: slices.Clone(encoded), Completion: make([]bool, len(encoded), len(encoded)+len(gotIDs))}
			for _, id := range gotIDs {
				sequence.Tokens, sequence.Completion = append(sequence.Tokens, int(id)), append(sequence.Completion, true)
			}
			answers, rollouts, sampled = append(answers, evaluation.SampledAnswer{Completion: gotIDs}), append(rollouts, trainingdata.Rollout{Sequence: sequence}), append(sampled, logarithm)
			if last := len(gotIDs) - 1; int(gotIDs[last]) == tokens.end {
				ended++
			} else if slices.Contains(gotIDs, tokenizer.TokenID(tokens.end)) || last+1 != 8 {
				t.Fatalf("%s, seed %d: %v neither ends at the end token nor runs to the limit", weights, seed, gotIDs)
			}
			drawn = append(drawn, gotIDs)
		}
		// An answer stands under the weights it was drawn from where it was drawn, and where the host scores it.
		standing, err := served.standing(prompt, answers, rollouts)
		if err != nil {
			t.Fatal(err)
		}
		for index, got := range standing {
			scored, err := model.SequenceLogProb(rollouts[index].Sequence)
			if err != nil {
				t.Fatal(err)
			}
			if bound := agreement * math.Abs(scored.LogProbability); math.Abs(got-scored.LogProbability) > bound || math.Abs(got-sampled[index]) > bound {
				t.Fatalf("%s, seed %d: the answer stands at %v by the runtime, %v by the host, and was drawn at %v", weights, index, got, scored.LogProbability, sampled[index])
			}
		}
		return drawn, ended
	}
	before, ended := agree("the base's weights")
	if ended == 0 || ended == seeds {
		t.Fatalf("%d of %d draws ended at the end token, want some and not all, so both endings are held", ended, seeds)
	}
	for _, values := range model.Weights {
		for index := range values {
			values[index] *= 1.5
		}
	}
	if err := served.release(); err != nil {
		t.Fatal(err)
	}
	after, _ := agree("the moved weights")
	if slices.EqualFunc(before, after, func(one, other []tokenizer.TokenID) bool { return slices.Equal(one, other) }) {
		t.Fatal("the moved weights draw what the base's drew: they were not the weights opened")
	}
	if err := served.release(); err != nil {
		t.Fatal(err)
	}
	if left, err := filepath.Glob(filepath.Join(root, "reward-weights-*")); err != nil || len(left) != 0 {
		t.Fatalf("released weights left %v behind (%v)", left, err)
	}
}

// rewardSessionOverGGUF runs a reward session over the fixture model, on the
// host or the device as the request asks, and holds it to sampling by the
// inference runtime from the weights as they stand: its groups are sampled
// and some stepped; the text of an answer it recorded is its own ids' text,
// with none of the prompt's; and stopped at its checkpoint and resumed in a
// store that held none of its draws, it ends at the bytes it ends at
// uninterrupted. It leaves nothing it wrote to open its weights.
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
	other.Host, other.Device = request.Host, request.Device
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

// TestBootstrappedRewardRecipeRunsARewardSession holds the recipe a training
// command publishes for a reward session to being one a reward session runs
// under: published for a model and a suite of prompts, it is the same recipe
// published again, and a session over that model and those prompts under it
// samples its groups and steps, as one under an authority a test assembles
// does. A suite that is not instruction prompts publishes none.
func TestBootstrappedRewardRecipeRunsARewardSession(t *testing.T) {
	ctx := t.Context()
	model := writeTrainingGGUF(t)
	root := filepath.Dir(model)
	prompts := filepath.Join(root, "prompts.json")
	if err := os.WriteFile(prompts, []byte(rewardPrompts), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	recipeID, err := bootstrapRewardRecipe(ctx, store, model, []byte(rewardPrompts))
	if err != nil {
		t.Fatal(err)
	}
	if again, err := bootstrapRewardRecipe(ctx, store, model, []byte(rewardPrompts)); err != nil || again != recipeID {
		t.Fatalf("published again the recipe is %s (%v), want %s", again, err, recipeID)
	}
	if _, err := bootstrapRewardRecipe(ctx, store, model, []byte("a b c d")); err == nil {
		t.Fatal("text that is no suite of prompts published a reward recipe")
	}
	const groups = 3
	result, err := Execute(ctx, Request{
		Repository: store, Observations: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: prompts, OutputDirectory: filepath.Join(root, "out"), Steps: groups, Host: true,
		Rewards: &RewardSampling{
			Protocol: evaluation.InstructionProtocol{Scorer: evaluation.InstructionScorer, Template: rewardPlain, Thinking: pilotThinking, Sampling: rewardSampling, Stop: pilotStop, Window: 16, MaxTokens: 4},
			Training: testutil.ArtifactID(t, artifact.KindProfile, "bootstrapped reward training"), Seed: 5, Draws: 4, Code: "bootstrapped reward code",
		},
	})
	if err != nil || result.Objective != trainingprogram.ObjectiveGRPO || result.Rewards.Sampled != groups || result.Rewards.Updated+result.Rewards.Skipped != groups {
		t.Fatalf("a session under the published recipe = %s with %+v (%v), want a reward session of %d groups", result.Objective, result.Rewards, err, groups)
	}
}
