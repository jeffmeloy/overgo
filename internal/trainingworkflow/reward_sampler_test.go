package trainingworkflow

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
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
// The request names no device trainer: a session over it gives its own.
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
		Repository: store, Observations: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: prompts,
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
// A session over the model trains on the device, and is held there to
// sampling by that runtime from the weights as they stand
// (TestDeviceSamplerDrawsWhatTheHostSamplerDraws).
func TestRewardSamplerDrawsARealModelsCurrentWeights(t *testing.T) {
	heldToHostSampler(t, rewardGGUFFixture(t), true, servedAgreement)
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

// bootstrappedRewardGroups is how many groups a session under a bootstrapped
// reward recipe is run for.
const bootstrappedRewardGroups = 3

// bootstrappedRewardSession publishes, in a new store, the recipe a training
// command publishes for a reward session over a model's weights file and the
// suite of prompts, and holds it to being the same recipe published again and
// to no recipe being published for text that is no suite of prompts. It
// returns the request of a session of bootstrappedRewardGroups groups over
// that file and those prompts under the recipe, which names no device
// trainer. A GGUF is its own model; for a directory's weights file the caller
// names the directory as the session's model.
func bootstrappedRewardSession(t *testing.T, model string) Request {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	prompts := filepath.Join(root, "prompts.json")
	if err := os.WriteFile(prompts, []byte(rewardPrompts), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
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
	return Request{
		Repository: store, Observations: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: prompts, OutputDirectory: filepath.Join(root, "out"), Steps: bootstrappedRewardGroups,
		Rewards: &RewardSampling{
			Protocol: evaluation.InstructionProtocol{Scorer: evaluation.InstructionScorer, Template: rewardPlain, Thinking: pilotThinking, Sampling: rewardSampling, Stop: pilotStop, Window: 16, MaxTokens: 4},
			Training: testutil.ArtifactID(t, artifact.KindProfile, "bootstrapped reward training"), Seed: 5, Draws: 4, Code: "bootstrapped reward code",
		},
	}
}

// heldToARewardSession runs a session under a bootstrapped reward recipe and
// holds it to being a reward session of its groups, each sampled and then
// stepped on or passed.
func heldToARewardSession(t *testing.T, request Request) {
	t.Helper()
	result, err := Execute(t.Context(), request)
	if err != nil || result.Objective != trainingprogram.ObjectiveGRPO || result.Rewards.Sampled != request.Steps || result.Rewards.Updated+result.Rewards.Skipped != request.Steps {
		t.Fatalf("a session under the published recipe = %s with %+v (%v), want a reward session of %d groups", result.Objective, result.Rewards, err, request.Steps)
	}
}

// TestBootstrappedRewardRecipeRunsARewardSession holds the recipe a training
// command publishes for a reward session to being one a reward session runs
// under: published for a model and a suite of prompts, it is the same recipe
// published again, and a session over that model and those prompts under it
// samples its groups and steps, as one under an authority a test assembles
// does. A suite that is not instruction prompts publishes none.
//
// The recipe is published for a GGUF, as a pilot publishes it, and for a
// model that is a directory of weights. The session here is over the
// directory, which the host samples, and trains by the stand-in that takes
// the weighted step for real on the host (hostWeightedStep); the session over
// the GGUF samples in the device's runtime and trains by the device's
// trainer, and is the device's test
// (TestBootstrappedRewardRecipeRunsARewardSessionOnTheDevice).
func TestBootstrappedRewardRecipeRunsARewardSession(t *testing.T) {
	bootstrappedRewardSession(t, writeTrainingGGUF(t))
	_, fixture := rewardFixture(t)
	// A directory's recipe is published for its weights file, as a training command names it.
	weights, err := IdentifyModelWeights(fixture.ModelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	request := bootstrappedRewardSession(t, weights)
	request.ModelDirectory, request.Device = fixture.ModelDirectory, hostWeightedStep
	heldToARewardSession(t, request)
}
