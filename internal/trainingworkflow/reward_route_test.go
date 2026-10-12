package trainingworkflow

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// orderedPrompt is the prompt of orderedRewardFixture's one group.
const orderedPrompt = "ab"

// orderedRewardFixture writes a tiny model and one reward group, the answers
// to one prompt, each rewarded over those after it, and makes the authority
// of a reward session over answers already scored. Its device trainer is the
// stand-in that takes the weighted step on the host (hostWeightedStep).
func orderedRewardFixture(t *testing.T, answers ...string) (string, Request) {
	t.Helper()
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 11,
	})
	root := t.TempDir()
	model := filepath.Join(root, "model")
	writeModel(t, model, weights, shapes)
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "resume reward evaluator")
	dataset := filepath.Join(root, "rollouts.jsonl")
	var rows strings.Builder
	for index, answer := range answers {
		// The rewards step down by two about nothing: one and its negative for two answers.
		fmt.Fprintf(&rows, `{"id":"candidate-%c","group":"prompt-a","prompt":"%s","completion":"%s","reward":%d,"evaluator":"%s"}`+"\n",
			'a'+rune(index), orderedPrompt, answer, len(answers)-1-2*index, evaluator)
	}
	if err := os.WriteFile(dataset, []byte(rows.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	store, recipeID := trainingAuthority(t, model, "", dataset, trainingprogram.ObjectiveGRPO)
	return root, Request{Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset, Device: hostWeightedStep}
}

// hostWeightedStep stands in for the device trainer and takes its step for
// real, on the host: for each group, the weights' sum of the tokens'
// next-token losses is lowered by one update. It is built from the masked
// loss, which is a mean over the tokens it is given: a sequence's part of the
// group's loss is its weight times the count of its weighted tokens times
// that mean over them, and its part of the gradient the same multiple of the
// mean's gradients. That holds only where a sequence's weighted tokens share
// one weight, as a reward session's and a control's do, so unequal weights
// are refused. A sequence of no weight is left out, since the masked loss
// refuses a window none of whose tokens it is to learn, and a group of none
// takes no step and is counted as passed.
func hostWeightedStep(model *densecausal.Model, groups [][]densecausal.WeightedSequence, learningRate, momentum float64, _ bool, resume *densecausal.TrainState, _ densecausal.TrainObserver) ([]float64, int, densecausal.TrainState, error) {
	update, err := model.NewDenseUpdate(learningRate, momentum, resume)
	if err != nil {
		return nil, 0, densecausal.TrainState{}, err
	}
	var losses []float64
	skipped := 0
	for _, group := range groups {
		total, loss := densecausal.Grads{}, 0.0
		for _, sequence := range group {
			if len(sequence.Weights) != len(sequence.Tokens)-1 {
				return nil, 0, densecausal.TrainState{}, fmt.Errorf("the host's weighted step was handed %d weights for %d tokens, and takes one for each token after the first", len(sequence.Weights), len(sequence.Tokens))
			}
			trained := make([]bool, len(sequence.Tokens))
			var weight float32
			count := 0
			for token, each := range sequence.Weights {
				if each == 0 {
					continue
				}
				if count != 0 && each != weight {
					return nil, 0, densecausal.TrainState{}, fmt.Errorf("the host's weighted step was handed the unequal weights %v, and steps only a sequence whose weighted tokens share one weight", sequence.Weights)
				}
				weight, trained[token+1] = each, true
				count++
			}
			if count == 0 {
				continue
			}
			mean, gradients, err := model.MaskedLossAndGrads(sequence.Tokens, trained)
			if err != nil {
				return nil, 0, densecausal.TrainState{}, err
			}
			scale := float64(weight) * float64(count)
			loss += scale * mean
			for name, values := range gradients {
				if total[name] == nil {
					total[name] = make([]float32, len(values))
				}
				for at, value := range values {
					total[name][at] += float32(scale * float64(value))
				}
			}
		}
		if len(total) == 0 {
			skipped++
			continue
		}
		if err := update.Pack.GatherGradients(total); err != nil {
			return nil, 0, densecausal.TrainState{}, err
		}
		update.Step()
		update.Pack.Scatter()
		losses = append(losses, loss)
	}
	return losses, skipped, update.Snapshot(), nil
}

// TestRewardDeviceRouteTakesARealStepOnTheHost holds the route a reward
// session takes to training a model, with no graphics card: the device
// trainer here is a stand-in that takes the weighted step for real on the
// host (hostWeightedStep), where the other stand-ins step on no gradient.
// Its group is three answers of unequal lengths, the first rewarded over the
// second and the second over the third, so the middle one weighs nothing.
//
// The step teaches what the rewards say. After one step, the model puts the
// best answer further above the worst than the model it began from did: the
// step is handed each answer's weight with the sign a loss takes, and a
// weight handed with the other sign lowers that margin.
//
// And a session stopped at its checkpoint and resumed ends at the weights,
// the optimizer state and the place in its stream that one session ends at,
// over answers already scored and over answers the session draws and scores
// itself, where it steps on each of its groups.
func TestRewardDeviceRouteTakesARealStepOnTheHost(t *testing.T) {
	answers := []string{"cc", "d", "cab"}
	root, request := orderedRewardFixture(t, answers...)
	tokens, err := loadTrainableTokenizer(request.ModelDirectory)
	if err != nil {
		t.Fatal(err)
	}
	loaded := func(directory string) *densecausal.Model {
		t.Helper()
		model, err := loadTrainableModel(directory)
		if err != nil {
			t.Fatal(err)
		}
		return model
	}
	// margin is how far a model puts the best answer above the worst: the
	// difference of the logarithms of the probabilities it gives them after the prompt.
	margin := func(model *densecausal.Model) float64 {
		t.Helper()
		prompt, err := tokens.encode(orderedPrompt)
		if err != nil {
			t.Fatal(err)
		}
		var standing []float64
		for _, answer := range []string{answers[0], answers[len(answers)-1]} {
			ids, err := tokens.encode(orderedPrompt + answer)
			if err != nil {
				t.Fatal(err)
			}
			completion := make([]bool, len(ids))
			for token := len(prompt); token < len(ids); token++ {
				completion[token] = true
			}
			score, err := model.SequenceLogProb(trainingdata.PreferenceSequence{Tokens: ids, Completion: completion})
			if err != nil {
				t.Fatal(err)
			}
			standing = append(standing, score.LogProbability)
		}
		return standing[0] - standing[1]
	}
	// run is a session of so many steps, and the model it ends at.
	run := func(request Request, name string, steps int) (Result, *densecausal.Model) {
		t.Helper()
		request.Steps, request.OutputDirectory = steps, filepath.Join(root, name)
		result, err := Execute(t.Context(), request)
		if err != nil {
			t.Fatal(err)
		}
		return result, loaded(request.OutputDirectory)
	}
	// The step teaches what the rewards say.
	one, stepped := run(request, "one", 1)
	if one.Backend != "cuda" || len(one.GRPO) != 1 || one.Checkpoint.Optimizer.Step != 1 {
		t.Fatalf("one group = backend %q, %d observations, step %d; want the device route's one step", one.Backend, len(one.GRPO), one.Checkpoint.Optimizer.Step)
	}
	if before, after := margin(loaded(request.ModelDirectory)), margin(stepped); !(after > before) {
		t.Errorf("after one step the model puts the best answer %.6f above the worst, and before it %.6f; want the step to have raised it", after, before)
	}

	// Stopped at a checkpoint and resumed, over answers already scored and over answers the session draws.
	verifyWorkflowResume(t, root, request)
	sampledRoot, sampling := rewardFixture(t)
	if sampled := verifyWorkflowResume(t, sampledRoot, sampling); sampled.Rewards.Updated == 0 || sampled.Rewards.Skipped != 0 {
		t.Fatalf("the session that drew its own answers = %+v; want a step on each of its groups, so that the resumed half takes one", sampled.Rewards)
	}
}

// TestRewardSessionsTrainThroughTheWeightedStepAlone holds a reward session
// to the one trainer it has: the weighted step, which the device takes. The
// host has no reward trainer of its own.
//
// A session held to the host is refused with that reason and leaves nothing
// where its checkpoint was to go, whether it is over answers already scored
// or draws and scores its own, and although its request carries a device
// trainer, as a training command's does.
//
// Not held to the host, each goes to the device trainer its request carries,
// here the stand-in that takes the weighted step for real on the host
// (hostWeightedStep): the session is reported as the device's, the checkpoint
// it returns is the one published, after the one step, and the model
// published has moved from the one the session began at.
func TestRewardSessionsTrainThroughTheWeightedStepAlone(t *testing.T) {
	ctx := t.Context()
	scoredRoot, scored := orderedRewardFixture(t, "c", "d")
	drawnRoot, drawn := rewardFixture(t)
	for _, session := range []struct {
		name, root string
		request    Request
	}{{"over answers already scored", scoredRoot, scored}, {"that draws its own answers", drawnRoot, drawn}} {
		held := session.request
		held.Host, held.Steps, held.OutputDirectory = true, 1, filepath.Join(session.root, "held")
		refused, err := Execute(ctx, held)
		if err == nil || !strings.Contains(err.Error(), "weighted step") {
			t.Errorf("a reward session %s, held to the host, ended on backend %q with %v; want it refused, since a reward session trains through the weighted step and the host has no reward trainer", session.name, refused.Backend, err)
		}
		if _, err := os.Stat(held.OutputDirectory); !os.IsNotExist(err) {
			t.Errorf("a reward session %s, held to the host, left %s behind (%v); want nothing published by a session that was refused", session.name, held.OutputDirectory, err)
		}

		stepped := session.request
		stepped.Steps, stepped.OutputDirectory = 1, filepath.Join(session.root, "stepped")
		result, err := Execute(ctx, stepped)
		if err != nil {
			t.Fatalf("a reward session %s, not held to the host: %v", session.name, err)
		}
		published, err := trainingprogram.LoadCheckpoint(stepped.OutputDirectory)
		if err != nil || result.Backend != "cuda" || published.ID() != result.Checkpoint.ID() || published.Optimizer.Step != 1 {
			t.Fatalf("a reward session %s, not held to the host = backend %q, checkpoint %s; %s holds %s at step %d (%v); want the device's session and its checkpoint published after one step",
				session.name, result.Backend, result.Checkpoint.ID(), stepped.OutputDirectory, published.ID(), published.Optimizer.Step, err)
		}
		began, err := loadTrainableModel(stepped.ModelDirectory)
		if err != nil {
			t.Fatal(err)
		}
		ended, err := loadTrainableModel(stepped.OutputDirectory)
		if err != nil {
			t.Fatal(err)
		}
		moved := false
		for name, values := range began.Weights {
			moved = moved || !slices.Equal(values, ended.Weights[name])
		}
		if !moved {
			t.Errorf("a reward session %s, not held to the host, published the weights it began at; want the weighted step to have moved them", session.name)
		}
	}
}
