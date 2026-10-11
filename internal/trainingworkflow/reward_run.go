package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/densecausal"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// rewardSampling names the one way a reward session draws its answers: from
// the model's own distribution, no top-p or top-k cut and no temperature but
// one. A protocol that asks for another is refused.
const rewardSampling = "unmodified"

// rewardDraws is the fewest answers a group is drawn: one answer has no order.
const rewardDraws = 2

// RewardSampling is what a reward session that samples its own answers is
// fixed by: the protocol its answers are made and scored under, which is its
// instruction acceptance's; the training specification it runs under; the
// seed its draws come of; how many answers it draws to a prompt; and the
// identity of the code that samples. The session's dataset is then a suite of
// instruction prompts, and no file of prepared rollouts is read. Render, when
// given, puts each prompt through the model's chat template as a user's turn
// with the reply invited, the way the evaluation asks it; without it a prompt
// is asked as it is written.
type RewardSampling struct {
	Protocol evaluation.InstructionProtocol
	Training artifact.ID
	Seed     int64
	Draws    int
	Code     string
	Render   ConversationRenderer
}

// RewardTrace is what a reward session that samples its own answers did:
// the groups it sampled, of which those it stepped on and those, their
// answers all standing together, it passed; the tokens it sampled and the
// tokens it trained on; the answers that ran to the output limit; the time
// it spent sampling, scoring and stepping; and how far, on the tokens it
// sampled, the stepped groups' answers moved in log-probability a token under
// the weights after each step, which describes the run and bounds nothing.
// Stopped, when set, says why the session ended before its prompts did.
type RewardTrace struct {
	Sampled, Updated, Skipped    int
	SampledTokens, TrainedTokens uint64
	Truncated                    int
	Generation, Scoring, Update  time.Duration
	Shift                        float64
	Stopped                      string
}

// modelSampler samples from a model in training on the host: each token is
// drawn by the request's sampler from the model's logits for the tokens so
// far, with the probability it was drawn at, until the model's end token, a
// stop text, the limit, or the model's context is reached.
type modelSampler struct {
	model  *densecausal.Model
	tokens trainableTokenizer
}

// Generate samples one answer to the prompt ids the options carry, by the
// options' sampler and as far as their limit, and returns the prompt's ids
// with the sampled ones after them and the sampled ones' text.
func (sampler modelSampler) Generate(ctx context.Context, _ string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	ids := make([]int, len(options.PromptTokenIDs))
	for index, id := range options.PromptTokenIDs {
		ids[index] = int(id)
	}
	var text strings.Builder
	for index := 0; index < options.MaxNewTokens && len(ids) < sampler.model.Dims.ContextLength; index++ {
		if err := ctx.Err(); err != nil {
			return nil, "", err
		}
		logits, err := sampler.model.Logits(ids)
		if err != nil {
			return nil, "", err
		}
		drawn, err := options.Sampler.SampleWithHistoryProbabilities(logits[len(logits)-sampler.model.Dims.Vocab:], ids, 1)
		if err != nil {
			return nil, "", err
		}
		piece := sampler.tokens.piece(drawn.Token)
		if err := options.OnToken(inference.TokenEvent{ID: tokenizer.TokenID(drawn.Token), Piece: piece, Index: index, SelectedProbability: drawn.SelectedProbability}); err != nil {
			return nil, "", err
		}
		ids = append(ids, drawn.Token)
		text.WriteString(piece)
		if sampler.tokens.ended && drawn.Token == sampler.tokens.end || slices.ContainsFunc(options.StopSequences, func(stop string) bool { return strings.HasSuffix(text.String(), stop) }) {
			break
		}
	}
	sampled := make([]tokenizer.TokenID, len(ids))
	for index, id := range ids {
		sampled[index] = tokenizer.TokenID(id)
	}
	return sampled, text.String(), nil
}

// produceRewards is a reward session that samples its own groups. For each
// prompt its stream gives it, in order, it draws the fixed number of answers
// from the weights as they stand, each draw recorded with its ids and
// probabilities under the identity of those weights (evaluation.SampleGroup);
// scores each answer by the prompt's rules; puts the answers in the
// instruction objective's one order; and takes at most one step on the group,
// on the host or the device as the session trains. A group is drawn whole
// from one set of weights before its step, and the next from the weights
// after it.
//
// A session that is run again from the same checkpoint reaches the same
// weights and so names the same groups: the draws it recorded are read back
// and not sampled again, a group it had not finished is finished from the
// same weights by the same seeds, and each step is taken once. So what an
// interrupted session keeps is its draws and its last checkpoint, and nothing
// it lost is a draw or a step it could count twice.
func (session denseSession) produceRewards(prepared densePrepared, resume *densecausal.RLState) (Result, densecausal.TrainState, error) {
	rewards := session.request.Rewards
	tokens, err := loadTrainableTokenizer(session.inputDirectory)
	if err != nil {
		return Result{}, densecausal.TrainState{}, err
	}
	base, err := identifyModel(session.inputDirectory)
	if session.resumed.ID().Valid() {
		base = session.resumed.Model
	}
	evaluator, evaluatorErr := prepared.prompts.Evaluator()
	switch protocol := rewards.Protocol; {
	case err != nil || evaluatorErr != nil:
		return Result{}, densecausal.TrainState{}, errors.Join(err, evaluatorErr)
	case protocol.Sampling != rewardSampling || protocol.Scorer != evaluation.InstructionScorer || rewards.Draws < rewardDraws || !rewards.Training.Valid() || !tokens.ended || session.request.Observations == nil:
		return Result{}, densecausal.TrainState{}, fmt.Errorf("training workflow: a reward session draws at least two answers a prompt from the model's %s distribution, scores them by %s, names its training specification and the store its draws are kept in, and trains a model that declares its end token; it was asked for %d draws under sampling %q and scorer %q", rewardSampling, evaluation.InstructionScorer, rewards.Draws, protocol.Sampling, protocol.Scorer)
	}
	state := densecausal.RLState{}
	if resume != nil {
		state = *resume
	}
	result := Result{Backend: "host", Objective: session.objective}
	if !session.request.Host {
		result.Backend = "cuda"
	}
	trace := &result.Rewards
	sampler := modelSampler{model: session.model, tokens: tokens}
	for _, produced := range prepared.cases {
		prompt, limit, _ := prepared.prompts.Asked(produced.name)
		var err error
		if rewards.Render != nil {
			prompt, err = rewards.Render([]dataset.ChatMessage{{Role: "user", Content: prompt}}, true)
		}
		var encoded []int
		if err == nil {
			encoded, err = tokens.encode(prompt)
		}
		if err != nil {
			return Result{}, densecausal.TrainState{}, fmt.Errorf("training workflow: prompt %q: %w", produced.name, err)
		}
		weights, err := evaluation.UpdatedWeights(base, rewards.Training, uint64(state.Optimizer.Step))
		if err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		group := evaluation.SampledGroup{
			Group: produced.name, Code: rewards.Code, Weights: weights, Tokenizer: base, Template: rewards.Protocol.Template,
			Ends: []tokenizer.TokenID{tokenizer.TokenID(tokens.end)}, Temperature: 1, Seed: rewards.Seed,
			MaxTokens: min(int(rewards.Protocol.MaxTokens), limit, session.model.Dims.ContextLength-len(encoded)),
		}
		for _, id := range encoded {
			group.Prompt = append(group.Prompt, tokenizer.TokenID(id))
		}
		began := time.Now()
		answers, err := evaluation.SampleGroup(session.ctx, session.request.Observations, sampler, group, rewards.Draws)
		if err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		trace.Generation += time.Since(began)
		began = time.Now()
		rollouts := make([]trainingdata.Rollout, len(answers))
		sampled := make([]float64, len(answers))
		for index, answer := range answers {
			passed, rules, err := prepared.prompts.Passed(produced.name, answer.Text)
			if err != nil {
				return Result{}, densecausal.TrainState{}, err
			}
			ids, completion := group.Tokens(answer)
			sequence := trainingdata.PreferenceSequence{Tokens: make([]int, len(ids)), Completion: completion}
			for at, id := range ids {
				sequence.Tokens[at] = int(id)
			}
			rollouts[index] = trainingdata.Rollout{ID: fmt.Sprintf("draw-%d", answer.Draw), Sequence: sequence, Reward: trainingprogram.InstructionOrder(passed, rules, uint64(len(answer.Completion)))}
			for _, logarithm := range answer.LogProbabilities {
				sampled[index] += logarithm
			}
			trace.SampledTokens += uint64(len(answer.Completion))
			if answer.End == "output limit" {
				trace.Truncated++
			}
		}
		trace.Scoring += time.Since(began)
		trace.Sampled++
		rollout, err := trainingdata.NewRolloutGroup(produced.name, evaluator, produced.state, rollouts)
		if err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		began = time.Now()
		observations, stepped, err := session.stepRewards(rollout, state)
		if err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		trace.Update += time.Since(began)
		state = stepped
		if len(observations) == 0 {
			trace.Skipped++
			continue
		}
		trace.Updated++
		observation := observations[0]
		trace.TrainedTokens += observation.CompletionTokens
		// How the sampled answers stand under the weights after the step, a token: a description of the step.
		for index, sequence := range rollouts {
			score, err := session.model.SequenceLogProb(sequence.Sequence)
			if err != nil {
				return Result{}, densecausal.TrainState{}, err
			}
			trace.Shift += (score.LogProbability - sampled[index]) / float64(observation.CompletionTokens)
		}
		if math.IsNaN(observation.Loss) || math.IsInf(observation.Loss, 0) || math.IsNaN(trace.Shift) || math.IsInf(trace.Shift, 0) {
			trace.Stopped = fmt.Sprintf("the step on prompt %q gave a loss of %v and a shift of %v", produced.name, observation.Loss, trace.Shift)
			break
		}
		if session.request.ObserveGRPO != nil {
			session.request.ObserveGRPO(observation)
		} else {
			result.GRPO = append(result.GRPO, observation)
		}
	}
	result.StreamPosition, result.Skipped = state.Stream.Position, uint64(trace.Skipped)
	return result, state.Optimizer, nil
}

// stepRewards takes at most one step on one reward group from a state, by
// the trainer the session trains with, and returns what was observed of the
// step, nothing when the group's answers all stand together, and the state
// after it.
func (session denseSession) stepRewards(group trainingdata.RolloutGroup, state densecausal.RLState) ([]trainingprogram.GRPOObservation, densecausal.RLState, error) {
	resume := &state
	if !state.Stream.Identity.Valid() {
		resume = nil
	}
	if session.request.Host {
		return session.model.TrainGRPOGroupsResume([]trainingdata.RolloutGroup{group}, session.optimizerConfig.BaseLearningRate, session.optimizerConfig.Momentum, resume, nil)
	}
	// The step's observation is returned to whoever asked for the step, who observes it.
	session.request.ObserveGRPO = nil
	stepped, optimizer, err := session.rewardsOnDevice([]trainingdata.RolloutGroup{group}, resume)
	return stepped.GRPO, densecausal.RLState{Optimizer: optimizer, Stream: group.State, Skipped: state.Skipped + stepped.Skipped}, err
}
