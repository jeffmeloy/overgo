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
	"overgo/internal/densecausal"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/processmeasure"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// phaseClock is the time a reward session subtracts to time a phase, from the
// high-resolution counter. time.Now steps by the system's tick, half a
// millisecond or more on Windows, and a phase on a small model is shorter, so
// a session's sampling read as no time whenever no draw crossed a tick. A
// counter that cannot be read reads as no time, as that clock did.
func phaseClock() time.Duration {
	now, _ := processmeasure.Counter()
	return now
}

// rewardSampling names the one way a reward session draws its answers: from
// the model's own distribution, no top-p or top-k cut and no temperature but
// one. A protocol that asks for another is refused.
const rewardSampling = "unmodified"

// rewardDraws is the fewest answers a group is drawn: one answer has no order.
const rewardDraws = 2

// rewardPlain names the template under which a prompt is asked as it is
// written. Under any other name it is asked as a user's turn of a chat.
const rewardPlain = "none"

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
	// Control makes the session the control of a reward experiment: it steps
	// on the answers that passed every rule alone, as fine-tuning does, and
	// not on every answer by its standing.
	Control bool
}

// tokenizer loads the tokenizer the answers are drawn with from a model, and
// refuses what no reward session and no reading of an arm is drawn under: a
// cut or sharpened distribution, another scorer, thinking asked for, a stop
// other than the vocabulary's end of generation, fewer than least answers a
// prompt, no training specification, no store to keep the draws in, or a
// model that declares no end token. The sampler does one thing with thinking
// and one with stopping, so a protocol that names another is refused here,
// before a draw, and not answered as if it had been honoured. A prompt asked
// as a chat turn carries the template's control tokens, which are then read
// as tokens.
func (rewards RewardSampling) tokenizer(input string, observations artifact.Repository, least int) (trainableTokenizer, error) {
	tokens, err := loadTrainableTokenizerFor(input, rewards.Render != nil || rewards.Protocol.Template != rewardPlain)
	if protocol := rewards.Protocol; err == nil && (protocol.Sampling != rewardSampling || protocol.Scorer != evaluation.InstructionScorer || protocol.Thinking != pilotThinking || protocol.Stop != pilotStop ||
		rewards.Draws < least || !rewards.Training.Valid() || !tokens.ended || observations == nil) {
		err = fmt.Errorf("training workflow: a reward session draws at least two answers a prompt from the model's %s distribution with thinking %s, to the %s, scores them by %s, names its training specification and the store its draws are kept in, and trains a model that declares its end token; it was asked for %d draws under sampling %q, thinking %q, stop %q and scorer %q",
			rewardSampling, pilotThinking, pilotStop, evaluation.InstructionScorer, rewards.Draws, protocol.Sampling, protocol.Thinking, protocol.Stop, protocol.Scorer)
	}
	return tokens, err
}

// sampledAs names everything a session's sampling is fixed by, its renderer
// by whether it has one: the identity a checkpoint of the session is bound to.
func (rewards RewardSampling) sampledAs() (artifact.ID, error) {
	return artifact.JSONID(artifact.KindProfile, struct {
		Protocol evaluation.InstructionProtocol `json:"protocol"`
		Training artifact.ID                    `json:"training"`
		Seed     int64                          `json:"seed"`
		Draws    int                            `json:"draws"`
		Code     string                         `json:"code"`
		Rendered bool                           `json:"rendered"`
		Control  bool                           `json:"control"`
	}{rewards.Protocol, rewards.Training, rewards.Seed, rewards.Draws, rewards.Code, rewards.Render != nil, rewards.Control})
}

// published says whether a checkpoint found where a session of this sampling
// would publish is that session's: trained from base, sampled as this
// sampling samples, continuing the checkpoint before it where there is one
// and none where there is not, and standing after so many groups. A
// checkpoint that fails any of them was made by another run, and what it is
// not is the error.
func (rewards RewardSampling) published(found trainingprogram.Checkpoint, base artifact.ID, before trainingprogram.Checkpoint, groups int) error {
	sampling, err := rewards.sampledAs()
	if err != nil {
		return err
	}
	continued := slices.IndexFunc(found.Lineage, func(parent trainingprogram.LineageParent) bool {
		return parent.Relation == artifact.RelationDerivedFrom
	})
	switch {
	case found.Model != base:
		return fmt.Errorf("it was trained from model %s, and the base is %s", found.Model, base)
	case !slices.Contains(found.Lineage, trainingprogram.LineageParent{Artifact: sampling, Relation: artifact.RelationDependsOn}):
		return errors.New("it was not sampled by this protocol, training specification, seed, draws, code and control")
	case before.ID().Valid() != (continued >= 0) || continued >= 0 && found.Lineage[continued].Artifact != before.ID():
		return errors.New("it does not continue the checkpoint before it")
	case found.Stream.Position != uint64(groups):
		return fmt.Errorf("it stands after %d groups", found.Stream.Position)
	}
	return nil
}

// rewardSamplingKey opens the key a session's sampling identity is published under.
const rewardSamplingKey = "overgo/reward-sampling/v1/"

// bound adds the session's sampling to what its checkpoint descends from,
// publishing its name in the store the draws are kept in, and refuses to
// continue a checkpoint that was sampled otherwise: a resume compares the
// model, the data and the optimizer, and none of those says by which
// protocol, specification, seed, draws, code or control the answers it
// trained on were made. A session with no store is refused before it draws
// (tokenizer), and names nothing here.
func (rewards RewardSampling) bound(ctx context.Context, store artifact.Repository, resumed trainingprogram.Checkpoint, lineage []trainingprogram.LineageParent) ([]trainingprogram.LineageParent, error) {
	sampling, err := rewards.sampledAs()
	if err != nil {
		return nil, err
	}
	parent := trainingprogram.LineageParent{Artifact: sampling, Relation: artifact.RelationDependsOn}
	if resumed.ID().Valid() && !slices.Contains(resumed.Lineage, parent) {
		return nil, fmt.Errorf("training workflow: checkpoint %s was not sampled as this session samples, by its protocol, training specification, seed, draws, code and control: resume authority differs", resumed.ID())
	}
	if store != nil {
		if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: rewardSamplingKey + sampling.String(), Artifacts: []artifact.Descriptor{{ID: sampling}}}); err != nil {
			return nil, err
		}
	}
	return append(lineage, parent), nil
}

// RewardTrace is what a reward session that samples its own answers did:
// the groups it sampled, of which those it stepped on and those, their
// answers all standing together, it passed; the tokens it sampled and the
// tokens it trained on; the answers that ran to the output limit; the time
// it spent sampling, which is the sampler's whole time, opening the weights
// and reading answers under them with it, scoring and stepping; and how far,
// on the tokens it sampled, the stepped groups' answers moved in
// log-probability a token under the weights after each step, which describes
// the run and bounds nothing.
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
// far, with the probability it was drawn at, until an ending token, a stop
// text, the limit, or the model's context is reached. It is how a model
// trained over a safetensors directory is sampled; a GGUF's weights are
// opened in the inference runtime (rewardSampler).
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
		// An ending token is no text of the answer, as the inference runtime gives it none.
		ending, piece := slices.Contains(sampler.tokens.ends, drawn.Token), ""
		if !ending {
			piece = sampler.tokens.piece(drawn.Token)
		}
		if err := options.OnToken(inference.TokenEvent{ID: tokenizer.TokenID(drawn.Token), Piece: piece, Index: index, SelectedProbability: drawn.SelectedProbability}); err != nil {
			return nil, "", err
		}
		ids = append(ids, drawn.Token)
		text.WriteString(piece)
		if ending || slices.ContainsFunc(options.StopSequences, func(stop string) bool { return strings.HasSuffix(text.String(), stop) }) {
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
// after it, by the sampler that serves the model's representation
// (rewardSampler), which also reads how a stepped group's answers stand
// under the weights after the step.
//
// A session that is run again from the same checkpoint reaches the same
// weights and so names the same groups: the draws it recorded are read back
// and not sampled again, a group it had not finished is finished from the
// same weights by the same seeds, and each step is taken once. So what an
// interrupted session keeps is its draws and its last checkpoint, and nothing
// it lost is a draw or a step it could count twice.
func (session denseSession) produceRewards(prepared densePrepared, resume *densecausal.RLState) (_ Result, _ densecausal.TrainState, err error) {
	rewards := *session.request.Rewards
	tokens, err := rewards.tokenizer(session.inputDirectory, session.request.Observations, rewardDraws)
	if err != nil {
		return Result{}, densecausal.TrainState{}, err
	}
	sampler := &rewardSampler{session: session, tokens: tokens}
	defer func() { err = errors.Join(err, sampler.release()) }()
	base, err := identifyModel(session.inputDirectory)
	if session.resumed.ID().Valid() {
		base = session.resumed.Model
	}
	evaluator, evaluatorErr := prepared.prompts.Evaluator()
	if err != nil || evaluatorErr != nil {
		return Result{}, densecausal.TrainState{}, errors.Join(err, evaluatorErr)
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
	for _, produced := range prepared.cases {
		prompt, limit, _ := prepared.prompts.Asked(produced.name)
		weights, err := evaluation.UpdatedWeights(base, rewards.Training, uint64(state.Optimizer.Step))
		if err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		began := phaseClock()
		group, answers, err := sampler.draw(rewards, produced.name, prompt, limit, base, weights)
		if err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		trace.Generation += phaseClock() - began
		began = phaseClock()
		rollouts, passes := make([]trainingdata.Rollout, len(answers)), make([]bool, len(answers))
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
			rollouts[index], passes[index] = trainingdata.Rollout{ID: fmt.Sprintf("draw-%d", answer.Draw), Sequence: sequence, Reward: trainingprogram.InstructionOrder(passed, rules, uint64(len(answer.Completion)))}, passed == rules
			for _, logarithm := range answer.LogProbabilities {
				sampled[index] += logarithm
			}
			trace.SampledTokens += uint64(len(answer.Completion))
			if answer.End == "output limit" {
				trace.Truncated++
			}
		}
		trace.Scoring += phaseClock() - began
		trace.Sampled++
		rollout, err := trainingdata.NewRolloutGroup(produced.name, evaluator, produced.state, rollouts)
		if err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		began = phaseClock()
		observations, stepped, err := session.stepRewards(rollout, passes, state)
		if err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		trace.Update += phaseClock() - began
		state = stepped
		if len(observations) == 0 {
			trace.Skipped++
			continue
		}
		trace.Updated++
		// The step moved the weights: what was opened of them is no longer the model.
		if err := sampler.release(); err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		observation := observations[0]
		trace.TrainedTokens += observation.CompletionTokens
		// How the sampled answers stand under the weights after the step, a token: a description of the step.
		// Opening those weights and reading the answers under them is the sampler's time.
		began = phaseClock()
		after, err := sampler.standing(group.Prompt, answers, rollouts)
		if err != nil {
			return Result{}, densecausal.TrainState{}, err
		}
		trace.Generation += phaseClock() - began
		for index, standing := range after {
			trace.Shift += (standing - sampled[index]) / float64(observation.CompletionTokens)
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
// after it. passes says which answers passed every rule. A control session
// steps on those alone, each weighed over its own tokens as a fine-tuning
// step weighs them, and passes a group none of whose answers passed; it
// trains through the weighted step, which is the device's.
func (session denseSession) stepRewards(group trainingdata.RolloutGroup, passes []bool, state densecausal.RLState) ([]trainingprogram.GRPOObservation, densecausal.RLState, error) {
	resume := &state
	if !state.Stream.Identity.Valid() {
		resume = nil
	}
	weigh := densecausal.RewardSequences
	if session.request.Rewards.Control {
		weigh = func(group trainingdata.RolloutGroup) ([]densecausal.WeightedSequence, error) {
			return passedSequences(group, passes), nil
		}
	} else if session.request.Host {
		return session.model.TrainGRPOGroupsResume([]trainingdata.RolloutGroup{group}, session.optimizerConfig.BaseLearningRate, session.optimizerConfig.Momentum, resume, nil)
	}
	if session.request.Host {
		return nil, state, errors.New("training workflow: a control session trains through the weighted step, which the device takes; it has no host trainer")
	}
	// The step's observation is returned to whoever asked for the step, who observes it.
	session.request.ObserveGRPO = nil
	stepped, optimizer, err := session.rewardsOnDevice([]trainingdata.RolloutGroup{group}, resume, weigh)
	return stepped.GRPO, densecausal.RLState{Optimizer: optimizer, Stream: group.State, Skipped: state.Skipped + stepped.Skipped}, err
}

// passedSequences is a group as a control session trains on it: each answer
// that passed every rule is one sequence whose own tokens share a weight of
// one, the mean a fine-tuning step takes over an answer, and an answer that
// did not pass, like every prompt token, weighs nothing.
func passedSequences(group trainingdata.RolloutGroup, passes []bool) []densecausal.WeightedSequence {
	sequences := make([]densecausal.WeightedSequence, len(group.Rollouts))
	for index, rollout := range group.Rollouts {
		weights := make([]float32, len(rollout.Sequence.Tokens)-1)
		answer := 0
		for _, own := range rollout.Sequence.Completion {
			if own {
				answer++
			}
		}
		for token := range weights {
			if passes[index] && rollout.Sequence.Completion[token+1] {
				weights[token] = 1 / float32(answer)
			}
		}
		sequences[index] = densecausal.WeightedSequence{Tokens: rollout.Sequence.Tokens, Weights: weights}
	}
	return sequences
}
