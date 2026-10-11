package trainingworkflow

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
)

const (
	// instructionTrainingSchema names the document an instruction pilot's
	// training specification is published as.
	instructionTrainingSchema = "overgo/instruction-training/v1"
	// pilotMatched says what the arms of the pilot are matched on, and what
	// is recorded of them and not matched.
	pilotMatched = "groups: every arm answers the same development prompts in one order, the same draws a prompt under one output limit; tokens sampled and trained, device time and memory are recorded and not matched"
	// checkpointDoubling is the growth from one checkpoint to the next.
	checkpointDoubling = 2
	// pilotOpenParts counts the parts of a pilot's groups whose prompts are
	// asked before an arm is frozen, the first of evaluation.PoolParts:
	// development, which the arms train on, and selection.
	pilotOpenParts = 2
	// pilotThinking and pilotStop are the pilot's protocol where it is not
	// chosen: no thinking is asked for, and an answer ends where the
	// vocabulary ends a generation.
	pilotThinking = "off"
	pilotStop     = "end of generation"
)

// pilotArms are the arms the pilot trains, each from the base's weights; the
// base is the third arm of its comparisons.
var pilotArms = []string{"reward", "control"}

// pilotComparisons are the comparisons the pilot is adopted by, named before
// any arm is trained.
var pilotComparisons = []evaluation.InstructionComparison{
	{Name: "reward over base", Candidate: "reward", Baseline: "base"},
	{Name: "reward over control", Candidate: "reward", Baseline: "control"},
}

// InstructionTraining is the specification the arms of the instruction pilot
// are trained under, published before the acceptance that names it: the arms,
// the answers drawn to a prompt, the passes over the development groups'
// prompts, the counts of groups after which an arm's weights are read on the
// selection groups, what the arms are matched on, and the most device memory a
// run may hold. The order answers are weighed by and the optimizer's policy
// are the training program's, which the run binds.
type InstructionTraining struct {
	Arms        []string `json:"arms"`
	Draws       int      `json:"draws"`
	Passes      int      `json:"passes"`
	Checkpoints []int    `json:"checkpoints"`
	Matched     string   `json:"matched"`
	DeviceBytes uint64   `json:"device_bytes"`
}

// InstructionPilot states what the instruction pilot's acceptance is fixed
// from: the base model, a GGUF; the template its prompts are asked under,
// "none" for a prompt as it is written and any other name for a user's turn
// of the model's chat, the tokens a prompt and its answer may fill together
// and an answer alone; the seed that draws every answer, each
// group's prompt and the division of the groups by Weights; the code that
// samples; the gain worth confirming; and the numbers of the training
// specification that are chosen and not derived, the draws, passes and device
// memory of Training. The rest of the protocol is a reward session's: answers
// drawn from the model's unmodified distribution to its end of generation,
// with no thinking asked for, and scored by the instruction scorer.
type InstructionPilot struct {
	Model             string
	Template          string
	Window, MaxTokens uint64
	Seed              int64
	Code              string
	Weights           [len(evaluation.PoolParts)]uint64
	Gain              float64
	Training          InstructionTraining
	Host              bool
}

// PilotReading is what the base answered on one part of the pilot's groups:
// the groups read, how many of them each draw passed every rule of, how many
// the first two draws differ on, the tokens sampled, and the answers that ran
// to the output limit.
type PilotReading struct {
	Part      string
	Groups    int
	Passed    []int
	Differing int
	Tokens    uint64
	Truncated int
}

// PilotOutcome is what fixing the pilot's acceptance came to: the base and
// the protocol it was read under; the cases and the groups they are, by part;
// the base's readings on the development and selection parts; the training
// specification as published and its identity; and the acceptance, or in
// Refused why its promotion groups cannot show the gain.
type PilotOutcome struct {
	Base          artifact.ID
	Protocol      evaluation.InstructionProtocol
	Cases         int
	Parts         [len(evaluation.PoolParts)]int
	Readings      []PilotReading
	Training      InstructionTraining
	Specification artifact.ID
	Acceptance    evaluation.InstructionAcceptance
	Refused       error
}

// FixInstructionPilot fixes the instruction pilot's acceptance from the
// store's instruction cases and readings of the base model, and publishes it.
// The cases are grouped by rule combination and the groups divided
// (evaluation.InstructionGroups). The base then answers the prompt of each
// development and selection group, drawn and scored as a reward session's
// answers are and kept in the store under the base's weights, so a session
// that begins from them reads them back; no prompt of the promotion or audit
// part is asked. The acceptance is sized from the development part alone:
// its groups, and those of them whose first two answers differ. The training
// specification is published first, with a checkpoint after one group and
// after each doubling up to a pass's end. An acceptance refused because its
// groups cannot show the gain is the pilot's answer and no fault: it is
// returned in the outcome, with the readings it was refused on.
func FixInstructionPilot(ctx context.Context, store artifact.Repository, pilot InstructionPilot) (_ PilotOutcome, err error) {
	path, err := trainedInput(pilot.Model)
	if err != nil || !isGGUFModelInput(path) {
		return PilotOutcome{}, errors.Join(fmt.Errorf("training workflow: the pilot's base %s is read by the inference runtime, which opens a GGUF", pilot.Model), err)
	}
	prompts, cases, err := evaluation.StoreInstructionPrompts(ctx, store)
	if err != nil {
		return PilotOutcome{}, err
	}
	base, err := identifyModel(path)
	if err != nil {
		return PilotOutcome{}, err
	}
	reading := RewardSampling{Seed: pilot.Seed, Draws: rewardDraws, Code: pilot.Code, Protocol: evaluation.InstructionProtocol{
		Scorer: evaluation.InstructionScorer, Template: pilot.Template, Thinking: pilotThinking, Sampling: rewardSampling, Stop: pilotStop, Window: pilot.Window, MaxTokens: pilot.MaxTokens,
	}}
	request := evaluation.InstructionAcceptanceRequest{
		Dataset: cases, Base: base, Cases: prompts.Cases(), Protocol: reading.Protocol, Seed: uint64(pilot.Seed),
		Weights: pilot.Weights, Gain: pilot.Gain, Comparisons: pilotComparisons,
	}
	parts, err := evaluation.InstructionGroups(request)
	if err != nil {
		return PilotOutcome{}, err
	}
	outcome := PilotOutcome{Base: base, Protocol: reading.Protocol, Cases: len(request.Cases), Training: pilot.Training}
	for index, names := range parts {
		outcome.Parts[index] = len(names)
	}
	training := &outcome.Training
	training.Arms, training.Matched = pilotArms, pilotMatched
	development := outcome.Parts[0]
	for groups := 1; groups < development*training.Passes; groups *= checkpointDoubling {
		training.Checkpoints = append(training.Checkpoints, groups)
	}
	training.Checkpoints = append(training.Checkpoints, development*training.Passes)
	if training.Draws < rewardDraws || training.Passes <= 0 || training.DeviceBytes == 0 || development == 0 {
		return PilotOutcome{}, fmt.Errorf("training workflow: the pilot's training specification names its draws, at least two, its passes and its device memory, and trains on development groups; it names %d draws, %d passes and %d bytes over %d groups", training.Draws, training.Passes, training.DeviceBytes, development)
	}
	specification, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, instructionTrainingSchema), *training)
	if err != nil {
		return PilotOutcome{}, err
	}
	request.Training, outcome.Specification = specification.Descriptor.ID, specification.Descriptor.ID
	if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: instructionTrainingSchema + "/" + request.Training.String(), Contents: []artifact.Content{specification}}); err != nil {
		return PilotOutcome{}, err
	}

	reading.Training = request.Training
	tokens, err := reading.tokenizer(path, store)
	if err != nil {
		return PilotOutcome{}, err
	}
	sampler := &rewardSampler{session: denseSession{ctx: ctx, request: Request{Repository: store, Observations: store, Host: pilot.Host}, inputDirectory: path}, tokens: tokens}
	defer func() { err = errors.Join(err, sampler.release()) }()
	for index, names := range parts[:pilotOpenParts] {
		read := PilotReading{Part: evaluation.PoolParts[index], Groups: len(names), Passed: make([]int, reading.Draws)}
		for _, name := range names {
			prompt, limit, _ := prompts.Asked(name)
			group, answers, err := sampler.draw(reading, name, prompt, limit, base, base)
			if err != nil {
				return PilotOutcome{}, err
			}
			passes := make([]bool, len(answers))
			for draw, answer := range answers {
				passed, rules, err := prompts.Passed(name, answer.Text)
				if err != nil {
					return PilotOutcome{}, err
				}
				if passes[draw] = passed == rules; passes[draw] {
					read.Passed[draw]++
				}
				read.Tokens += uint64(len(answer.Completion))
				if len(answer.Completion) >= group.MaxTokens {
					read.Truncated++
				}
			}
			if passes[0] != passes[1] {
				read.Differing++
			}
			if index != 0 {
				continue
			}
			drawn, err := group.Identity()
			if err != nil {
				return PilotOutcome{}, err
			}
			request.Basis = append(request.Basis, drawn)
		}
		outcome.Readings = append(outcome.Readings, read)
	}
	request.Pairs, request.Discordant = uint64(outcome.Readings[0].Groups), uint64(outcome.Readings[0].Differing)
	acceptance, batch, err := evaluation.BuildInstructionAcceptance(request)
	if errors.Is(err, evaluation.ErrInstructionEvidence) {
		outcome.Refused = err
		return outcome, nil
	}
	if err != nil {
		return PilotOutcome{}, err
	}
	outcome.Acceptance = acceptance
	_, err = artifact.Publish(ctx, store, batch)
	return outcome, err
}
