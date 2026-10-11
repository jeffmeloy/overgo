package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/modelintake"
	"overgo/internal/overgodb"
	"overgo/internal/trainingworkflow"
)

const (
	// pilotTemplate names how the pilot asks a prompt: as one user's turn of
	// the model's own chat template, or the runtime's for a model that
	// carries none, with the reply invited.
	pilotTemplate = "chat/user-turn"
	// gibibyte is the unit -pilot-memory is stated in.
	gibibyte = 1 << 30
)

// pilotRequest is what the flags state of an instruction pilot.
type pilotRequest struct {
	model, weights         string
	gain                   float64
	seed                   int64
	window, tokens, memory uint64
	draws, passes          int
}

// poolWeights reads the weights that divide a pool's or a pilot's units among
// the four parts, each of which must have one.
func poolWeights(text string) (weights [len(evaluation.PoolParts)]uint64, err error) {
	for field := range strings.SplitSeq(text, ",") {
		name, value, _ := strings.Cut(strings.TrimSpace(field), "=")
		weight, err := strconv.Atoi(value)
		if at := slices.Index(evaluation.PoolParts[:], name); err == nil && at >= 0 && weight > 0 {
			weights[at] = uint64(weight)
		}
	}
	if slices.Contains(weights[:], 0) {
		return weights, fmt.Errorf("training-objective: -pool-weights %q: want a positive weight for each of %s", text, strings.Join(evaluation.PoolParts[:], ", "))
	}
	return weights, nil
}

// fixPilot fixes the instruction pilot's acceptance over a base model from
// the store's instruction cases (trainingworkflow.FixInstructionPilot) and
// writes what it was fixed from: the cases and their groups by part, the
// protocol, what the base answered on the development and selection groups,
// the training specification and the acceptance, each document on a line
// beside the identity it is published under, so a record of the output can be
// held to them. A pilot whose groups cannot show the gain is written as
// refused, with the readings it was refused on, and is no failure of the
// command. Every number is stated by whoever runs this: none has a default.
func fixPilot(repository string, request pilotRequest, output io.Writer) error {
	weights, err := poolWeights(request.weights)
	if err != nil {
		return err
	}
	if request.gain <= 0 || request.seed == 0 || request.window == 0 || request.tokens == 0 || request.draws == 0 || request.passes == 0 || request.memory == 0 {
		return fmt.Errorf("training-objective: -pilot-gain, -pilot-seed, -pilot-window, -pilot-tokens, -pilot-draws, -pilot-passes and -pilot-memory are required: the gain in the share of groups passed that is worth confirming, the seed, the tokens a prompt and its answer may fill and an answer alone, the answers an arm draws to a prompt in training, its passes over the development groups, and the device memory a run may hold, in GiB")
	}
	root, err := dataroot.StoreRoot(repository)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	code, err := modelintake.CleanRevision(ctx)
	if err != nil {
		return err
	}
	// What was charged before a part named what it holds is found from the split fixed here only once those parts are named.
	if _, err := evaluation.NameEarlierParts(ctx, store); err != nil {
		return err
	}
	outcome, err := trainingworkflow.FixInstructionPilot(ctx, store, trainingworkflow.InstructionPilot{
		Model: request.model, Template: pilotTemplate, Window: request.window, MaxTokens: request.tokens, Seed: request.seed, Code: code, Weights: weights, Gain: request.gain,
		Training: trainingworkflow.InstructionTraining{Draws: request.draws, Passes: request.passes, DeviceBytes: request.memory * gibibyte},
	})
	if err != nil {
		return err
	}
	document := func(id artifact.ID) (string, error) {
		content, found, err := artifact.ReadContent(ctx, store, id)
		if err != nil || !found {
			return "", fmt.Errorf("training-objective: %s is not in the store it was published to: %w", id, err)
		}
		return string(content.Data), nil
	}
	specification, err := document(outcome.Specification)
	if err != nil {
		return err
	}
	groups := 0
	for _, part := range outcome.Parts {
		groups += part
	}
	protocol := outcome.Protocol
	fmt.Fprintf(output, "instruction pilot over base %s (%s), built at commit %s\n"+
		"cases: %d prompts in %d groups, one a combination of rules, divided %v by seed %d and weights %s: %v\n"+
		"protocol: scorer %s, template %s, thinking %s, sampling %s, stop %s, window %d tokens, output limit %d tokens\n",
		outcome.Base, filepath.Base(request.model), code, outcome.Cases, groups, evaluation.PoolParts, request.seed, request.weights, outcome.Parts,
		protocol.Scorer, protocol.Template, protocol.Thinking, protocol.Sampling, protocol.Stop, protocol.Window, protocol.MaxTokens)
	for _, read := range outcome.Readings {
		fmt.Fprintf(output, "%s: %d groups read twice, %d differing; passed by draw %v; %d tokens sampled, %d answers at the output limit\n", read.Part, read.Groups, read.Differing, read.Passed, read.Tokens, read.Truncated)
	}
	fmt.Fprintf(output, "training: %s\nspecification: %s\n", outcome.Specification, specification)
	if outcome.Refused != nil {
		_, err = fmt.Fprintf(output, "refused: %v\n", outcome.Refused)
		return err
	}
	acceptance := outcome.Acceptance
	fixed, err := document(acceptance.ID)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "sized: a gain of %.2f in the share of groups passed is shown with chance %.3f over %d promotion groups, arms differing on at most %.3f of them; a gain that is not there is reported at most %.2f of the time over the %d comparisons, and a gain of that size is missed at most %.2f\n"+
		"split: %s\nacceptance: %s\ndocument: %s\n",
		acceptance.Gain, acceptance.Power, acceptance.Groups, acceptance.Discordance, acceptance.FalseGain, len(acceptance.Comparisons), acceptance.MissedGain, acceptance.Split, acceptance.ID, fixed)
	return err
}
