package trainingworkflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/trainingprogram"
)

// pilotBase names the arm that is the base model, which no session trains.
const pilotBase = "base"

// PilotRun states one run of the instruction pilot under a published
// acceptance: the base model, a GGUF; the directory its arms' checkpoints are
// written under, which a run begun again continues from; the code that
// samples; and where it trains.
type PilotRun struct {
	Acceptance artifact.ID
	Model, Out string
	Code       string
	Host       bool
	Device     DeviceTrainer
}

// PilotCheckpoint is an arm's weights after so many development groups: where
// they are published, how many selection groups their answers passed, and
// what the session that reached them from the checkpoint before did.
type PilotCheckpoint struct {
	Groups    int
	Directory string
	Selected  int
	Trace     RewardTrace
}

// PilotArm is one trained arm: its checkpoints in order, and the one frozen
// for the comparisons, chosen on the selection groups alone.
type PilotArm struct {
	Name        string
	Checkpoints []PilotCheckpoint
	Frozen      int
}

// PilotComparison is one comparison of the acceptance's family as the
// promotion part answered it, and how many the part has left after it.
type PilotComparison struct {
	Name         string
	Confirmation evaluation.PoolConfirmation
	Left         uint64
}

// PilotResult is what a run of the pilot came to: the selection groups the
// base passed and their count, each arm, each comparison, and whether the
// reward arm is adopted, with the reason it is not.
type PilotResult struct {
	Selection, BaseSelected int
	Arms                    []PilotArm
	Comparisons             []PilotComparison
	Adopted                 bool
	Reason                  string
}

// pilotAdopted is the pilot's adoption rule: the reward arm is adopted only
// when every comparison the acceptance named was charged and answered a gain.
// A comparison that is missing, was never charged, or answered a loss or not
// enough evidence leaves the base in place, and is named as the reason.
func pilotAdopted(acceptance evaluation.InstructionAcceptance, comparisons []PilotComparison) (bool, string) {
	for _, named := range acceptance.Comparisons {
		at := slices.IndexFunc(comparisons, func(answered PilotComparison) bool { return answered.Name == named.Name })
		switch {
		case at < 0:
			return false, fmt.Sprintf("comparison %q was not made", named.Name)
		case !comparisons[at].Confirmation.Charge.Valid() || comparisons[at].Confirmation.Acceptance != acceptance.ID:
			return false, fmt.Sprintf("comparison %q was not charged under acceptance %s", named.Name, acceptance.ID)
		case !comparisons[at].Confirmation.Gained():
			return false, fmt.Sprintf("comparison %q answered %s", named.Name, comparisons[at].Confirmation.Verdict)
		}
	}
	return len(acceptance.Comparisons) != 0, ""
}

// RunInstructionPilot trains the pilot's arms and judges them under its
// acceptance. Each arm starts from the base's weights and answers the
// development groups' prompts in one order, a reward session or its control,
// to each of the training specification's checkpoints in turn; a checkpoint a
// run before this one published is kept and continued from. Each checkpoint
// answers the selection groups' prompts once, and the one that passes most of
// them, the earliest of equals, is frozen: no promotion outcome enters the
// choice. Then each comparison the acceptance named is charged and read on
// the promotion groups, one answer an arm a group, and the reward arm is
// adopted only if every one answered a gain (pilotAdopted). Every answer is
// drawn and kept as a reward session's are, under the weights that gave it.
func RunInstructionPilot(ctx context.Context, store *overgodb.Store, run PilotRun) (PilotResult, error) {
	acceptance, parts, err := evaluation.RequireInstructionAcceptance(ctx, store, run.Acceptance)
	if err != nil {
		return PilotResult{}, err
	}
	var training InstructionTraining
	content, found, err := artifact.ReadContent(ctx, store, acceptance.Training)
	if err != nil || !found || json.Unmarshal(content.Data, &training) != nil || len(training.Checkpoints) == 0 {
		return PilotResult{}, errors.Join(fmt.Errorf("training workflow: acceptance %s names training specification %s, which is not a published pilot specification", acceptance.ID, acceptance.Training), err)
	}
	prompts, _, err := evaluation.StoreInstructionPrompts(ctx, store)
	if err != nil {
		return PilotResult{}, err
	}
	base, err := identifyModel(run.Model)
	if err != nil || base != acceptance.Base {
		return PilotResult{}, errors.Join(fmt.Errorf("training workflow: acceptance %s was fixed over base %s, and %s is %s", acceptance.ID, acceptance.Base, run.Model, base), err)
	}
	development, selection := parts[0], parts[1]
	suite, err := prompts.Suite(development)
	if err != nil {
		return PilotResult{}, err
	}
	asked := filepath.Join(run.Out, "development.json")
	if err := errors.Join(os.MkdirAll(run.Out, changeDirectoryMode), os.WriteFile(asked, suite, changeRunFileMode)); err != nil {
		return PilotResult{}, err
	}
	recipeID, err := bootstrapRewardRecipe(ctx, store, run.Model, suite)
	if err != nil {
		return PilotResult{}, err
	}
	reading := RewardSampling{Protocol: acceptance.Protocol, Training: acceptance.Training, Seed: int64(acceptance.Seed), Draws: 1, Code: run.Code}
	// read is how many of the named prompts an arm's weights pass every rule of, and which.
	read := func(path string, names []string) ([]bool, int, error) {
		path, err := trainedInput(path)
		if err != nil {
			return nil, 0, err
		}
		weights, err := identifyModel(path)
		if err != nil {
			return nil, 0, err
		}
		tokens, err := reading.tokenizer(path, store, reading.Draws)
		if err != nil {
			return nil, 0, err
		}
		sampler := &rewardSampler{session: denseSession{ctx: ctx, request: Request{Repository: store, Observations: store, Host: run.Host}, inputDirectory: path}, tokens: tokens}
		passes, count := make([]bool, len(names)), 0
		for index, name := range names {
			prompt, limit, _ := prompts.Asked(name)
			_, answers, err := sampler.draw(reading, name, prompt, limit, base, weights)
			if err != nil {
				return nil, 0, errors.Join(err, sampler.release())
			}
			passed, rules, err := prompts.Passed(name, answers[0].Text)
			if err != nil {
				return nil, 0, errors.Join(err, sampler.release())
			}
			if passes[index] = passed == rules; passes[index] {
				count++
			}
		}
		return passes, count, sampler.release()
	}
	result := PilotResult{Selection: len(selection)}
	if _, result.BaseSelected, err = read(run.Model, selection); err != nil {
		return PilotResult{}, err
	}
	frozen := map[string]string{pilotBase: run.Model}
	for _, name := range training.Arms {
		arm := PilotArm{Name: name}
		specification, err := artifact.JSONID(artifact.KindProfile, struct {
			Training artifact.ID `json:"training"`
			Arm      string      `json:"arm"`
		}{acceptance.Training, name})
		if err != nil {
			return PilotResult{}, err
		}
		if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: instructionTrainingSchema + "/arm/" + specification.String(), Artifacts: []artifact.Descriptor{{ID: specification}}}); err != nil {
			return PilotResult{}, err
		}
		if err := os.MkdirAll(filepath.Join(run.Out, name), changeDirectoryMode); err != nil {
			return PilotResult{}, err
		}
		previous, reached := "", 0
		for _, groups := range training.Checkpoints {
			checkpoint := PilotCheckpoint{Groups: groups, Directory: filepath.Join(run.Out, name, fmt.Sprintf("groups-%03d", groups))}
			if _, err := trainingprogram.LoadCheckpoint(checkpoint.Directory); err != nil {
				request := Request{
					Repository: store, Observations: store, Recipe: recipeID, DatasetPath: asked, OutputDirectory: checkpoint.Directory,
					ModelDirectory: run.Model, ResumeDirectory: previous, Steps: groups - reached, Host: run.Host, Device: run.Device,
					Rewards: &RewardSampling{Protocol: acceptance.Protocol, Training: specification, Seed: int64(acceptance.Seed), Draws: training.Draws, Code: run.Code, Control: name != pilotArms[0]},
				}
				if previous != "" {
					request.ModelDirectory = ""
				}
				session, err := Execute(ctx, request)
				if err != nil {
					return PilotResult{}, fmt.Errorf("training workflow: the %s arm to %d groups: %w", name, groups, err)
				}
				checkpoint.Trace = session.Rewards
			}
			if _, checkpoint.Selected, err = read(checkpoint.Directory, selection); err != nil {
				return PilotResult{}, err
			}
			if len(arm.Checkpoints) == 0 || checkpoint.Selected > arm.Checkpoints[arm.Frozen].Selected {
				arm.Frozen = len(arm.Checkpoints)
			}
			arm.Checkpoints = append(arm.Checkpoints, checkpoint)
			previous, reached = checkpoint.Directory, groups
		}
		frozen[name] = arm.Checkpoints[arm.Frozen].Directory
		result.Arms = append(result.Arms, arm)
	}
	for _, named := range acceptance.Comparisons {
		candidate, baseline := frozen[named.Candidate], frozen[named.Baseline]
		if candidate == "" || baseline == "" {
			return PilotResult{}, fmt.Errorf("training workflow: comparison %q is of %s over %s, and the pilot trains %v beside its base", named.Name, named.Candidate, named.Baseline, training.Arms)
		}
		var identities []artifact.ID
		for _, path := range []string{candidate, baseline} {
			path, err := trainedInput(path)
			if err != nil {
				return PilotResult{}, err
			}
			identity, err := identifyModel(path)
			if err != nil {
				return PilotResult{}, err
			}
			identities = append(identities, identity)
		}
		confirmation, left, err := evaluation.ConfirmInstruction(ctx, store, acceptance.ID, acceptance.Protocol, named.Name, identities[0], identities[1], func(names []string) ([]bool, []bool, error) {
			passed, _, err := read(candidate, names)
			if err != nil {
				return nil, nil, err
			}
			against, _, err := read(baseline, names)
			return passed, against, err
		})
		if err != nil {
			return PilotResult{}, err
		}
		result.Comparisons = append(result.Comparisons, PilotComparison{Name: named.Name, Confirmation: confirmation, Left: left})
	}
	result.Adopted, result.Reason = pilotAdopted(acceptance, result.Comparisons)
	return result, nil
}
