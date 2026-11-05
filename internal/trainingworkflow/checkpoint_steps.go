package trainingworkflow

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/atomicfile"
	"overgo/internal/clioptions"
	"overgo/internal/densecausal"
	"overgo/internal/fsatomic"
	"overgo/internal/hostoptimizer"
	"overgo/internal/loratrain"
	"overgo/internal/optimizer"
	"overgo/internal/peft"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
	"overgo/internal/trainingprogram"
)

const (
	// stepCheckpointInfix joins a run's output directory to the number of updates a
	// checkpoint it saved along the way follows: output-step-000040.
	stepCheckpointInfix = "-step-"
	// stepCheckpointDigits is the width of that number, so the directories sort by it.
	stepCheckpointDigits = 6
	// averageDirectory holds a checkpoint's moving average in a directory of its own,
	// since its tensors carry the model's names and a safetensors reader of the
	// checkpoint reads every shard beside the weights: averageWeightsFile has the
	// average's values by tensor name, and averageStateFile how many updates it has
	// taken under what decay.
	averageDirectory   = "average"
	averageWeightsFile = "weights.safetensors"
	averageStateFile   = "state.json"
	// averageFileMode is the mode the average's state file is written with.
	averageFileMode = 0o600
)

// stepCheckpointDirectory is where a run writing to output saves its checkpoint
// after the given number of updates, beside output so that the checkpoints root
// holds them and a name given there resumes from one. The name joins the output's
// last element, so an output written with a trailing separator is still the
// directory the checkpoint sits beside, never inside.
func stepCheckpointDirectory(output string, updates int) string {
	clean := filepath.Clean(output)
	return filepath.Join(filepath.Dir(clean), filepath.Base(clean)+stepCheckpointInfix+fmt.Sprintf("%0*d", stepCheckpointDigits, updates))
}

// checkStepNames refuses a run to output that would save a checkpoint after the
// updates following first, every that many of the run's updates, where a directory
// already stands, before the run spends anything: the publish would refuse it only
// when the run reached it. An output that is not one path element beside its
// checkpoints is refused too. Directories a run does not write are left alone.
func checkStepNames(output string, every, first, updates int) error {
	if err := fsatomic.PathComponent(filepath.Base(filepath.Clean(output))); err != nil {
		return fmt.Errorf("training workflow: a run that saves checkpoints beside its output needs an output named by one element: %w", err)
	}
	for done := (first/every + 1) * every; done < updates; done += every {
		target := stepCheckpointDirectory(output, done)
		if _, err := os.Lstat(target); err == nil {
			return fmt.Errorf("training workflow: %s, where the run would save its checkpoint after update %d, already exists: resume into another output or remove it", target, done)
		} else if !errors.Is(err, fs.ErrNotExist) {
			return err
		}
	}
	return nil
}

// stepSaves is the checkpoints one run saved along the way to its output, oldest
// first. A run prunes only these: a directory an earlier run left beside the
// output, however it is named or numbered, is neither outranked by the run's saves
// nor deleted by them.
type stepSaves struct {
	keep int
	own  []string
}

// saved records the checkpoint the run just wrote at target and deletes its own
// oldest ones beyond the newest keep.
func (saves *stepSaves) saved(target string) error {
	saves.own = append(saves.own, target)
	for len(saves.own) > saves.keep {
		if err := os.RemoveAll(saves.own[0]); err != nil {
			return err
		}
		saves.own = saves.own[1:]
	}
	return nil
}

// averageState is what a checkpoint records of its moving average besides its values.
type averageState struct {
	Decay   float64 `json:"decay"`
	Updates int     `json:"updates"`
}

// averageCompanion writes the average into a staged checkpoint, its values split by
// the optimizer plan's tensor names, which is nothing for a run that keeps none.
func averageCompanion(average *optimizer.Average, plan optimizer.Plan) func(stage string) error {
	if average == nil {
		return nil
	}
	return func(stage string) error {
		if err := os.MkdirAll(filepath.Join(stage, averageDirectory), clioptions.OutputDirectoryMode); err != nil {
			return err
		}
		tensors, shapes := map[string][]float32{}, map[string][]int{}
		for _, group := range plan.Groups() {
			tensors[group.Name] = average.Values()[group.Start:group.End]
			shapes[group.Name] = []int{group.End - group.Start}
		}
		if err := safetensors.Save(filepath.Join(stage, averageDirectory, averageWeightsFile), tensors, shapes, nil); err != nil {
			return err
		}
		encoded, err := json.Marshal(averageState{Decay: average.Decay(), Updates: average.Updates()})
		if err != nil {
			return err
		}
		return atomicfile.Write(filepath.Join(stage, averageDirectory, averageStateFile), encoded, averageFileMode)
	}
}

// readAverage restores the moving average a checkpoint holds over the optimizer
// plan's layout, or nil when the checkpoint holds none.
func readAverage(directory string, plan optimizer.Plan) (*optimizer.Average, error) {
	encoded, err := os.ReadFile(filepath.Join(directory, averageDirectory, averageStateFile))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	var state averageState
	if err := strictjson.DecodeBytes(encoded, &state); err != nil {
		return nil, err
	}
	source, err := safetensors.OpenFile(filepath.Join(directory, averageDirectory, averageWeightsFile))
	if err != nil {
		return nil, err
	}
	defer source.Close()
	values := make([]float32, plan.ParameterCount())
	for _, group := range plan.Groups() {
		tensor, found := source.Tensors[group.Name]
		if !found {
			return nil, fmt.Errorf("training workflow: the checkpoint's average has no %s", group.Name)
		}
		read, err := safetensors.ReadF32(tensor)
		if err != nil {
			return nil, err
		}
		if len(read) != group.End-group.Start {
			return nil, fmt.Errorf("training workflow: the checkpoint's average holds %d values for %s, the plan's are %d", len(read), group.Name, group.End-group.Start)
		}
		copy(values[group.Start:group.End], read)
	}
	return optimizer.RestoreAverage(values, state.Updates, state.Decay)
}

// resumedRun is what a checkpoint a run saved along the way gives back: how many
// updates the run had taken, the optimizer's state then, its moving average, nil
// when it kept none, and for an adapter run the trainer over the adapter as it
// stood.
type resumedRun struct {
	done    int
	state   densecausal.TrainState
	average *optimizer.Average
	adapter *loratrain.Trainer
}

// restoreRun reads the checkpoint in directory, which must be one an earlier run of
// this objective on this model saved before its updates ended, and puts the model
// as the checkpoint left it: a full run's weights are copied into the model, and
// an adapter run's adapter, from the checkpoint's PEFT directory, is applied to it
// by a trainer that takes the optimizer state and average too. The plan is the
// layout the optimizer state must match.
func restoreRun(directory string, modelID artifact.ID, objective trainingprogram.ObjectiveDocument, model *densecausal.Model, plan hostoptimizer.Plan, lora bool, updates int, learningRate, momentum float64) (resumedRun, error) {
	checkpoint, err := trainingprogram.LoadCheckpoint(directory)
	if err != nil {
		return resumedRun{}, err
	}
	if checkpoint.Model != modelID || checkpoint.Dataset != objective.Dataset || checkpoint.Split != objective.Split {
		return resumedRun{}, errors.New("training workflow: the checkpoint belongs to another model or objective than the run resuming from it")
	}
	state := densecausal.TrainState(checkpoint.Optimizer)
	if err := hostoptimizer.ValidateState(state, plan.Identity(), plan.ParameterCount()); err != nil {
		return resumedRun{}, fmt.Errorf("training workflow: the checkpoint's optimizer state does not fit this run: %w", err)
	}
	if state.Step <= 0 || state.Step >= updates {
		return resumedRun{}, fmt.Errorf("training workflow: the checkpoint follows %d updates of a run of %d: nothing is left to resume", state.Step, updates)
	}
	average, err := readAverage(directory, plan)
	if err != nil {
		return resumedRun{}, err
	}
	resumed := resumedRun{done: state.Step, state: state, average: average}
	if lora {
		adapter, err := peft.Load(filepath.Join(directory, LoRAAdapterDirectory))
		if err != nil {
			return resumedRun{}, err
		}
		if resumed.adapter, err = loratrain.Resume(model, adapter); err != nil {
			return resumedRun{}, err
		}
		return resumed, resumed.adapter.Restore(state, average, learningRate, momentum)
	}
	saved, err := loadTrainableModel(directory)
	if err != nil {
		return resumedRun{}, err
	}
	for name, values := range model.Weights {
		if len(saved.Weights[name]) != len(values) {
			return resumedRun{}, fmt.Errorf("training workflow: the checkpoint holds %d values for %s, the model's are %d", len(saved.Weights[name]), name, len(values))
		}
		copy(values, saved.Weights[name])
	}
	return resumed, nil
}
