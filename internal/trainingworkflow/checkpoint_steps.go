package trainingworkflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/atomicfile"
	"overgo/internal/clioptions"
	"overgo/internal/densecausal"
	"overgo/internal/fsatomic"
	"overgo/internal/hostoptimizer"
	"overgo/internal/loratrain"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/pathidentity"
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
	// modelPage bounds one read of the store's listing of its models, which
	// refuses a read with no bound; the listing is read to its end, a page
	// after a page.
	modelPage = 1000
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

// RemoveCheckpoints removes the checkpoints a run wrote at directories, once
// the run's own record is kept, and writes receipt a line for each: removed,
// kept or left, then the bytes, the place, the path and the rule, between
// tabs. With a reason in kept it removes nothing and the reason is the rule.
//
// A directory that holds no checkpoint document is passed over. A checkpoint
// the store records at a path in its directory is kept: a recorded checkpoint
// is read by its location, and only one the store was never told the place of
// is a run's to throw away. A directory in which the store records a model is
// kept as well, whatever its checkpoint is to the store: a registered model
// is never removed, and whether any model is in use is never asked. The
// store's models are read once, before anything is removed, and a read that
// fails removes nothing (registeredModels). A removal that fails is left, its
// error the rule, and the others still go; the errors are returned together.
// A folder the removals left empty is removed too, and one that still holds
// something is given a line.
func RemoveCheckpoints(ctx context.Context, store modelCatalog, receipt io.Writer, kept string, directories ...string) error {
	var failed error
	line := func(verb string, bytes int64, path, rule string) {
		_, err := fmt.Fprintf(receipt, "%s\t%d\trun-checkpoint\t%s\t%s\n", verb, bytes, path, rule)
		failed = errors.Join(failed, err)
	}
	var models []modelLocation
	if kept == "" {
		var err error
		if models, err = registeredModels(ctx, store); err != nil {
			return err
		}
	}
	var parents []string
	for _, directory := range directories {
		if kept != "" {
			line("kept", 0, directory, kept)
			continue
		}
		data, err := os.ReadFile(filepath.Join(directory, trainingprogram.CheckpointFilename))
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		var bytes int64
		held := ""
		if err == nil {
			bytes, held, err = throwawayBytes(ctx, store, models, directory, data)
		}
		if err == nil && held == "" {
			err = fsatomic.RemoveTree(directory)
		}
		switch {
		case err != nil:
			line("left", bytes, directory, err.Error())
			failed = errors.Join(failed, err)
		case held != "":
			line("kept", 0, directory, held)
		default:
			line("removed", bytes, directory, "a throw-away checkpoint of a run whose record is kept")
			if parent := filepath.Dir(directory); !slices.Contains(parents, parent) {
				parents = append(parents, parent)
			}
		}
	}
	for _, parent := range parents {
		if entries, err := os.ReadDir(parent); err != nil || len(entries) != 0 {
			line("kept", 0, parent, "it still holds what this run did not remove")
			failed = errors.Join(failed, err)
			continue
		}
		failed = errors.Join(failed, os.Remove(parent))
	}
	return failed
}

// modelCatalog is the store RemoveCheckpoints reads: where an artifact is
// recorded, and the catalog's listing, which names its models. An
// *overgodb.Store is one.
type modelCatalog interface {
	Locations(context.Context, artifact.ID) ([]artifact.Location, error)
	Query(context.Context, overgodb.Query) (overgodb.QueryResult, error)
}

type modelLocation struct {
	artifact.Location
	model artifact.ID
}

// registeredModels reads every place on this machine the store records a
// model at, a page of its catalog at a time, as the model listings of
// internal/modeltest and cmd/loop read it. Any read that fails fails the
// whole, and so does a listing cut short with nowhere to go on from: what is
// decided from this is what may be removed, and a listing that did not come
// whole must never read as a store that registers no model.
func registeredModels(ctx context.Context, store modelCatalog) ([]modelLocation, error) {
	var recorded []modelLocation
	query := overgodb.Query{Kind: artifact.KindModel, MaxResults: modelPage, Projection: overgodb.ProjectArtifacts | overgodb.ProjectManifests}
	for {
		page, err := store.Query(ctx, query)
		if err != nil {
			return nil, err
		}
		manifests := make(map[artifact.ID]artifact.Manifest, len(page.Manifests))
		for _, manifest := range page.Manifests {
			manifests[manifest.ID] = manifest
		}
		for _, model := range page.Artifacts {
			manifest, known := manifests[model.ID]
			for _, component := range append(manifest.Components, artifact.Component{Artifact: model.ID}) {
				locations, err := store.Locations(ctx, component.Artifact)
				if err != nil {
					return nil, err
				}
				if !known && len(locations) == 0 {
					return nil, fmt.Errorf("training workflow: model %s has neither a manifest nor locations, so no checkpoint is removed", model.ID)
				}
				for _, location := range locations {
					if location.Kind == artifact.LocationRemote {
						continue
					}
					if !known {
						return nil, fmt.Errorf("training workflow: local model %s has no manifest, so no checkpoint is removed", model.ID)
					}
					recorded = append(recorded, modelLocation{Location: location, model: model.ID})
				}
			}
		}
		if query.Cursor = page.Next; query.Cursor != nil {
			continue
		}
		if page.Truncated {
			return nil, errors.New("training workflow: the store's listing of its models is cut short, so no checkpoint is removed")
		}
		return recorded, nil
	}
}

// throwawayBytes reads the checkpoint document data of directory and gives
// the rule the directory is held by, when the store records that checkpoint
// or one of models, the registered models' locations, at a path in it. When
// it records neither there, the rule is empty and bytes is how much the
// directory's files hold. The checkpoint's own record is looked for first, so
// a recorded checkpoint that holds its trained model is kept as a checkpoint.
func throwawayBytes(ctx context.Context, store modelCatalog, models []modelLocation, directory string, data []byte) (bytes int64, held string, err error) {
	checkpoint, err := trainingprogram.ParseCheckpoint(data)
	if err != nil {
		return 0, "", err
	}
	locations, err := store.Locations(ctx, checkpoint.ID())
	if err != nil {
		return 0, "", err
	}
	all := make([]modelLocation, len(locations), len(locations)+len(models))
	for i, location := range locations {
		all[i].Location = location
	}
	for _, location := range append(all, models...) {
		if location.Kind == artifact.LocationRemote {
			continue
		}
		inside, err := pathidentity.Contains(directory, location.Value)
		if err != nil {
			return 0, "", err
		}
		if inside && location.Artifact == checkpoint.ID() {
			return 0, "the store records this checkpoint at this path, so it is not a run's to throw away", nil
		}
		if inside {
			return 0, fmt.Sprintf("the store records %s for model %s at %s, and a registered model is never a run's to throw away", location.Artifact, location.model, location.Value), nil
		}
	}
	err = filepath.WalkDir(directory, func(_ string, entry fs.DirEntry, err error) error {
		if err != nil || !entry.Type().IsRegular() {
			return err
		}
		info, err := entry.Info()
		if err == nil {
			bytes += info.Size()
		}
		return err
	})
	return bytes, "", err
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
