package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/processcontrol"
	"overgo/internal/trainingprogram"
)

// rewardStepsEnvironment names the file a training command leaves a reward
// run's steps in for the process that started it. A run by hand leaves it
// unset and writes none.
const rewardStepsEnvironment = "OVERGO_TRAIN_REWARD_STEPS"

// handRewardSteps writes a finished run's reward steps to the file its
// starter named. The file must be new: one already there is another run's.
func handRewardSteps(steps []trainingprogram.GRPOObservation) error {
	path := os.Getenv(rewardStepsEnvironment)
	if path == "" {
		return nil
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, clioptions.PrivateFileMode)
	if err == nil {
		err = errors.Join(clioptions.WriteJSON(file, steps), file.Close())
	}
	if err != nil {
		return fmt.Errorf("training workflow: reward steps handoff: %w", err)
	}
	return nil
}

// ExecuteInCommand runs a reward request in the training command at the path
// command, and returns when the command has ended. The command is a child
// process under the process supervisor: it is stopped when ctx ends, and it
// does not outlive the caller's process. It trains on the device and holds
// the trainer alone, so the device memory its session records is the run's
// and not the caller's. store must name the store the request's Repository
// and Observations are: the command opens it by path. A request the command's
// flags cannot carry is refused before anything starts. Of the result only
// Checkpoint is set, and GRPO when the request observes no steps itself;
// either way the steps arrive when the run has ended, not as it takes them.
// The child's output goes to the request's Progress.
func ExecuteInCommand(ctx context.Context, command, store string, request Request) (Result, error) {
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	if request.Audio != nil || request.ReferenceDirectory != "" || request.MaximumSequence != 0 || request.ObjectiveScale != 0 ||
		request.Host || request.Device != nil || request.FreezeLexical || request.Rewards != nil ||
		request.PreferenceMode != "" || request.PreferenceMargin != 0 || request.ObserveDPO != nil ||
		request.MaxProjectedWall != 0 || request.Observations == nil {
		return Result{}, errors.New("training workflow: the training command takes a recipe, a model or a checkpoint to resume, a dataset, an output and a count of steps, and trains off the host with its session recorded; this request asks for something else")
	}
	scratch, err := os.MkdirTemp("", "overgo-reward-run-")
	if err != nil {
		return Result{}, err
	}
	// Removed on return, and a removal that fails is let be: what the run
	// trained and recorded stands whatever becomes of two scratch files.
	defer os.RemoveAll(scratch)
	steps, refusal := filepath.Join(scratch, "steps.json"), filepath.Join(scratch, "error.txt")
	arguments := []string{"-store", store, "-recipe", request.Recipe.String(), "-dataset", request.DatasetPath, "-out", request.OutputDirectory, "-steps", strconv.Itoa(request.Steps)}
	if request.ModelDirectory != "" {
		arguments = append(arguments, "-model", request.ModelDirectory)
	}
	if request.ResumeDirectory != "" {
		arguments = append(arguments, "-resume", request.ResumeDirectory)
	}
	// Both of the child's streams go to one writer: the tail a failure is
	// reported with and, when the request has one, its progress.
	tail := clioptions.NewTailWriter(clioptions.DiagnosticTailBytes)
	output := io.Writer(tail)
	if request.Progress != nil {
		output = io.MultiWriter(tail, request.Progress)
	}
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: command, Args: arguments, Stdout: output, Stderr: output,
		Env: append(os.Environ(), rewardStepsEnvironment+"="+steps, clioptions.ErrorFileEnvironment+"="+refusal),
	})
	// The run's error before its status: it carries the context's, so a
	// cancelled run is recorded as cancelled, where its status would say only
	// that the command ended with one.
	if err != nil {
		return Result{}, err
	}
	if receipt.ExitCode != 0 {
		// The command's own error as it wrote it, or, from one that ended
		// outside its error path, the last of what it printed.
		cause, readErr := os.ReadFile(refusal)
		if readErr != nil {
			cause = []byte(tail.Tail())
		}
		return Result{}, fmt.Errorf("training workflow: the training command ended with status %d: %s", receipt.ExitCode, strings.TrimSpace(string(cause)))
	}
	var observed []trainingprogram.GRPOObservation
	if err := jsonfile.DecodeStrict(steps, &observed); err != nil {
		return Result{}, fmt.Errorf("training workflow: the training command ended without handing back its steps, as a training command of another build does: %w", err)
	}
	checkpoint, err := trainingprogram.LoadCheckpoint(request.OutputDirectory)
	if err != nil {
		return Result{}, err
	}
	result := Result{Checkpoint: checkpoint}
	for _, step := range observed {
		if request.ObserveGRPO != nil {
			request.ObserveGRPO(step)
		} else {
			result.GRPO = append(result.GRPO, step)
		}
	}
	return result, nil
}
