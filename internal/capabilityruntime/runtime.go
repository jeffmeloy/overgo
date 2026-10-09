// Package capabilityruntime executes verified model capabilities: it places
// a selection on the exact local or peer implementation that may run it and
// binds compiled recipe entry modules to their executors.
package capabilityruntime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

type Executor func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error)

// ExecutorCatalog binds compiled entry modules to implementations.
type ExecutorCatalog map[recipe.ModuleID]Executor

// Execute selects execution from the compiled recipe entry module.
func (catalog ExecutorCatalog) Execute(
	ctx context.Context,
	store artifact.Repository,
	path string,
	execution modelrecipe.CapabilityEvidenceSelection,
	raw string,
) (any, error) {
	program := execution.Program
	stages := program.Stages()
	if len(stages) == 0 {
		return nil, errors.New("capability runtime: compiled program has no entry module")
	}
	entry := stages[0].Module.ID
	execute := catalog[entry]
	if execute == nil {
		return nil, fmt.Errorf("capability runtime: entry module %q has no executor", entry)
	}
	return execute(ctx, store, path, execution, raw)
}

// besideContextKey carries the state of the files a model reads beside its
// inventory to the key its run is kept under.
type besideContextKey struct{}

// Reading returns an executor whose answers are kept under the files its
// model reads beside its inventory as well: a text encoder, a tokenizer, an
// image processor, which files names from the model's path. A model's
// identity is its inventory's, so a file outside it can change with no
// identity moving, and the answer it shaped would be given again. Each file is
// taken as a change to it shows, its place, size and time, since a checkpoint
// of gigabytes cannot be digested for every request; one that is absent is
// recorded so.
func Reading(execute Executor, files func(path string) []string) Executor {
	return func(ctx context.Context, store artifact.Repository, path string, execution modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
		state := sha256.New()
		for _, file := range files(path) {
			if info, err := os.Stat(file); err == nil {
				fmt.Fprintf(state, "%s %d %d\n", file, info.Size(), info.ModTime().UnixNano())
			} else {
				fmt.Fprintf(state, "%s absent\n", file)
			}
		}
		return execute(context.WithValue(ctx, besideContextKey{}, hex.EncodeToString(state.Sum(nil))), store, path, execution, raw)
	}
}

func JSONScalar[Input, Model, Output any](
	name string,
	validate func(Input) error,
	load func(context.Context, artifact.Repository, string, recipe.Program, Input) (Model, error),
	bind func(*workflowruntime.Runtime, artifact.ID, Model) error,
) Executor {
	return func(
		ctx context.Context,
		store artifact.Repository,
		path string,
		execution modelrecipe.CapabilityEvidenceSelection,
		raw string,
	) (any, error) {
		modelID, program := execution.Program.Definition().Model, execution.Program
		input, content, err := decodeScalarInput(name, validate, modelID, program, raw)
		if err != nil {
			return nil, err
		}
		// A request the store already answers is answered from it: no model is loaded.
		inputPort := program.Definition().Inputs[0]
		if output, result, held, err := measuredRun[Output](ctx, store, modelID, program, RunKeyOf[Model](program.Definition(), content),
			map[recipe.PortName]workflowruntime.Value{inputPort.Name: workflowruntime.ArtifactValue(inputPort.Data, input, content)}, nil,
		); err != nil || held {
			return Measured{Output: output, Run: result.Run.ID, Input: content}, err
		}
		clock := processmeasure.NewStopwatch()
		model, err := load(ctx, store, path, program, input)
		if err != nil {
			return nil, err
		}
		// A wall the clock could not read stays zero: absent, not guessed.
		loaded, _ := clock.Elapsed()
		output, result, executeErr := executeScalar[Input, Model, Output](
			ctx, store, modelID, program, content, input, model, bind,
		)
		measured := Measured{Run: result.Run.ID}
		if executeErr == nil {
			measured = measure(output, model, result, time.Duration(loaded), program)
		}
		measured.Input = content
		executeErr = errors.Join(executeErr, closeModel(context.WithoutCancel(ctx), model))
		return measured, executeErr
	}
}

func IgnoreInput[Input, Model any](load func(string) (Model, error)) func(context.Context, artifact.Repository, string, recipe.Program, Input) (Model, error) {
	return func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ Input) (Model, error) {
		return load(path)
	}
}

var (
	errorProgramModelDiffers = errors.New("capability runtime: program model differs from binding")
	errorProgramOutputArity  = errors.New("capability runtime: program requires one output")
)

// Execute binds adapters and runs a program with one typed output.
func Execute[Output any](
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	key string,
	inputs map[recipe.PortName]workflowruntime.Value,
	bind func(*workflowruntime.Runtime) error,
) (Output, error) {
	value, _, err := ExecuteMeasured[Output](ctx, store, modelID, program, key, inputs, bind)
	return value, err
}

// ContentDecoder is an output type that rebuilds itself from its recorded
// artifact: a request the store already answered replays from the
// recorded document, and a media output (PNG, GIF, WAV) is not JSON.
type ContentDecoder interface {
	DecodeContent(artifact.Content) error
}

// singleOutput unwraps the program's one output datum into its typed
// value, decoding recorded content when the value is not resident: JSON
// documents by decoding, media documents through the type's own decoder.
func singleOutput[Output any](definition recipe.Definition, result workflowruntime.Result) (Output, error) {
	var zero Output
	output := definition.Outputs[0]
	datum, ok := result.Outputs[output.Name].Single()
	if !ok {
		return zero, fmt.Errorf("capability runtime: output %q is not scalar", output.Name)
	}
	value, ok := datum.Value.(Output)
	if ok {
		return value, nil
	}
	if datum.Content != nil {
		if decoder, ok := any(&zero).(ContentDecoder); ok {
			if err := decoder.DecodeContent(*datum.Content); err == nil {
				return zero, nil
			}
		}
		if err := strictjson.DecodeBytes(datum.Content.Data, &zero); err == nil {
			return zero, nil
		}
	}
	return zero, fmt.Errorf("capability runtime: output %q has invalid value type", output.Name)
}
