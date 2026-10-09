package capabilityruntime

import (
	"context"
	"crypto/rand"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

type freshRunContextKey struct{}

// FreshRun gives an intentional measurement its own operation scope.
// Reusing this context for the same request recovers that operation's
// stages; another FreshRun executes independently. Ordinary request
// identity, activation evidence and workflow recovery are unchanged.
func FreshRun(ctx context.Context) context.Context {
	return context.WithValue(ctx, freshRunContextKey{}, rand.Text())
}

// Measured wraps a capability output with the run's measured
// decomposition: the persisted workflow run, the per-node phase walls the
// runtime observed, how long the run's own model load took, the peak device
// bytes the loaded model session reports, and, of a run whose model decoded
// its output token by token, how long that decode took and how many tokens
// it gave. Absent measurements stay zero; nothing is invented: a run a
// resident session served loaded nothing, and a run that recovered its
// output from the store, or whose output is whole, decoded nothing.
// Input is the request document the runtime recorded for the run (the
// decoded request under its capability's input schema), so a consumer
// publishing its own run record cites the one document the store holds.
type Measured struct {
	Output          any                        `json:"output"`
	Run             artifact.ID                `json:"run,omitzero"`
	Phases          []workflowruntime.NodeWall `json:"phases,omitempty"`
	Load            time.Duration              `json:"load_ns,omitzero"`
	PeakDeviceBytes uint64                     `json:"peak_device_bytes,omitzero"`
	OutputTokens    uint64                     `json:"output_tokens,omitzero"`
	Decode          time.Duration              `json:"decode_ns,omitzero"`
	Input           artifact.Content           `json:"-"`
}

// Unwrap returns the output inside a measured envelope, or the value
// itself when it is not enveloped.
func Unwrap(output any) any {
	if measured, ok := output.(Measured); ok {
		return measured.Output
	}
	return output
}

// deviceMeter is implemented by model runtimes whose device sessions
// account allocations; hosts without device accounting simply do not
// implement it.
type deviceMeter interface {
	PeakDeviceBytes() (uint64, error)
}

// tokenMeter is implemented by model runtimes that decode their output
// token by token: the number of tokens the model's last run decoded. A
// runtime whose output is whole, an image or a prediction, does not
// implement it.
type tokenMeter interface {
	OutputTokens() uint64
}

// measure assembles the envelope around an executed output, with the wall
// of the load the run made for it. A meter error yields an absent peak
// rather than failing a run that already produced its output. A token
// runtime's decode is the wall of the node the program's output comes from,
// the stage that selected the tokens, and is read with the count only when
// this run executed that node: a run that recovered its output decoded
// nothing, and the count a resident model holds is then an earlier run's.
func measure[Model any](output any, model Model, result workflowruntime.Result, load time.Duration, program recipe.Program) Measured {
	measured := Measured{Output: output, Run: result.Run.ID, Phases: result.NodeWalls, Load: load}
	if meter, ok := any(model).(deviceMeter); ok {
		if peak, err := meter.PeakDeviceBytes(); err == nil {
			measured.PeakDeviceBytes = peak
		}
	}
	if meter, ok := any(model).(tokenMeter); ok {
		decoder := program.Definition().Outputs[0].Source.Node
		for _, phase := range result.NodeWalls {
			if phase.Node == decoder {
				measured.OutputTokens, measured.Decode = meter.OutputTokens(), time.Duration(phase.WallNS)
			}
		}
	}
	return measured
}

// ExecuteMeasured runs a program with one typed output and returns the
// persisted workflow result alongside it.
func ExecuteMeasured[Output any](
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	key string,
	inputs map[recipe.PortName]workflowruntime.Value,
	bind func(*workflowruntime.Runtime) error,
) (Output, workflowruntime.Result, error) {
	value, result, _, err := measuredRun[Output](ctx, store, modelID, program, key, inputs, bind)
	return value, result, err
}

// measuredRun runs a program under the operation its key and the context's
// scope name, with the adapters bind registers. Given no bind it only
// replays: it answers the request from the store when the store holds its
// whole answer, and reports false when the request must be executed. No model
// is bound for a replay, because no stage runs, so a caller asks for one
// before it loads a model, and a request already answered costs neither a
// load nor a stage.
func measuredRun[Output any](
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	key string,
	inputs map[recipe.PortName]workflowruntime.Value,
	bind func(*workflowruntime.Runtime) error,
) (Output, workflowruntime.Result, bool, error) {
	var zero Output
	var result workflowruntime.Result
	definition := program.Definition()
	if definition.Model != modelID {
		return zero, result, false, errorProgramModelDiffers
	}
	if len(definition.Outputs) != 1 {
		return zero, result, false, errorProgramOutputArity
	}
	runtime, err := workflowruntime.NewForProgram(store, program)
	if err != nil {
		return zero, result, false, err
	}
	if bind != nil {
		if err := bind(runtime); err != nil {
			return zero, result, false, err
		}
	}
	if ctx != nil {
		if scope, ok := ctx.Value(freshRunContextKey{}).(string); ok {
			key += "/measurement/" + scope
		}
	}
	operation, err := workflowruntime.ExecutionID(definition.ID, key)
	if err != nil {
		return zero, result, false, err
	}
	held := true
	if bind == nil {
		result, held, err = runtime.Replay(ctx, key, operation, program, inputs)
	} else {
		result, err = runtime.ExecuteProgram(ctx, key, operation, nil, program, inputs)
	}
	if err != nil || !held {
		return zero, result, false, err
	}
	value, err := singleOutput[Output](definition, result)
	return value, result, err == nil, err
}
