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
// decomposition: the per-node phase walls the workflow runtime
// observed, how long the run's own model load took, and the peak device
// bytes the loaded model session reports. Absent measurements stay zero;
// nothing is invented: a run a resident session served loaded nothing.
// Input is the request document the runtime recorded for the run (the
// decoded request under its capability's input schema), so a consumer
// publishing its own run record cites the one document the store holds.
type Measured struct {
	Output          any                        `json:"output"`
	Phases          []workflowruntime.NodeWall `json:"phases,omitempty"`
	Load            time.Duration              `json:"load_ns,omitzero"`
	PeakDeviceBytes uint64                     `json:"peak_device_bytes,omitzero"`
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

// measure assembles the envelope around an executed output, with the wall
// of the load the run made for it. A meter error yields an absent peak
// rather than failing a run that already produced its output.
func measure[Model any](output any, model Model, phases []workflowruntime.NodeWall, load time.Duration) Measured {
	measured := Measured{Output: output, Phases: phases, Load: load}
	if meter, ok := any(model).(deviceMeter); ok {
		if peak, err := meter.PeakDeviceBytes(); err == nil {
			measured.PeakDeviceBytes = peak
		}
	}
	return measured
}

// ExecuteMeasured runs a program with one typed output and returns the
// per-node walls alongside it.
func ExecuteMeasured[Output any](
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	key string,
	inputs map[recipe.PortName]workflowruntime.Value,
	bind func(*workflowruntime.Runtime) error,
) (Output, []workflowruntime.NodeWall, error) {
	var zero Output
	definition := program.Definition()
	if definition.Model != modelID {
		return zero, nil, errorProgramModelDiffers
	}
	if len(definition.Outputs) != 1 {
		return zero, nil, errorProgramOutputArity
	}
	runtime, err := workflowruntime.NewForProgram(store, program)
	if err != nil {
		return zero, nil, err
	}
	if err := bind(runtime); err != nil {
		return zero, nil, err
	}
	if ctx != nil {
		if scope, ok := ctx.Value(freshRunContextKey{}).(string); ok {
			key += "/measurement/" + scope
		}
	}
	operation, err := workflowruntime.ExecutionID(definition.ID, key)
	if err != nil {
		return zero, nil, err
	}
	result, err := runtime.ExecuteProgram(ctx, key, operation, nil, program, inputs)
	if err != nil {
		return zero, nil, err
	}
	value, err := singleOutput[Output](definition, result)
	return value, result.NodeWalls, err
}
