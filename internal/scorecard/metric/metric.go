// Package metric names the user scorecard's columns. It imports nothing, so
// a tool that only speaks of a metric -- the landing that measures a slice,
// a message that predicts one -- does not link the store and the device
// stack the scorecard reads.
package metric

// Name names one scorecard column; a landing message predicts a metric by
// this exact name.
type Name string

// FirstOutput is how long a user waits for the first output.
const FirstOutput Name = "first-output"

// Throughput is how fast output arrives once it starts.
const Throughput Name = "throughput"

// Quality is the headline score on held-out evaluation cases.
const Quality Name = "held-out-quality"

// Failures is the share of attempts that failed.
const Failures Name = "failure-rate"

// Load is how long the model takes to load.
const Load Name = "load-time"

// PeakMemory is the most device memory the model held.
const PeakMemory Name = "peak-device-memory"

// SetupSteps is how many steps took the model from registration to active.
const SetupSteps Name = "setup-steps"

// Columns is the column order: the order a user meets them.
var Columns = []Name{FirstOutput, Throughput, Quality, Failures, Load, PeakMemory, SetupSteps}

// Record is one scorecard slice as the benchmark prints it and a landing's
// trailer carries it. A record with only a slice, an identity and a model
// says what a run selected before it measured, so a crash keeps what it was
// measuring. A measured record carries the outputs and its metrics, a failed
// one the cause and a failure-rate of one; an unmeasured one says why the
// slice could not run. A metric the slice does not measure is absent, never
// zero.
type Record struct {
	Slice      string           `json:"slice,omitempty"`
	Identity   string           `json:"identity,omitempty"`
	Model      string           `json:"model,omitempty"`
	Outputs    string           `json:"outputs,omitempty"`
	Failure    string           `json:"failure,omitempty"`
	Unmeasured string           `json:"unmeasured,omitempty"`
	Metrics    map[Name]float64 `json:"metrics,omitempty"`
}

// VariedOutputs stands for the outputs of a slice whose runs disagreed:
// there is no one output to compare.
const VariedOutputs = "varies"

// Selected reports whether the record only names what a run selected.
func (record Record) Selected() bool {
	return record.Outputs == "" && record.Failure == "" && record.Unmeasured == "" && len(record.Metrics) == 0
}
