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
