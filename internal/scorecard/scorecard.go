// Package scorecard is the user-facing measure of improvement: for every
// activated model and task, what a user feels -- time to first output,
// throughput, held-out answer quality, failures, load time, peak device
// memory and setup steps -- read from the store's records. A cell no record
// measures carries the reason, so a gap is visible rather than blank.
package scorecard

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// Metric names one scorecard column.
type Metric string

// FirstOutput is how long a user waits for the first output.
const FirstOutput Metric = "first-output"

// Throughput is how fast output arrives once it starts.
const Throughput Metric = "throughput"

// Quality is the headline score on held-out evaluation cases.
const Quality Metric = "held-out-quality"

// Failures is the share of attempts that failed.
const Failures Metric = "failure-rate"

// Load is how long the model takes to load.
const Load Metric = "load-time"

// PeakMemory is the most device memory the model held.
const PeakMemory Metric = "peak-device-memory"

// SetupSteps is how many steps took the model from registration to active.
const SetupSteps Metric = "setup-steps"

// Metrics is the column order: the order a user meets them.
var Metrics = []Metric{FirstOutput, Throughput, Quality, Failures, Load, PeakMemory, SetupSteps}

// catalogLimit bounds the activation listing; a truncated catalog is
// refused rather than read as every model.
const catalogLimit = 4096

// Cell is one measured value with its unit, what it measures and the record
// it came from, or the reason no record measures it.
type Cell struct {
	Value  float64
	Unit   string
	Label  string
	Source artifact.ID
	Reason string
}

// Measured reports whether a record measures the cell.
func (c Cell) Measured() bool { return c.Reason == "" }

// Row is one activated model and task.
type Row struct {
	Model  artifact.ID
	Name   string
	Task   recipe.Task
	Recipe artifact.ID
	Cells  map[Metric]Cell
}

// Build reads the scorecard from the store: one row for every activation
// the capability catalog lists, across every task.
func Build(ctx context.Context, store *overgodb.Store) ([]Row, error) {
	entries, truncated, err := discovery.CapabilityCatalog(ctx, store, catalogLimit, discovery.LoadMemo(ctx, store))
	if err != nil {
		return nil, err
	}
	if truncated {
		return nil, fmt.Errorf("scorecard: the activation listing was truncated at %d", catalogLimit)
	}
	suites, descriptors := evaluation.DerivedSuiteCatalog(ctx, store, evaluation.ListingAuthorities())
	index := evaluation.LatestEvidence(ctx, store, suites, 0)
	serving, err := servingByRecipe(ctx, store)
	if err != nil {
		return nil, err
	}
	names, err := verifiedNames(ctx, store)
	if err != nil {
		return nil, err
	}
	var rows []Row
	for _, entry := range entries {
		name := names[entry.Model]
		if name == "" && entry.Location != "" {
			name = discovery.LocationName(entry.Location)
		}
		name = cmp.Or(name, entry.Model.String())
		benchmark, benchmarked := index.BenchmarksByLocation[entry.Location]
		for _, capability := range entry.Capabilities {
			row := Row{Model: entry.Model, Name: name, Task: capability.Task, Recipe: capability.Recipe, Cells: map[Metric]Cell{}}
			// A benchmark claim measures the weights' inference, so it
			// speaks for the inference activation only.
			measured := benchmarked && capability.Task == recipe.TaskInference
			observed := serving[capability.Recipe]
			row.Cells[FirstOutput] = firstOutputCell(capability.Task, benchmark, measured, observed)
			row.Cells[Throughput] = throughputCell(benchmark, measured, observed)
			row.Cells[Quality] = qualityCell(index.EvaluationsByRecipe[capability.Recipe], descriptors)
			row.Cells[Failures] = failureCell(observed)
			row.Cells[Load] = loadCell(benchmark, measured, observed)
			row.Cells[PeakMemory] = peakMemoryCell(capability.Task, benchmark, measured, observed)
			row.Cells[SetupSteps] = setupCell(ctx, store, entry.Model, capability.Task)
			rows = append(rows, row)
		}
	}
	slices.SortFunc(rows, func(left, right Row) int {
		return cmp.Or(strings.Compare(left.Name, right.Name), strings.Compare(string(left.Task), string(right.Task)))
	})
	return rows, nil
}

// verifiedNames maps each model to the name the verification records give
// it, by the same rule the verification matrix uses.
func verifiedNames(ctx context.Context, store *overgodb.Store) (map[artifact.ID]string, error) {
	var records []runrecord.ModelVerification
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ModelVerificationMediaType, Schema: runrecord.ModelVerificationSchema,
		}},
		// The matrix is order-independent; any one order reads them all.
		Order: overgodb.DocumentOldestFirst,
	}, runrecord.ParseModelVerification, func(_ overgodb.DocumentView, record runrecord.ModelVerification) error {
		records = append(records, record)
		return nil
	})
	names := map[artifact.ID]string{}
	for _, row := range runrecord.VerificationMatrix(records) {
		names[row.Model] = row.Name
	}
	return names, err
}

// servingSummary aggregates the serving attempts recorded under one recipe:
// what users actually hit.
type servingSummary struct {
	attempts, failures, cancelled int
	firstOutputNS, loadNS         []uint64
	decodeTokens, decodeNS        uint64
	peakDeviceBytes               uint64
	latest                        artifact.ID
}

// servingByRecipe reads every serving observation, newest first, into one
// summary per recipe, so a row counts its own activation's attempts and a
// fix shipped as a new recipe starts clean. A cancelled attempt counts as
// neither success nor failure and contributes no samples: the server
// records both a user's cancel and its own deadline that way.
func servingByRecipe(ctx context.Context, store *overgodb.Store) (map[artifact.ID]*servingSummary, error) {
	summaries := map[artifact.ID]*servingSummary{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
		}},
		Order: overgodb.DocumentNewestFirst,
	}, runrecord.ParseServingObservation, func(_ overgodb.DocumentView, observation runrecord.ServingObservation) error {
		summary := summaries[observation.Recipe]
		if summary == nil {
			summary = &servingSummary{latest: observation.ID}
			summaries[observation.Recipe] = summary
		}
		switch observation.Outcome {
		case runrecord.OutcomeSucceeded:
			summary.attempts++
		case runrecord.OutcomeFailed:
			summary.attempts++
			summary.failures++
		default:
			summary.cancelled++
			return nil
		}
		for _, sample := range observation.Hardware {
			if sample.Stage == runrecord.ServingHardwarePrefill {
				summary.firstOutputNS = append(summary.firstOutputNS, sample.ElapsedNS)
			}
			summary.peakDeviceBytes = max(summary.peakDeviceBytes, sample.DevicePeakBytes)
		}
		summary.peakDeviceBytes = max(summary.peakDeviceBytes, observation.Resources.PeakDeviceBytes)
		for _, phase := range observation.Phases {
			switch {
			case phase.Phase == runrecord.PhaseLoad && phase.DurationNS != 0:
				summary.loadNS = append(summary.loadNS, phase.DurationNS)
			case phase.Phase == runrecord.PhaseDecode && phase.DurationNS != 0 && observation.Outcome == runrecord.OutcomeSucceeded:
				// The decode phase alone, as the benchmark's decode rate is.
				summary.decodeTokens += observation.Usage.OutputTokens
				summary.decodeNS += phase.DurationNS
			}
		}
		return nil
	})
	return summaries, err
}

func gap(reason string) Cell { return Cell{Reason: reason} }

func milliseconds(nanoseconds uint64) float64 {
	return float64(nanoseconds) / float64(time.Millisecond)
}

func firstOutputCell(task recipe.Task, benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if task == recipe.TaskTraining {
		// A training session's prefill sample marks its first step, not an
		// answer a user waits for.
		return gap("a training session has no first output")
	}
	if benchmarked && benchmark.FirstTokenMS != 0 {
		return Cell{Value: benchmark.FirstTokenMS, Unit: "ms", Label: "benchmark time to first token, median", Source: benchmark.Record}
	}
	if observed != nil && len(observed.firstOutputNS) != 0 {
		return Cell{Value: milliseconds(runrecord.MedianUint(observed.firstOutputNS)), Unit: "ms",
			Label: fmt.Sprintf("serving time to prefill, median of %d", len(observed.firstOutputNS)), Source: observed.latest}
	}
	return gap("no benchmark time to first token and no serving prefill sample")
}

func throughputCell(benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if benchmarked && benchmark.DecodeTokensPerSecond != 0 {
		return Cell{Value: benchmark.DecodeTokensPerSecond, Unit: "tok/s", Label: "benchmark decode rate, median", Source: benchmark.Record}
	}
	if observed != nil && observed.decodeTokens != 0 && observed.decodeNS != 0 {
		rate := float64(observed.decodeTokens) / time.Duration(observed.decodeNS).Seconds()
		return Cell{Value: rate, Unit: "tok/s", Label: "serving output tokens over decode time", Source: observed.latest}
	}
	return gap("no benchmark decode rate and no served decode phase with output tokens")
}

// qualityCell is the headline metric of the recipe's newest evaluation of
// its largest derived suite: the most cases say the most about held-out
// answers, and the newest run is the current one.
func qualityCell(summaries []evaluation.EvalSummary, descriptors map[string]evaluation.SuiteDescriptor) Cell {
	var best *evaluation.EvalSummary
	for index := range summaries {
		// Summaries arrive newest first; only a larger suite displaces.
		if candidate := &summaries[index]; best == nil || descriptors[candidate.Suite].Cases > descriptors[best.Suite].Cases {
			best = candidate
		}
	}
	if best == nil {
		return gap("no evaluation of a derived benchmark suite")
	}
	name, value, ok := evaluation.HeadlineMetric(best.Metrics)
	if !ok {
		return gap("the latest evaluation reports no metric")
	}
	return Cell{Value: value, Label: fmt.Sprintf("%s, %s (%s)", name, best.Suite, best.Prompting), Source: best.Record}
}

func failureCell(observed *servingSummary) Cell {
	if observed == nil || observed.attempts == 0 {
		return gap("no completed serving attempts recorded")
	}
	return Cell{Value: float64(observed.failures) / float64(observed.attempts),
		Label:  fmt.Sprintf("%d of %d attempts failed, %d cancelled", observed.failures, observed.attempts, observed.cancelled),
		Source: observed.latest}
}

func loadCell(benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if benchmarked && benchmark.LoadMS != 0 {
		return Cell{Value: benchmark.LoadMS, Unit: "ms", Label: "benchmark load", Source: benchmark.Record}
	}
	if observed != nil && len(observed.loadNS) != 0 {
		return Cell{Value: milliseconds(runrecord.MedianUint(observed.loadNS)), Unit: "ms",
			Label: fmt.Sprintf("load phase, median of %d", len(observed.loadNS)), Source: observed.latest}
	}
	return gap("no benchmark load time and no recorded load phase")
}

func peakMemoryCell(task recipe.Task, benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if benchmarked && benchmark.PeakDeviceBytes != 0 {
		return Cell{Value: float64(benchmark.PeakDeviceBytes), Unit: "bytes", Label: "benchmark peak device memory", Source: benchmark.Record}
	}
	if task == recipe.TaskTraining {
		// A training session samples device-wide use, other processes
		// included, not its own peak.
		return gap("training sessions record device-wide memory use, not their own peak")
	}
	if observed != nil && observed.peakDeviceBytes != 0 {
		return Cell{Value: float64(observed.peakDeviceBytes), Unit: "bytes", Label: "serving peak device memory", Source: observed.latest}
	}
	return gap("no benchmark or serving peak device memory")
}

// setupCell counts the lifecycle transitions from registration up to the
// recipe's first activation: each is a step someone ran. Re-verifications
// and rollbacks after that are maintenance, not setup.
func setupCell(ctx context.Context, store *overgodb.Store, model artifact.ID, task recipe.Task) Cell {
	activation, found, err := modelrecipe.ActiveRecord(ctx, store, model, task)
	if err != nil {
		return gap(err.Error())
	}
	if !found {
		return gap("the activation is not recorded")
	}
	chain := []recipe.LifecycleEvent{activation.Event}
	seen := map[artifact.ID]bool{activation.Event.ID: true}
	for previous := activation.Event.PreviousEvent; previous != nil; {
		if seen[*previous] {
			return gap("the lifecycle chain repeats an event")
		}
		seen[*previous] = true
		event, found, err := recipe.ReadLifecycleEvent(ctx, store, *previous)
		if err != nil {
			return gap(err.Error())
		}
		if !found {
			return gap("the lifecycle chain names an absent event")
		}
		chain = append(chain, event)
		previous = event.PreviousEvent
	}
	slices.Reverse(chain)
	steps := slices.IndexFunc(chain, func(event recipe.LifecycleEvent) bool { return event.To == recipe.StatusActive }) + 1
	if steps == 0 {
		return gap("the lifecycle chain never reaches active")
	}
	return Cell{Value: float64(steps), Label: "lifecycle transitions to first activation", Source: activation.Event.ID}
}
