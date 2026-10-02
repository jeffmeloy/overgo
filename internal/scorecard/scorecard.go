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
	"overgo/internal/scorecard/metric"
)

// catalogLimit bounds the activation listing; a truncated catalog is
// refused rather than read as every model.
const catalogLimit = 4096

// Cell is one measured value with its unit, what it measures and the record
// it came from, or the reason no record measures it. NotApplicable marks a
// metric the task's output cannot have, or a held-out quality whose
// evaluation is only proposed: its reason says why, and no measurement is
// missing. Direction says which way a held-out quality improves; Contract
// says what the value is, typed, for comparison. Base is a trained model's
// base model value on the same contract, from the held-out verdict that
// judged the two; nil where no verdict judged the value against a base.
type Cell struct {
	Value         float64             `json:"value,omitzero"`
	Base          *float64            `json:"base,omitempty"`
	Unit          string              `json:"unit,omitzero"`
	Label         string              `json:"label,omitzero"`
	Source        artifact.ID         `json:"source,omitzero"`
	Reason        string              `json:"reason,omitzero"`
	NotApplicable bool                `json:"not_applicable,omitzero"`
	Direction     runrecord.Direction `json:"direction,omitzero"`
	Contract      Contract            `json:"contract,omitzero"`
}

// QualityText renders a held-out quality as a user reads it: its value, its
// base model's beside it when a verdict judged it against one, and what it
// measures.
func (c Cell) QualityText() string {
	if c.Base != nil {
		return fmt.Sprintf("%.4g vs base %.4g (%s)", c.Value, *c.Base, c.Label)
	}
	return fmt.Sprintf("%.4g (%s)", c.Value, c.Label)
}

// Contract is what a measured value is, so two cells compare only when both
// measure the same thing: the metric, the kind of record it comes from, and
// for a held-out quality the suite and the protocol it was scored under: the
// prompting, with a transcription report's normalization, or a held-out
// verdict's target plan.
// A serving aggregate's scope is the recipe whose attempts it counts. A cell
// published before contracts has none, and compares with nothing.
type Contract struct {
	Metric   string      `json:"metric"`
	Origin   string      `json:"origin"`
	Suite    string      `json:"suite,omitzero"`
	Protocol string      `json:"protocol,omitzero"`
	Scope    artifact.ID `json:"scope,omitzero"`
}

// The kinds of record a cell's value comes from.
const (
	originBenchmark  = "benchmark"
	originServing    = "serving"
	originEvaluation = "evaluation"
	originLifecycle  = "lifecycle"
)

// wholeOutputTasks return their output whole, an image, a clip or a
// prediction, rather than as a stream of tokens: their time to first output
// is their rate.
var wholeOutputTasks = []recipe.Task{
	recipe.TaskImageGen, recipe.TaskVideoGen, recipe.TaskSpeech, recipe.TaskForecast,
	recipe.TaskTabular, recipe.TaskImageEmbedding, recipe.TaskActivityDetection,
}

// Measured reports whether a record measures the cell.
func (c Cell) Measured() bool { return c.Reason == "" }

// Row is one activated model and task. Architecture is the architecture the
// recipe's profile dependency declares; a recipe that resolves no
// architecture profile, a media or specialized one, leaves it empty.
// Heldout holds the recipe's current held-out qualities besides the
// headline one, each on its own contract: the headline is what a user reads,
// and a baseline compares every one, whichever is the headline since.
type Row struct {
	Model        artifact.ID          `json:"model,omitzero"`
	Name         string               `json:"name"`
	Task         recipe.Task          `json:"task"`
	Recipe       artifact.ID          `json:"recipe,omitzero"`
	Architecture string               `json:"architecture,omitzero"`
	Cells        map[metric.Name]Cell `json:"cells"`
	Heldout      []Cell               `json:"heldout,omitempty"`
}

// Measured lists the row's measured cells of a column: of held-out quality,
// the headline and each other held-out quality, on its own contract; of any
// other metric, its one cell. A baseline row's are what it holds a later
// record to, and a later row's are where each is found again.
func (row Row) Measured(column metric.Name) []Cell {
	cells := []Cell{row.Cells[column]}
	if column == metric.Quality {
		cells = append(cells, row.Heldout...)
	}
	return slices.DeleteFunc(cells, func(cell Cell) bool { return !cell.Measured() })
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
	if err := nameEvaluatedSets(ctx, store, suites); err != nil {
		return nil, err
	}
	index := evaluation.LatestEvidence(ctx, store, suites, 0)
	heldout, err := heldoutRecords(ctx, store, entries)
	if err != nil {
		return nil, err
	}
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
			architecture, err := declaredArchitecture(ctx, store, capability.Recipe)
			if err != nil {
				return nil, err
			}
			row := Row{Model: entry.Model, Name: name, Task: capability.Task, Recipe: capability.Recipe, Architecture: architecture, Cells: map[metric.Name]Cell{}}
			// A benchmark claim measures the weights' inference, so it
			// speaks for the inference activation only.
			measured := benchmarked && capability.Task == recipe.TaskInference
			observed := serving[capability.Recipe]
			row.Cells[metric.FirstOutput] = firstOutputCell(capability.Task, benchmark, measured, observed)
			row.Cells[metric.Throughput] = throughputCell(capability.Task, benchmark, measured, observed)
			row.Cells[metric.Quality], row.Heldout = qualityCell(capability.Task, index.EvaluationsByRecipe[capability.Recipe], descriptors)
			if record, found := heldout[capability.Recipe]; found {
				if circuit := row.Cells[metric.Quality]; circuit.Measured() {
					row.Heldout = append(row.Heldout, circuit)
				}
				row.Cells[metric.Quality] = record.cell(capability.Task)
			}
			row.Cells[metric.Failures] = failureCell(observed)
			row.Cells[metric.Load] = loadCell(benchmark, measured, observed)
			row.Cells[metric.PeakMemory] = peakMemoryCell(capability.Task, benchmark, measured, observed)
			row.Cells[metric.SetupSteps] = setupCell(ctx, store, entry.Model, capability.Task)
			rows = append(rows, row)
		}
	}
	slices.SortFunc(rows, func(left, right Row) int {
		return cmp.Or(strings.Compare(left.Name, right.Name), strings.Compare(string(left.Task), string(right.Task)))
	})
	return rows, nil
}

// declaredArchitecture is the architecture the recipe's profile dependency
// declares. A recipe with no profile dependency, or whose profile is a
// media or specialized contract rather than an architecture profile,
// declares none.
func declaredArchitecture(ctx context.Context, store *overgodb.Store, id artifact.ID) (string, error) {
	definition, err := recipe.RequireDefinition(ctx, store, id)
	if err != nil {
		return "", err
	}
	profileID, ok := definition.PrimaryDependency(recipe.DependencyProfile)
	if !ok {
		return "", nil
	}
	content, found, err := artifact.ReadContent(ctx, store, profileID)
	if err != nil || !found {
		return "", cmp.Or(err, fmt.Errorf("scorecard: recipe %s names an absent profile %s", id, profileID))
	}
	profile, err := modelrecipe.ParseProfileDocument(content.Data)
	if err != nil {
		return "", nil
	}
	return profile.Architecture, nil
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
	attempts, failures, cancelled        int
	firstOutputNS, wholeOutputNS, loadNS []uint64
	decodeTokens, decodeNS               uint64
	peakDeviceBytes                      uint64
	latest, recipe                       artifact.ID
}

// cell is a value aggregated from the summary: the metric over the serving
// attempts of its recipe, read as label says.
func (summary *servingSummary) cell(value float64, unit, metric, label string) Cell {
	return Cell{Value: value, Unit: unit, Label: label, Source: summary.latest, Contract: Contract{Metric: metric, Origin: originServing, Scope: summary.recipe}}
}

// servingByRecipe reads every serving observation, newest first, into one
// summary per recipe, so a row counts its own activation's attempts and a
// fix shipped as a new recipe starts clean. A cancelled attempt counts as
// neither success nor failure and contributes no samples: the server
// records both a user's cancel and its own deadline that way. An
// observation a gate result depends on is a validation run's measurement,
// deliberate interrupts included, not an attempt a user made, and is left
// out: E4B's media resource proof published 324 deliberate failures under
// its projection recipe.
func servingByRecipe(ctx context.Context, store *overgodb.Store) (map[artifact.ID]*servingSummary, error) {
	summaries := map[artifact.ID]*servingSummary{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
		}},
		Order: overgodb.DocumentNewestFirst,
	}, runrecord.ParseServingObservation, func(view overgodb.DocumentView, observation runrecord.ServingObservation) error {
		if validation, err := validationMeasured(ctx, store, view.Content.Descriptor.ID); err != nil || validation {
			return err
		}
		summary := summaries[observation.Recipe]
		if summary == nil {
			summary = &servingSummary{latest: observation.ID, recipe: observation.Recipe}
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
		prefilled := false
		for _, sample := range observation.Hardware {
			if sample.Stage == runrecord.ServingHardwarePrefill {
				summary.firstOutputNS = append(summary.firstOutputNS, sample.ElapsedNS)
				prefilled = true
			}
			summary.peakDeviceBytes = max(summary.peakDeviceBytes, sample.DevicePeakBytes)
		}
		// A capability that returns its output whole, an image or a
		// clip, first outputs when it finishes.
		if !prefilled && observation.Outcome == runrecord.OutcomeSucceeded && observation.MeasuredNS != 0 {
			summary.wholeOutputNS = append(summary.wholeOutputNS, observation.MeasuredNS)
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

// validationMeasured reports whether a gate result depends on the artifact.
func validationMeasured(ctx context.Context, store *overgodb.Store, id artifact.ID) (bool, error) {
	children, err := store.Children(ctx, id)
	if err != nil {
		return false, err
	}
	for _, edge := range children {
		if edge.Relation != artifact.RelationDependsOn {
			continue
		}
		child, found, err := store.Artifact(ctx, edge.Child)
		if err != nil {
			return false, err
		}
		if found && child.Schema == runrecord.GateSchema {
			return true, nil
		}
	}
	return false, nil
}

func gap(reason string) Cell { return Cell{Reason: reason} }

// benchmarkCell is a value the benchmark record measured.
func benchmarkCell(value float64, unit, metric string, record artifact.ID) Cell {
	return Cell{Value: value, Unit: unit, Label: originBenchmark + " " + metric, Source: record, Contract: Contract{Metric: metric, Origin: originBenchmark}}
}

func inapplicable(reason string) Cell { return Cell{Reason: reason, NotApplicable: true} }

func milliseconds(nanoseconds uint64) float64 {
	return float64(nanoseconds) / float64(time.Millisecond)
}

func firstOutputCell(task recipe.Task, benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if task == recipe.TaskTraining {
		// A training session's prefill sample marks its first step, not an
		// answer a user waits for.
		return inapplicable("a training session has no first output")
	}
	if benchmarked && benchmark.FirstTokenMS != 0 {
		return benchmarkCell(benchmark.FirstTokenMS, "ms", "time to first token, median", benchmark.Record)
	}
	if observed != nil && len(observed.firstOutputNS) != 0 {
		return observed.cell(milliseconds(runrecord.MedianUint(observed.firstOutputNS)), "ms", "time to prefill",
			fmt.Sprintf("serving time to prefill, median of %d", len(observed.firstOutputNS)))
	}
	if observed != nil && len(observed.wholeOutputNS) != 0 {
		return observed.cell(milliseconds(runrecord.MedianUint(observed.wholeOutputNS)), "ms", "time to the whole output",
			fmt.Sprintf("serving time to the whole output, median of %d", len(observed.wholeOutputNS)))
	}
	return gap("no benchmark time to first token and no serving prefill sample")
}

func throughputCell(task recipe.Task, benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if benchmarked && benchmark.DecodeTokensPerSecond != 0 {
		return benchmarkCell(benchmark.DecodeTokensPerSecond, "tok/s", "decode rate, median", benchmark.Record)
	}
	if observed != nil && observed.decodeTokens != 0 && observed.decodeNS != 0 {
		rate := float64(observed.decodeTokens) / time.Duration(observed.decodeNS).Seconds()
		return observed.cell(rate, "tok/s", "output tokens over decode time", "serving output tokens over decode time")
	}
	if slices.Contains(wholeOutputTasks, task) {
		return inapplicable(fmt.Sprintf("%s returns its output whole, so its rate is its first-output time", task))
	}
	return gap("no benchmark decode rate and no served decode phase with output tokens")
}

// nameEvaluatedSets adds to suites, by identity, each set the evaluation
// circuit scored a recipe on outside the derived catalog. Its evidence binds
// the evaluation to a plan over the set and its split, so a held-out set of
// another task, such as E4B projection's image, video and audio caption
// matching, is read like a derived suite; a gate's or smoke run's wall time
// has no such evidence.
func nameEvaluatedSets(ctx context.Context, store *overgodb.Store, suites map[artifact.ID]string) error {
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: evaluation.EvaluationEvidenceMediaType, Schema: evaluation.EvaluationEvidenceSchema,
		}},
		// Naming is order-independent; any one order reads them all.
		Order: overgodb.DocumentOldestFirst,
	}, evaluation.ParseEvaluationEvidence, func(_ overgodb.DocumentView, evidence evaluation.EvaluationEvidence) error {
		if _, named := suites[evidence.Dataset]; !named {
			suites[evidence.Dataset] = evidence.Dataset.String()
		}
		return nil
	})
	return err
}

// verdictTasks are the served tasks whose held-out quality a held-out
// verdict's base model measures: a table-prediction or forecast objective
// scores the served head's own predictions. An inference recipe runs the
// derived suites instead.
var verdictTasks = []recipe.Task{recipe.TaskTabular, recipe.TaskForecast}

// heldoutRecord is the held-out record that scores a recipe's own output
// and, for a trained model's, the same verdict's value of the same metric
// for its base model: one contract, so the two compare.
type heldoutRecord struct {
	summary evaluation.EvalSummary
	base    *float64
}

// cell is the record's held-out quality, beside its base model's value.
func (record heldoutRecord) cell(task recipe.Task) Cell {
	cell, _ := qualityCell(task, []evaluation.EvalSummary{record.summary}, nil)
	cell.Base = record.base
	return cell
}

// heldoutRecords reads, by recipe, the held-out records that score a
// recipe's own output: the newest transcription report of each recipe, and
// each held-out verdict, newest first, as the trained model's value, beside
// its base model's, for the training recipe its session ran and the base
// model's value for each tabular or forecast recipe whose model holds the
// weights the session trained. A model holds them when its manifest names a
// weights component of the same bytes: the session names a head's weights
// file, the row the model directory around it. Where one exists it is the
// recipe's held-out quality, and each set the evaluation circuit scored the
// recipe on stays among its other held-out qualities, so a baseline measured
// on one still compares.
func heldoutRecords(ctx context.Context, store *overgodb.Store, entries []discovery.CatalogEntry) (map[artifact.ID]heldoutRecord, error) {
	reports, err := evaluation.TranscriptionQualities(ctx, store)
	if err != nil {
		return nil, err
	}
	records := make(map[artifact.ID]heldoutRecord, len(reports))
	for recipeID, report := range reports {
		records[recipeID] = heldoutRecord{summary: report}
	}
	served := map[string][]artifact.ID{}
	for _, entry := range entries {
		for _, capability := range entry.Capabilities {
			if !slices.Contains(verdictTasks, capability.Task) {
				continue
			}
			manifest, _, err := store.Manifest(ctx, entry.Model)
			if err != nil {
				return nil, err
			}
			for _, component := range manifest.Components {
				if component.Role == artifact.ComponentWeights {
					served[component.Artifact.DigestHex()] = append(served[component.Artifact.DigestHex()], capability.Recipe)
				}
			}
		}
	}
	err = evaluation.VisitHeldoutVerdicts(ctx, store, func(observation artifact.ID, base, trained evaluation.EvalSummary) error {
		session, err := runrecord.RequireServingObservation(ctx, store, observation)
		if err != nil {
			return err
		}
		read := func(recipeID artifact.ID, record heldoutRecord) {
			if _, found := records[recipeID]; !found {
				records[recipeID] = record
			}
		}
		_, value, _ := evaluation.HeadlineMetric(base.Metrics)
		read(session.Recipe, heldoutRecord{summary: trained, base: &value})
		for _, recipeID := range served[session.Model.DigestHex()] {
			read(recipeID, heldoutRecord{summary: base})
		}
		return nil
	})
	return records, err
}

// heldoutProposals names, for each task no held-out evaluation measures yet,
// the held-out set and metric proposed to measure it. Such a row's quality
// is not applicable rather than missing: it withholds no promotion, and the
// loop proposes defining the evaluation. A task's entry goes when its
// held-out evaluation lands; inference has none, since every inference
// recipe runs the derived suites.
var heldoutProposals = map[recipe.Task]string{
	recipe.TaskGeneration:     "WikiText-2 test (Salesforce/wikitext wikitext-2-raw-v1, 0.7 MB) perplexity, as a sequence-scoring suite",
	recipe.TaskTraining:       "a token-prediction objective whose split commits a held-out membership of WikiText-2 (on disk), judged by train -objective",
	recipe.TaskProjection:     "the image, video, multi-image and mixed caption-matching sets the store already scores a projection on",
	recipe.TaskVQA:            "the image caption-matching set the store already scores a projection on, then A-OKVQA validation (HuggingFaceM4/A-OKVQA, 60 MB) multiple-choice accuracy",
	recipe.TaskImageGen:       "TIFA v1.0 (github.com/Yushi-Hu/tifa, 10 MB) question answering over each image by a local vision-language model",
	recipe.TaskVideoGen:       "held-out VidGen-1M captions (on disk), each clip matched to its caption by a local vision-language model",
	recipe.TaskTranscription:  "LibriSpeech test-clean (on disk) word error rate, as evaluate's transcription report",
	recipe.TaskSpeech:         "LibriSpeech test-clean texts (on disk), synthesized then transcribed by a local speech recognizer: round-trip word error rate",
	recipe.TaskSeq2Seq:        "BFCL v3 simple (gorilla-llm/Berkeley-Function-Calling-Leaderboard, 0.3 MB) exact match of the parsed call",
	recipe.TaskTabular:        "a table-prediction objective over BeyondArena credit-g (TabArena/BeyondArena, 0.2 MB), its held-out rows judged by train -route tabular",
	recipe.TaskForecast:       "a forecast objective over ETTh1 (github.com/zhouhaoyi/ETDataset, 2.6 MB), its held-out windows judged by train -route seriesforecast",
	recipe.TaskImageEmbedding: "Imagenette 160 px (fast.ai imagenette2-160, 100 MB) k-nearest-neighbour accuracy from the model's own embeddings",
}

// qualityCell is the headline metric of the recipe's newest evaluation of
// its largest held-out suite: the most cases say the most about held-out
// answers, and the newest run is the current one. A set outside the derived
// catalog counts no cases, so of two equal suites the one named first, then
// the protocol named first, is read, the same in every record. It also
// returns the recipe's other current held-out qualities, one per suite and
// protocol. A recipe no suite measures reads not applicable when its task
// has a held-out evaluation proposed, else a gap.
func qualityCell(task recipe.Task, summaries []evaluation.EvalSummary, descriptors map[string]evaluation.SuiteDescriptor) (Cell, []Cell) {
	type heldout struct {
		cases uint64
		cell  Cell
	}
	type suite struct{ name, protocol string }
	var measured []heldout
	seen := map[suite]bool{}
	for _, summary := range summaries {
		// Summaries arrive newest first, so the first of each suite and
		// protocol is its current evaluation.
		if seen[suite{summary.Suite, summary.Prompting}] {
			continue
		}
		seen[suite{summary.Suite, summary.Prompting}] = true
		if name, value, ok := evaluation.HeadlineMetric(summary.Metrics); ok {
			measured = append(measured, heldout{cases: descriptors[summary.Suite].Cases, cell: Cell{
				Value: value, Label: fmt.Sprintf("%s, %s (%s)", name, summary.Suite, summary.Prompting), Source: summary.Record, Direction: summary.Directions[name],
				Contract: Contract{Metric: name, Origin: originEvaluation, Suite: summary.Suite, Protocol: summary.Prompting},
			}})
		}
	}
	switch {
	case len(measured) != 0:
	case len(summaries) != 0:
		return gap("the latest evaluation reports no metric"), nil
	case heldoutProposals[task] != "":
		return inapplicable(fmt.Sprintf("no held-out evaluation for %s yet; proposed: %s", task, heldoutProposals[task])), nil
	default:
		return gap("no held-out evaluation of this recipe"), nil
	}
	slices.SortFunc(measured, func(left, right heldout) int {
		return cmp.Or(cmp.Compare(right.cases, left.cases), strings.Compare(left.cell.Contract.Suite, right.cell.Contract.Suite),
			strings.Compare(left.cell.Contract.Protocol, right.cell.Contract.Protocol))
	})
	others := make([]Cell, 0, len(measured)-1)
	for _, other := range measured[1:] {
		others = append(others, other.cell)
	}
	return measured[0].cell, others
}

func failureCell(observed *servingSummary) Cell {
	if observed == nil || observed.attempts == 0 {
		return gap("no completed serving attempts recorded")
	}
	return observed.cell(float64(observed.failures)/float64(observed.attempts), "", "failed share of attempts",
		fmt.Sprintf("%d of %d attempts failed, %d cancelled", observed.failures, observed.attempts, observed.cancelled))
}

func loadCell(benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if benchmarked && benchmark.LoadMS != 0 {
		return benchmarkCell(benchmark.LoadMS, "ms", "load", benchmark.Record)
	}
	if observed != nil && len(observed.loadNS) != 0 {
		return observed.cell(milliseconds(runrecord.MedianUint(observed.loadNS)), "ms", "load phase",
			fmt.Sprintf("load phase, median of %d", len(observed.loadNS)))
	}
	return gap("no benchmark load time and no recorded load phase")
}

func peakMemoryCell(task recipe.Task, benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if benchmarked && benchmark.PeakDeviceBytes != 0 {
		return benchmarkCell(float64(benchmark.PeakDeviceBytes), "bytes", "peak device memory", benchmark.Record)
	}
	if task == recipe.TaskTraining {
		// A training session samples device-wide use, other processes
		// included, not its own peak.
		return gap("training sessions record device-wide memory use, not their own peak")
	}
	if observed != nil && observed.peakDeviceBytes != 0 {
		return observed.cell(float64(observed.peakDeviceBytes), "bytes", "peak device memory", "serving peak device memory")
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
	return Cell{Value: float64(steps), Label: "lifecycle transitions to first activation", Source: activation.Event.ID,
		Contract: Contract{Metric: "transitions to first activation", Origin: originLifecycle}}
}
