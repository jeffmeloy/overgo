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
	"math"
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
	"overgo/internal/tensorstats"
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
	Value         float64     `json:"value,omitzero"`
	Base          *float64    `json:"base,omitempty"`
	Unit          string      `json:"unit,omitzero"`
	Label         string      `json:"label,omitzero"`
	Source        artifact.ID `json:"source,omitzero"`
	CodeCommit    string      `json:"code_commit,omitzero"`
	Environment   artifact.ID `json:"environment,omitzero"`
	Reason        string      `json:"reason,omitzero"`
	NotApplicable bool        `json:"not_applicable,omitzero"`
	// Proposed marks a held-out quality that is not applicable only because its
	// task's held-out evaluation is proposed and not yet defined: the one case that
	// is work to do, and the one the cycle files a row for.
	Proposed  bool                `json:"proposed,omitzero"`
	Direction runrecord.Direction `json:"direction,omitzero"`
	Contract  Contract            `json:"contract,omitzero"`
	// Attempts is the denominator of a failed share: the attempts it counts, so
	// two shares compare as counts and not as rates over unlike samples.
	Attempts int `json:"attempts,omitzero"`
	// Spread is how far the recent measured attempts a serving timing is the median of stand from it:
	// their median distance from that median, as a share of it, which one attempt far from the others
	// does not widen. It is stated beside the reading and judges nothing: what a later record's cell
	// is allowed is its reference's, from the cell's earlier readings (Reference). Zero where one
	// attempt measured it.
	Spread float64 `json:"spread,omitzero"`
	// Cases is how many cases a held-out quality rests on: a suite's cases, a
	// held-out verdict's records, a transcription report's utterances. Zero
	// where no record counts them.
	Cases uint64 `json:"cases,omitzero"`
	// Noise is the standard error supplied or estimated from the measurement's records.
	// Share estimates use a binomial model, whose independence is not established by case counts.
	// Verdict means use record scores and pooled word-error rates use utterance observations.
	// Zero means no estimate or an estimated error of zero.
	// Supported uses Noise where per-case outcomes are absent.
	Noise float64 `json:"noise,omitzero"`
	// Constant is what an answer that reads no input scores on a held-out
	// quality's measure, where the measure has one: chance for a choice among a
	// case's alternatives, the commonest label or class for label accuracy. Nil
	// where it has none, as a judged generation has: there another model answers
	// about what this one drew.
	Constant *float64 `json:"constant,omitempty"`
	// Compared names the test that compares the path a request takes with a
	// reference output, for a held-out quality that path produced, and
	// Uncompared says why no such comparison is made where one was decided
	// against. A reading with neither has met no reference: a fault in its
	// path would read as its score.
	Compared   string `json:"compared,omitzero"`
	Uncompared string `json:"uncompared,omitzero"`
	// OutsideUse says what a held-out quality measures where that is not what
	// its model is for by the model's own documentation. Empty where the
	// reading is of the model's use, or no use is recorded for it.
	OutsideUse string `json:"outside_use,omitzero"`
	// Measure names what a held-out quality from an evaluation was scored on: a
	// catalogued suite by its name, a set the evaluation circuit scored by the
	// source its builder gave it and its cases, which a set built again shares.
	Measure string `json:"measure,omitzero"`
	// Unread is how many of the answers a held-out quality rests on named no
	// option the evaluation could read, where the answers were generated: each
	// is counted wrong, so a reading with many says as much of how answers are
	// read as of what the model knows.
	Unread uint64 `json:"unread,omitzero"`
	// Prompt is the length, in tokens, of the fresh prompts a first output
	// read at a longer prompt was timed at (Row.Lengths); zero for every other
	// cell.
	Prompt int `json:"prompt,omitzero"`
	// allowed and unjudged are a reference's alone (Reference) and are never
	// published: how far, in the cell's unit, a later reading may stand worse
	// than the reference's value, and why nothing is judged against it, said
	// where its earlier readings are too few to say what the cell holds.
	allowed  float64
	unjudged string
}

// QualityText renders a held-out quality as a user reads it: its value, its
// base model's beside it when a verdict judged it against one, the noise of
// the value and the cases it rests on where a record gives them, how many of
// its answers named no option where some did not, what a constant answer
// scores where the measure has one, in words when the value is no better than
// it, that it is not a measure of what the model is for where that is
// recorded, and what it measures.
func (c Cell) QualityText() string {
	text := fmt.Sprintf("%.4g", c.Value)
	if c.Noise > 0 {
		text += fmt.Sprintf(" ±%.2g", c.Noise)
	}
	if c.Base != nil {
		text += fmt.Sprintf(" vs base %.4g", *c.Base)
	}
	if c.Cases > 0 {
		text += fmt.Sprintf(" on %d cases", c.Cases)
		if c.Contract.Metric == shareMetric {
			switch c.Value {
			case 1:
				text += ", every one right, at the observed ceiling; generalization is unknown"
			case 0:
				text += ", none right; generalization is unknown"
			}
		}
	}
	if c.Unread > 0 {
		text += fmt.Sprintf(", %d of them answered with no option named and counted wrong", c.Unread)
	}
	if c.Constant != nil {
		advantage, directed := c.Direction.Advantage(c.Value, *c.Constant)
		switch {
		case !worse(c.Direction, *c.Constant, c.Value, 0):
			text += fmt.Sprintf(", no better than the %.4g a constant answer scores", *c.Constant)
		case directed && advantage < noiseDeviates*c.Noise:
			// Better by less than its own noise tells nothing from a constant answer.
			text += fmt.Sprintf(", within its noise of the %.4g a constant answer scores", *c.Constant)
		default:
			text += fmt.Sprintf(", a constant answer scores %.4g", *c.Constant)
		}
	}
	if c.OutsideUse != "" {
		text += ", not a measure of what the model is for"
	}
	return text + " (" + c.Label + ")"
}

// shareMetric is the metric a suite reports as the share of its cases a model
// answered right, whose standard error follows from the share and the cases.
const shareMetric = "accuracy"

// shareError is the standard error of a share of cases answered right.
func shareError(share float64, cases uint64) float64 {
	if cases == 0 || share < 0 || share > 1 {
		return 0
	}
	return math.Sqrt(share * (1 - share) / float64(cases))
}

// noiseDeviates is how many of its own standard errors a reading must stand
// above what a constant answer scores before it is read as better than one,
// and how far a held-out quality must move before its decline stands (Supported).
const noiseDeviates = 2

// Contract is what a measured value is, so two cells compare only when both
// measure the same thing: the metric, the kind of record it comes from, and
// for a held-out quality the suite and the protocol it was scored under: the
// prompting, with a transcription report's normalization, or a held-out
// verdict's target plan. Comparison binds evaluation acceptance and environment
// or a benchmark's measured execution contract;
// a display label alone cannot establish those authorities.
// A serving aggregate's scope is the recipe whose attempts it counts. A cell
// published before contracts has none, and compares with nothing.
type Contract struct {
	Metric     string      `json:"metric"`
	Origin     string      `json:"origin"`
	Suite      string      `json:"suite,omitzero"`
	Protocol   string      `json:"protocol,omitzero"`
	Comparison artifact.ID `json:"comparison,omitzero"`
	Scope      artifact.ID `json:"scope,omitzero"`
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
	recipe.TaskTabular, recipe.TaskImageEmbedding, recipe.TaskReconstruction, recipe.TaskActivityDetection, recipe.TaskTranscription,
}

// Measured reports whether a record measures the cell.
func (c Cell) Measured() bool { return c.Reason == "" }

// Row is one activated model and task. Architecture is the architecture the
// recipe's profile dependency declares; a recipe that resolves no
// architecture profile, a media or specialized one, leaves it empty.
// Heldout holds the recipe's current held-out qualities besides the
// headline one, each on its own contract: the headline is what a user reads,
// and a baseline compares every one, whichever is the headline since.
// Lengths holds a chat model's first output at each longer prompt it was
// timed at (evaluation.PromptLengths), each on its own contract, shortest
// first, beside the headline's at its verified request.
type Row struct {
	Model        artifact.ID          `json:"model,omitzero"`
	Name         string               `json:"name"`
	Task         recipe.Task          `json:"task"`
	Recipe       artifact.ID          `json:"recipe,omitzero"`
	Architecture string               `json:"architecture,omitzero"`
	Cells        map[metric.Name]Cell `json:"cells"`
	Heldout      []Cell               `json:"heldout,omitempty"`
	Lengths      []Cell               `json:"lengths,omitempty"`
}

// Measured lists the row's measured cells of a column: its one cell, then
// the others it holds of the column, each on its own contract: of held-out
// quality, every other held-out quality, and of first output, each longer
// prompt's. A baseline row's are what it holds a later record to, and a
// later row's are where each is found again.
func (row Row) Measured(column metric.Name) []Cell {
	cells := []Cell{row.Cells[column]}
	if others := row.others(column); others != nil {
		cells = append(cells, *others...)
	}
	return slices.DeleteFunc(cells, func(cell Cell) bool { return !cell.Measured() })
}

// others are the cells the row holds of a column besides its one cell, nil
// for a column that has none.
func (row *Row) others(column metric.Name) *[]Cell {
	switch column {
	case metric.Quality:
		return &row.Heldout
	case metric.FirstOutput:
		return &row.Lengths
	}
	return nil
}

// Build reads the scorecard from the store: one row for every activation
// the capability catalog lists, across every task.
func Build(ctx context.Context, store *overgodb.Store) ([]Row, error) {
	// The registered weights this pass hashes persist, so the next build answers from stat checks, as
	// the capability census does, instead of reading every model's bytes again: about 21 of the
	// 26 seconds of a build on the real store, 2026-10-05.
	entries, truncated, err := discovery.RememberedCapabilityCatalog(ctx, store, catalogLimit)
	if err != nil {
		return nil, err
	}
	if truncated {
		return nil, fmt.Errorf("scorecard: the activation listing was truncated at %d", catalogLimit)
	}
	suites, descriptors := evaluation.DerivedSuiteCatalog(ctx, store, evaluation.ListingAuthorities())
	if err := nameEvaluatedSets(ctx, store, suites, descriptors); err != nil {
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
			row.Lengths = lengthCells(capability.Task, observed)
			row.Cells[metric.Throughput] = throughputCell(capability.Task, benchmark, measured, observed)
			row.Cells[metric.Quality], row.Heldout = qualityCell(capability.Task, index.EvaluationsByRecipe[capability.Recipe], descriptors)
			if record, found := heldout[capability.Recipe]; found {
				if circuit := row.Cells[metric.Quality]; circuit.Measured() {
					row.Heldout = append(row.Heldout, circuit)
				}
				row.Cells[metric.Quality] = record.cell(capability.Task)
			}
			if quality := row.Cells[metric.Quality]; quality.Proposed && (capability.Task == recipe.TaskImageGen || capability.Task == recipe.TaskVideoGen) {
				definition, err := recipe.RequireDefinition(ctx, store, capability.Recipe)
				if err != nil {
					return nil, err
				}
				row.Cells[metric.Quality] = unmeasuredGeneration(quality, definition)
			}
			row.Cells[metric.Failures] = failureCell(observed)
			row.Cells[metric.Load] = loadCell(benchmark, measured, observed)
			row.Cells[metric.PeakMemory] = peakMemoryCell(benchmark, measured, observed)
			row.Cells[metric.SetupSteps] = setupCell(ctx, store, entry.Model, capability.Task)
			stateNotApplicable(&row)
			rows = append(rows, row)
		}
	}
	slices.SortFunc(rows, func(left, right Row) int {
		return cmp.Or(strings.Compare(left.Name, right.Name), strings.Compare(string(left.Task), string(right.Task)))
	})
	// The uses choose which of a row's readings is read first, and the path
	// comparison is stated on that one.
	uses, err := ModelUses()
	if err != nil {
		return nil, err
	}
	markUses(rows, uses)
	return rows, comparePaths(rows)
}

// uncaptionedGeneration is why an image or video model that takes no words has no held-out
// quality a vision-language model's answers could measure: the held-out evaluations ask about what
// a caption said to draw or an instruction said to change, and a class index over a few pixels, or
// no condition at all, says nothing to ask about.
const uncaptionedGeneration = "this model takes no caption or instruction (its input is a class, or nothing), so the held-out evaluations of the generators and editors, a vision-language model's answers about what a caption said to draw or an instruction said to change, cannot be asked of it"

// editorProposal is the held-out evaluation proposed for a model that changes a video it is
// given: a caption alone does not drive it, so the generators' evaluation is not its own.
const editorProposal = "no held-out evaluation for this editor yet; proposed: held-out VidGen clips (on disk) changed by instruction, a local vision-language model asked whether each edited clip shows what it showed with the change made"

// unmeasuredGeneration is what an image or video recipe's held-out quality reads while no
// evaluation has measured it. A generator, which draws from a caption alone, keeps the proposal
// of its task. An editor, which changes a video by an instruction, is proposed the edit
// evaluation. A model conditioned on a class, on an image or on nothing has no words to be asked
// about: its quality is not applicable, and no row is filed for it.
func unmeasuredGeneration(proposed Cell, definition recipe.Definition) Cell {
	switch {
	case acceptsCaption(definition):
		return proposed
	case editsVideo(definition):
		proposal := inapplicable(editorProposal)
		proposal.Proposed = true
		return proposal
	default:
		return inapplicable(uncaptionedGeneration)
	}
}

// takes reports whether a definition has an input of the given kind.
func takes(definition recipe.Definition, kind recipe.DataKind) bool {
	return slices.ContainsFunc(definition.Inputs, func(input recipe.Input) bool { return input.Data == kind })
}

// acceptsCaption reports whether a definition has an input of prompt conditioning and no video to
// edit: a generator draws from a caption alone.
func acceptsCaption(definition recipe.Definition) bool {
	return takes(definition, recipe.DataPromptConditioning) && !takes(definition, recipe.DataVideo)
}

// editsVideo reports whether a definition takes both words and a video: an editor changes the
// video it is given as the words say.
func editsVideo(definition recipe.Definition) bool {
	return takes(definition, recipe.DataPromptConditioning) && takes(definition, recipe.DataVideo)
}

// projectionInRequest is why a projection cell no attempt measured is not applicable.
const projectionInRequest = "a projector runs inside the model's multimodal request, whose timing, memory and failures are measured on the model's inference and vqa rows"

// vqaStreamsWeights is why a vqa row's load time, when no attempt measured one, is not applicable.
const vqaStreamsWeights = "vqa reads each layer's weights during its first pass, so its load is part of its first output and no separate load phase exists"

// stateNotApplicable states, on a row, each cell no attempt measured that is not a measurement
// missing: a projector runs inside the model's multimodal request, whose first output, throughput,
// load, memory and failures the model's own rows measure, and the vqa pipeline reads its weights
// layer by layer during its first pass, so it has no load phase of its own. A cell an attempt
// measured stays.
func stateNotApplicable(row *Row) {
	reasons := map[recipe.Task]map[metric.Name]string{
		recipe.TaskProjection: {
			metric.FirstOutput: projectionInRequest, metric.Throughput: projectionInRequest, metric.Failures: projectionInRequest,
			metric.Load: projectionInRequest, metric.PeakMemory: projectionInRequest,
		},
		recipe.TaskVQA: {metric.Load: vqaStreamsWeights},
	}
	for column, reason := range reasons[row.Task] {
		if cell := row.Cells[column]; !cell.Measured() && !cell.NotApplicable {
			row.Cells[column] = inapplicable(reason)
		}
	}
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

// servingSummary reads the serving attempts recorded under one recipe as a
// user meets the recipe now. Its attempts, failures and cancellations count
// back to the recipe's newest success, which succeeded says it has. Its
// memory is that of one attempt, measured: the newest success that timed
// its model's load, which loaded says it found, else the newest success.
// deviceSampled says that attempt sampled its device memory at its finish,
// so a peak of zero is what it measured. Its timings are read from window.
type servingSummary struct {
	attempts, failures, cancelled    int
	succeeded, loaded, deviceSampled bool
	peakDeviceBytes                  uint64
	latest, measured, recipe         artifact.ID
	// environment is where the measured attempt ran, or the latest while none has succeeded: a serving value
	// is read under it, and a baseline measured under another, or naming none, is Unknown, not a change.
	environment artifact.ID
	// window is the measured attempt and, after it, the next older ones that timed their load under its
	// environment, recentAttempts at most: an attempt under another environment is no repeat of it.
	window []servingSample
	// lengths reads, by each of evaluation.PromptLengths, the successes whose prompt was of about that
	// length: the newest, under whose environment its window is, and the next older ones under it.
	lengths map[int]*servingSummary
}

// lengthSpan is the share of a prompt length, one over it, that a prompt may be shorter or longer
// and still read at that length: a quarter.
const lengthSpan = 4

// longPrompt reports whether a chat model's attempt read a longer prompt than its verified request
// does, and the length it read at, if any: a prompt within a quarter of a length either side is that
// length's. A prompt more than a quarter short of the shortest length is the request a row's own
// cells read; a longer one between two lengths is timed by neither.
func longPrompt(task recipe.Task, tokens uint64) (length int, long bool) {
	if task != recipe.TaskInference {
		return 0, false
	}
	for _, length := range evaluation.PromptLengths {
		if quarter := uint64(length / lengthSpan); tokens+quarter >= uint64(length) && tokens <= uint64(length)+quarter {
			return length, true
		}
	}
	return 0, tokens+uint64(evaluation.PromptLengths[0]/lengthSpan) >= uint64(evaluation.PromptLengths[0])
}

// recentAttempts is how many of a recipe's newest measured attempts a serving timing is read from. One
// attempt of a short operation varies by a factor of two or more with scheduling and cache state, and
// more beside other work, so a cell reads the median of a few and carries how far they stand from it.
const recentAttempts = 3

// The timings a serving attempt is read for, and how many they are.
const (
	timingFirstOutput = iota
	timingWholeOutput
	timingLoad
	timingDecodeRate
	timingCount
)

// servingSample is one measured attempt and what each of its timings took, the times in milliseconds.
type servingSample struct {
	attempt artifact.ID
	took    [timingCount]float64
}

// half is the quantile that is a median, and the share of a normal reading's two tails on one side.
const half = 0.5

// median sorts values and returns the middle one, or the point midway between the middle two.
func median(values []float64) float64 {
	slices.Sort(values)
	return tensorstats.Quantile(values, half)
}

// timed is the cell of one timing, read from the summary's window: the median of what its attempts
// took, and as its spread their median distance from it over it, so one attempt far from the others,
// taken beside other work, neither stands as the cell nor widens what a later reading is allowed. It
// reports false where the measured attempt did not take the timing.
func (summary *servingSummary) timed(timing int, unit, metric, label string) (Cell, bool) {
	// An older attempt's value never stands in for one the measured attempt did not take.
	if summary == nil || len(summary.window) == 0 || summary.window[0].took[timing] <= 0 {
		return Cell{}, false
	}
	var values []float64
	for _, sample := range summary.window {
		if taken := sample.took[timing]; taken > 0 {
			values = append(values, taken)
		}
	}
	middle := median(values)
	// The cell names the attempt whose value the median is. The median of two lies midway and is
	// neither's: the cell then names the newer, the measured attempt, which the memory cell reads.
	named := max(slices.IndexFunc(summary.window, func(sample servingSample) bool { return sample.took[timing] == middle }), 0)
	cell := summary.cell(summary.window[named].attempt, middle, unit, metric, label)
	for index, taken := range values {
		values[index] = math.Abs(taken - middle)
	}
	cell.Spread = median(values) / middle
	return cell, true
}

// readLength adds a success at a longer prompt to its length's window: the newest opens it under its
// environment, and the next older ones under that environment join it, recentAttempts at most. A length
// is read from its prompt alone, so a success that timed no prefill is no first output and is passed.
func (summary *servingSummary) readLength(length int, observation runrecord.ServingObservation) {
	at := slices.IndexFunc(observation.Hardware, func(sample runrecord.ServingHardwareSample) bool {
		return sample.Stage == runrecord.ServingHardwarePrefill && sample.ElapsedNS != 0
	})
	if at < 0 {
		return
	}
	if summary.lengths == nil {
		summary.lengths = map[int]*servingSummary{}
	}
	read := summary.lengths[length]
	if read == nil {
		read = &servingSummary{measured: observation.ID, recipe: observation.Recipe, environment: observation.Environment}
		summary.lengths[length] = read
	}
	if len(read.window) >= recentAttempts || observation.Environment != read.environment {
		return
	}
	sample := servingSample{attempt: observation.ID}
	sample.took[timingFirstOutput] = milliseconds(observation.Hardware[at].ElapsedNS)
	read.window = append(read.window, sample)
}

// lengthCells are a row's first outputs at the longer prompts its attempts read, shortest first.
func lengthCells(task recipe.Task, observed *servingSummary) []Cell {
	if task != recipe.TaskInference || observed == nil {
		return nil
	}
	var cells []Cell
	for _, length := range evaluation.PromptLengths {
		at := fmt.Sprintf("a fresh prompt of %d tokens", length)
		if cell, measured := observed.lengths[length].timed(timingFirstOutput, "ms", "time to prefill, "+at, "serving time to prefill "+at+", the median of the newest attempts"); measured {
			cell.Prompt = length
			cells = append(cells, cell)
		}
	}
	return cells
}

// cell is a value read from source, an attempt of the summary's window or
// its latest, as label says.
func (summary *servingSummary) cell(source artifact.ID, value float64, unit, metric, label string) Cell {
	return Cell{Value: value, Unit: unit, Label: label, Source: source, Environment: summary.environment, Contract: Contract{Metric: metric, Origin: originServing, Scope: summary.recipe}}
}

// servingByRecipe reads every serving observation, newest first, into one
// summary per recipe, so a row counts its own activation's attempts and a
// fix shipped as a new recipe starts clean. A failure counts until a later
// attempt of its recipe succeeds: the failure rate is the failed share of
// the attempts back to the newest success, so a fix shipped in code closes
// it with the first attempt that succeeds, and a recipe failing now reads
// as failing whatever it did before. The memory cell reads one attempt, the
// newest success that timed its load: the cycle measures each recipe that
// way, from a fresh load on the request its verification ran, and a request
// served from a resident session does not stand in for it. A timing cell
// reads the median of that attempt and the next older ones that timed their
// load under its environment, recentAttempts in all, so one attempt taken
// beside other work is not the cell; a step in a timing then shows once
// most of them are past it, one measurement run late. A cancelled attempt
// counts as neither success nor failure and contributes no samples: the
// server records both a user's cancel and its own deadline that way. An
// observation a gate result depends on is a validation run's measurement,
// deliberate interrupts included, not an attempt a user made, and is left
// out: E4B's media resource proof published 324 deliberate failures under
// its projection recipe. An attempt that was measured beside other work,
// which the record of a contended reading depends on, is left out the same
// way, so its recipe's cells keep what the earlier attempts gave them.
func servingByRecipe(ctx context.Context, store *overgodb.Store) (map[artifact.ID]*servingSummary, error) {
	summaries := map[artifact.ID]*servingSummary{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ServingObservationMediaType, Schema: runrecord.ServingObservationSchema,
		}},
		Order: overgodb.DocumentNewestFirst,
	}, runrecord.ParseServingObservation, func(view overgodb.DocumentView, observation runrecord.ServingObservation) error {
		if aside, err := setAside(ctx, store, view.Content.Descriptor.ID); err != nil || aside {
			return err
		}
		summary := summaries[observation.Recipe]
		if summary == nil {
			summary = &servingSummary{latest: observation.ID, recipe: observation.Recipe, environment: observation.Environment}
			summaries[observation.Recipe] = summary
		}
		counted := !summary.succeeded
		switch observation.Outcome {
		case runrecord.OutcomeSucceeded:
			summary.succeeded = true
		case runrecord.OutcomeFailed:
			if counted {
				summary.failures++
			}
		default:
			if counted {
				summary.cancelled++
			}
			return nil
		}
		if counted {
			summary.attempts++
		}
		// An attempt that read a longer prompt counts as an attempt, and is timed as its length's alone: the
		// tokens it read, not those a prompt cache answered, set its length.
		if length, long := longPrompt(observation.Task, observation.Usage.InputTokens-min(observation.Usage.CachedTokens, observation.Usage.InputTokens)); long {
			if length != 0 && observation.Outcome == runrecord.OutcomeSucceeded {
				summary.readLength(length, observation)
			}
			return nil
		}
		// Past the attempt that timed its load, an older success is read only for the timings the
		// medians are of, and for nothing once the window is full.
		if observation.Outcome != runrecord.OutcomeSucceeded || summary.loaded && len(summary.window) >= recentAttempts {
			return nil
		}
		measured := servingSummary{measured: observation.ID, environment: observation.Environment, lengths: summary.lengths}
		timings := servingSample{attempt: observation.ID}
		for _, sample := range observation.Hardware {
			if sample.Stage == runrecord.ServingHardwarePrefill {
				timings.took[timingFirstOutput] = milliseconds(sample.ElapsedNS)
			}
			measured.peakDeviceBytes = max(measured.peakDeviceBytes, sample.DevicePeakBytes)
			measured.deviceSampled = measured.deviceSampled || sample.Stage == runrecord.ServingHardwareFinish
		}
		// A capability that returns its output whole, an image or a
		// clip, first outputs when it finishes.
		if timings.took[timingFirstOutput] == 0 {
			timings.took[timingWholeOutput] = milliseconds(observation.MeasuredNS)
		}
		measured.peakDeviceBytes = max(measured.peakDeviceBytes, observation.Resources.PeakDeviceBytes)
		for _, phase := range observation.Phases {
			switch {
			case phase.Phase == runrecord.PhaseLoad && phase.DurationNS != 0:
				timings.took[timingLoad], measured.loaded = milliseconds(phase.DurationNS), true
			case phase.Phase == runrecord.PhaseDecode && phase.DurationNS != 0:
				// The decode phase alone, as the benchmark's decode rate is.
				timings.took[timingDecodeRate] = float64(observation.Usage.OutputTokens) / time.Duration(phase.DurationNS).Seconds()
			}
		}
		// The newest success stands until an older one that timed its load is found: that one is the
		// measurement and opens the window, and the older ones past it only join it.
		if summary.loaded || summary.measured.Valid() && !measured.loaded {
			if summary.loaded && measured.loaded && observation.Environment == summary.environment {
				summary.window = append(summary.window, timings)
			}
			return nil
		}
		measured.attempts, measured.failures, measured.cancelled = summary.attempts, summary.failures, summary.cancelled
		measured.succeeded, measured.latest, measured.recipe = true, summary.latest, summary.recipe
		measured.window = []servingSample{timings}
		*summary = measured
		return nil
	})
	return summaries, err
}

// setAside reports whether a record that takes a serving attempt out of the
// scorecard's reading depends on the artifact. A gate result does, since its
// run measured the attempt, and so does the record of a contended reading.
func setAside(ctx context.Context, store *overgodb.Store, id artifact.ID) (bool, error) {
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
		if found && (child.Schema == runrecord.GateSchema || child.Schema == contendedSchema) {
			return true, nil
		}
	}
	return false, nil
}

func gap(reason string) Cell { return Cell{Reason: reason} }

// benchmarkCell is a value the benchmark record measured.
func benchmarkCell(value float64, unit, metric string, summary evaluation.BenchmarkSummary) Cell {
	return Cell{Value: value, Unit: unit, Label: originBenchmark + " " + metric, Source: summary.Record,
		CodeCommit: summary.CodeCommit, Environment: summary.Environment, Contract: Contract{Metric: metric, Origin: originBenchmark, Comparison: summary.Comparison}}
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
	if cell, measured := observed.timed(timingFirstOutput, "ms", "time to prefill", "serving time to prefill, the median of the newest measured attempts"); measured {
		return cell
	}
	if benchmarked && benchmark.FirstTokenMS != 0 {
		return benchmarkCell(benchmark.FirstTokenMS, "ms", "time to first token, median", benchmark)
	}
	if cell, measured := observed.timed(timingWholeOutput, "ms", "time to the whole output", "serving time to the whole output, the median of the newest measured attempts"); measured {
		return cell
	}
	return gap("no benchmark time to first token and no serving prefill sample")
}

func throughputCell(task recipe.Task, benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if task == recipe.TaskTraining {
		// A training session learns; it decodes no output tokens, so a decode rate is not its rate.
		return inapplicable("a training session does not decode output tokens")
	}
	if cell, measured := observed.timed(timingDecodeRate, "tok/s", "output tokens over decode time", "serving output tokens over decode time, the median of the newest measured attempts"); measured {
		return cell
	}
	if benchmarked && benchmark.DecodeTokensPerSecond != 0 {
		return benchmarkCell(benchmark.DecodeTokensPerSecond, "tok/s", "decode rate, median", benchmark)
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
// has no such evidence. It adds to descriptors, under the set's identity, the
// name of the measure the set's plan was run for and the cases it holds, where
// the store holds a case profile that gives them: a set's identity is that of
// its cases, so only that name tells a measure built again from another
// measure. Such a descriptor has a kind only where the set's cases count one
// unit, as a speech round trip's texts do, so that the set is ranked by them.
func nameEvaluatedSets(ctx context.Context, store *overgodb.Store, suites map[artifact.ID]string, descriptors map[string]evaluation.SuiteDescriptor) error {
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: evaluation.EvaluationEvidenceMediaType, Schema: evaluation.EvaluationEvidenceSchema,
		}},
		// Naming is order-independent; any one order reads them all.
		Order: overgodb.DocumentOldestFirst,
	}, evaluation.ParseEvaluationEvidence, func(_ overgodb.DocumentView, evidence evaluation.EvaluationEvidence) error {
		if _, named := suites[evidence.Dataset]; named {
			return nil
		}
		name := evidence.Dataset.String()
		suites[evidence.Dataset] = name
		if set, ok := evaluation.EvaluatedMeasure(ctx, store, evidence.Plan); ok && descriptors != nil {
			descriptors[name] = evaluatedDescriptor(set)
		}
		return nil
	})
	return err
}

// evaluatedDescriptor is the descriptor an evaluated set is read by.
func evaluatedDescriptor(set evaluation.EvaluatedSet) evaluation.SuiteDescriptor {
	return evaluation.SuiteDescriptor{Kind: set.Kind, Source: set.Measure, Cases: set.Cases, Chance: set.Chance}
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
	// unreadable says why the newest verdict or transcription report of the
	// recipe could not be reconstructed: its quality is then a gap with that
	// reason, not an older record's value in its place.
	unreadable string
}

// cell is the record's held-out quality, beside its base model's value.
func (record heldoutRecord) cell(task recipe.Task) Cell {
	if record.unreadable != "" {
		return gap(record.unreadable)
	}
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
	// A report that cannot be reconstructed leaves its recipe's quality a gap that says why, and the
	// other recipes' measured.
	unreadable := map[artifact.ID]string{}
	reports, err := evaluation.TranscriptionQualities(ctx, store, func(recipeID artifact.ID, reason error) {
		unreadable[recipeID] = reason.Error()
	})
	if err != nil {
		return nil, err
	}
	records := make(map[artifact.ID]heldoutRecord, len(reports)+len(unreadable))
	for recipeID, report := range reports {
		records[recipeID] = heldoutRecord{summary: report}
	}
	for recipeID, reason := range unreadable {
		records[recipeID] = heldoutRecord{unreadable: "its transcription report cannot be reconstructed: " + reason}
	}
	served := map[string][]artifact.ID{}
	for _, entry := range entries {
		for _, capability := range entry.Capabilities {
			if !slices.Contains(verdictTasks, capability.Task) {
				continue
			}
			// Current training binds the complete model manifest. Legacy
			// sessions bound one raw weights file, matched below by its bytes.
			served[entry.Model.DigestHex()] = append(served[entry.Model.DigestHex()], capability.Recipe)
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
	// A verdict whose observation cannot be read names no recipe, so it claims none; and every verdict
	// older than it, read after it, may then not be its recipe's newest, so those recipes read a gap that
	// names it and not the older verdict's value in its place.
	var unattributed string
	read := func(recipeID artifact.ID, record heldoutRecord) {
		if _, found := records[recipeID]; found {
			return
		}
		if unattributed != "" {
			record = heldoutRecord{unreadable: unattributed}
		}
		records[recipeID] = record
	}
	err = evaluation.VisitHeldoutVerdicts(ctx, store, func(observation artifact.ID, base, trained evaluation.EvalSummary) error {
		session, err := runrecord.RequireServingObservation(ctx, store, observation)
		if err != nil {
			return err
		}
		_, value, _ := evaluation.HeadlineMetric(base.Metrics)
		read(session.Recipe, heldoutRecord{summary: trained, base: &value})
		for _, recipeID := range served[session.Model.DigestHex()] {
			read(recipeID, heldoutRecord{summary: base})
		}
		return nil
	}, func(verdict, observation artifact.ID, reason error) {
		// The verdict claims its recipe like a readable one, newest first, and its gap names the verdict.
		session, err := runrecord.RequireServingObservation(ctx, store, observation)
		if err != nil {
			if unattributed == "" {
				unattributed = fmt.Sprintf("a newer held-out verdict %s cannot be attributed to a recipe (it cannot be reconstructed: %v; its observation %s cannot be read: %v), so an older verdict is not known to be this recipe's newest", verdict, reason, observation, err)
			}
			return
		}
		read(session.Recipe, heldoutRecord{unreadable: "its held-out verdict " + verdict.String() + " cannot be reconstructed: " + reason.Error()})
	})
	return records, err
}

// trainingProvesRoute is why a training recipe no held-out verdict covers has no held-out quality:
// only a train -objective session over a registered objective holds data out and publishes a
// verdict, and the other routes' sessions prove the route, a bounded run whose evidence is that
// its loss descends. The owner decided (2026-10-06, training-proof-route-quality) that such a
// recipe reads not applicable with this reason, and no row is filed for it.
const trainingProvesRoute = "a training recipe is scored by a held-out verdict, which only a train -objective session over a registered objective publishes; this recipe's session proves its route, a bounded run whose evidence is that its loss descends, and holds out no data"

// heldoutProposals names, for each task no held-out evaluation measures yet,
// the held-out set and metric proposed to measure it. Such a row's quality
// is not applicable rather than missing: it files no decline, and the
// loop proposes defining the evaluation. A task's entry goes when its
// held-out evaluation lands; inference has none, since every inference
// recipe runs the derived suites.
var heldoutProposals = map[recipe.Task]string{
	recipe.TaskGeneration:     "WikiText-2 test (Salesforce/wikitext wikitext-2-raw-v1, 0.7 MB) perplexity, as a sequence-scoring suite",
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
	recipe.TaskReconstruction: "held-out VidGen-1M frames and clips (on disk) through the autoencoder and back: the mean squared error, in eight-bit levels, of what comes out against what went in",
}

// qualityCell is the headline metric of the recipe's newest evaluation of
// its largest held-out suite: the most cases say the most about held-out
// answers, and the index selects the newest run per bound contract. A set no
// descriptor counts holds no cases, so of two equal suites the one named
// first, then the protocol named first, is read, the same in every record. A
// measure built again, as an edit evaluation is each time its model draws the
// clips again, is another set its descriptor's source names as the same
// measure: it stands where its first build's set stood among the recipe's
// suites, and of its builds the one evaluated last is read, under whatever
// protocol it was, as a set counted again another way is. One set's
// execution alternatives keep the order of their contracts, whichever ran
// last. It also returns the recipe's other held-out qualities, one per suite
// and protocol, an earlier build's among them. A recipe no suite measures
// reads not applicable when its task has a held-out evaluation proposed, else
// a gap.
func qualityCell(task recipe.Task, summaries []evaluation.EvalSummary, descriptors map[string]evaluation.SuiteDescriptor) (Cell, []Cell) {
	type heldout struct {
		cases uint64
		// first is the set of the measure's first build, and built where the
		// set's newest evaluation arrived among the summaries.
		first string
		built int
		cell  Cell
	}
	// The summaries arrive newest first: the last set of a measure is its
	// first build's, and a set's first summary its newest evaluation.
	measure := func(set string) string { return cmp.Or(descriptors[set].Source, set) }
	first, built := map[string]string{}, map[string]int{}
	for arrived, summary := range summaries {
		first[measure(summary.Suite)] = summary.Suite
		if _, seen := built[summary.Suite]; !seen {
			built[summary.Suite] = arrived
		}
	}
	var measured []heldout
	for _, summary := range summaries {
		// LatestEvidence owns deduplication by the bound contract; display
		// labels cannot discard independently sufficient alternatives here.
		if name, value, ok := evaluation.HeadlineMetric(summary.Metrics); ok {
			label, commit := fmt.Sprintf("%s, %s (%s)", name, summary.Suite, summary.Prompting), summary.CodeCommit
			if summary.Carried.Valid() {
				// A carried result stands on the run that scored it, which
				// the carrying run only replayed a sample of.
				label, commit = label+", carried", summary.ScoredCommit
			}
			// A reading rests on the cases its record's reader counted, else on
			// its set's. Only a suite with a kind is ranked by them: one the
			// derived catalog compiled, or an evaluated set whose cases count
			// one unit, as a speech round trip's texts do. Another evaluated
			// set's cases are frames at one size or clips at another, and say
			// nothing of which measure says more.
			descriptor := descriptors[summary.Suite]
			cases, noise, ranked, constant := cmp.Or(summary.Cases, descriptor.Cases), summary.Noise, descriptor.Cases, summary.Constant
			if name == shareMetric {
				noise = shareError(value, cases)
				// A share of choices answered right has chance as its constant answer.
				if constant == nil && descriptor.Chance > 0 {
					constant = &descriptor.Chance
				}
			}
			if descriptor.Kind == "" {
				ranked = 0
			}
			measured = append(measured, heldout{cases: ranked, first: first[measure(summary.Suite)], built: built[summary.Suite], cell: Cell{
				Value: value, Label: label, Source: summary.Record, Direction: summary.Directions[name],
				CodeCommit: commit, Environment: summary.Environment, Cases: cases, Noise: noise, Constant: constant, Measure: measure(summary.Suite), Unread: summary.Unread,
				Contract: Contract{Metric: name, Origin: originEvaluation, Suite: summary.Suite, Protocol: summary.Prompting, Comparison: summary.Comparison},
			}})
		}
	}
	switch {
	case len(measured) != 0:
	case len(summaries) != 0:
		return gap("the latest evaluation reports no metric"), nil
	case task == recipe.TaskTraining:
		return inapplicable(trainingProvesRoute), nil
	case heldoutProposals[task] != "":
		proposal := inapplicable(fmt.Sprintf("no held-out evaluation for %s yet; proposed: %s", task, heldoutProposals[task]))
		proposal.Proposed = true
		return proposal, nil
	default:
		return gap("no held-out evaluation of this recipe"), nil
	}
	slices.SortFunc(measured, func(left, right heldout) int {
		// A measure's builds are ordered before a build's protocols: a build
		// evaluated later is read first whatever its protocol is called, and
		// only one set's own alternatives fall to the order of their names.
		return cmp.Or(cmp.Compare(right.cases, left.cases), strings.Compare(left.first, right.first),
			cmp.Compare(left.built, right.built), strings.Compare(left.cell.Contract.Protocol, right.cell.Contract.Protocol),
			artifact.CompareID(left.cell.Contract.Comparison, right.cell.Contract.Comparison), artifact.CompareID(left.cell.Source, right.cell.Source))
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
	since := "since the last success"
	if !observed.succeeded {
		since = "and none succeeded"
	}
	cell := observed.cell(observed.latest, float64(observed.failures)/float64(observed.attempts), "", "failed share of attempts",
		fmt.Sprintf("%d failed and %d cancelled %s", observed.failures, observed.cancelled, since))
	cell.Attempts = observed.attempts
	return cell
}

func loadCell(benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if cell, measured := observed.timed(timingLoad, "ms", "load phase", "load phase, the median of the newest measured attempts"); measured {
		return cell
	}
	if benchmarked && benchmark.LoadMS != 0 {
		return benchmarkCell(benchmark.LoadMS, "ms", "load", benchmark)
	}
	return gap("no benchmark load time and no recorded load phase")
}

func peakMemoryCell(benchmark evaluation.BenchmarkSummary, benchmarked bool, observed *servingSummary) Cell {
	if observed != nil && observed.peakDeviceBytes != 0 {
		return observed.cell(observed.measured, float64(observed.peakDeviceBytes), "bytes", "peak device memory", "serving peak device memory, the newest measured attempt")
	}
	if observed != nil && observed.deviceSampled {
		// Sampled at an attempt's finish and never above zero: the model
		// holds no device memory, which is a measurement and no gap.
		return observed.cell(observed.measured, 0, "bytes", "peak device memory", "0 bytes: runs on the host")
	}
	if benchmarked && benchmark.PeakDeviceBytes != 0 {
		return benchmarkCell(float64(benchmark.PeakDeviceBytes), "bytes", "peak device memory", benchmark)
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
