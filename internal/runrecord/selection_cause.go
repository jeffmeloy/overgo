package runrecord

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// SelectionCauseMediaType names the retained selection-cause document.
	SelectionCauseMediaType = "application/vnd.overgo.gate-selection-cause+json"
	// SelectionCauseSchema versions the retained selection-cause document.
	SelectionCauseSchema = "overgo/gate-selection-cause/v1"
)

// SelectionCauseContract names the retained explanation of one gate's
// package selection: why each package was selected and what its execution cost.
var SelectionCauseContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: SelectionCauseMediaType, Schema: SelectionCauseSchema,
}

// SelectionPackage binds one package's selection causes to its observed
// execution in one gate step. Inputs are repository paths of the change.
type SelectionPackage struct {
	Package        string            `json:"package"`
	Step           string            `json:"step"`
	Input          artifact.ID       `json:"input,omitzero"`
	Action         string            `json:"action,omitzero"`
	Started        bool              `json:"started,omitzero"`
	ElapsedSeconds *float64          `json:"elapsed_seconds,omitempty"`
	CompilerInputs []string          `json:"compiler_inputs,omitempty"`
	RuntimeInputs  []string          `json:"runtime_inputs,omitempty"`
	UnboundInputs  []string          `json:"unbound_inputs,omitempty"`
	RuntimeReaders map[string]string `json:"runtime_readers,omitempty"`
}

// SelectionCauseRecord retains every package the gate's test steps requested
// with its attribution, so the histogram derives from retained records alone.
type SelectionCauseRecord struct {
	Version     uint16             `json:"version"`
	Result      artifact.ID        `json:"result"`
	Changed     []string           `json:"changed"`
	Packages    []SelectionPackage `json:"packages"`
	Limitations string             `json:"limitations"`
	ID          artifact.ID        `json:"-"`
}

var selectionCauseCodec = artifact.JSONDocumentCodec(
	"gate selection cause", SelectionCauseContract.Kind, SelectionCauseContract.MediaType, SelectionCauseContract.Schema,
	canonicalizeSelectionCause, func(value SelectionCauseRecord) artifact.ID { return value.ID },
	func(value *SelectionCauseRecord, id artifact.ID) { value.ID = id },
	func(value SelectionCauseRecord) SelectionCauseRecord {
		value.Changed = slices.Clone(value.Changed)
		value.Packages = slices.Clone(value.Packages)
		for index := range value.Packages {
			value.Packages[index].CompilerInputs = slices.Clone(value.Packages[index].CompilerInputs)
			value.Packages[index].RuntimeInputs = slices.Clone(value.Packages[index].RuntimeInputs)
			value.Packages[index].UnboundInputs = slices.Clone(value.Packages[index].UnboundInputs)
			value.Packages[index].RuntimeReaders = maps.Clone(value.Packages[index].RuntimeReaders)
		}
		return value
	},
)

func canonicalizeSelectionCause(value *SelectionCauseRecord) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion {
		return errors.New("run record: invalid selection cause version")
	}
	if value.Result.Kind() != artifact.KindEvidence {
		return errors.New("run record: selection cause requires the gate result it explains")
	}
	slices.Sort(value.Changed)
	value.Changed = slices.Compact(value.Changed)
	seen := map[string]bool{}
	for index := range value.Packages {
		entry := &value.Packages[index]
		if strings.TrimSpace(entry.Package) == "" || strings.TrimSpace(entry.Step) == "" {
			return errors.New("run record: selection cause package requires its package and step")
		}
		key := entry.Step + "\x00" + entry.Package
		if seen[key] {
			return fmt.Errorf("run record: selection cause repeats %s in %s", entry.Package, entry.Step)
		}
		seen[key] = true
		for _, names := range []*[]string{&entry.CompilerInputs, &entry.RuntimeInputs, &entry.UnboundInputs} {
			slices.Sort(*names)
			*names = slices.Compact(*names)
		}
	}
	slices.SortFunc(value.Packages, func(left, right SelectionPackage) int {
		return strings.Compare(left.Step+"\x00"+left.Package, right.Step+"\x00"+right.Package)
	})
	return nil
}

// NewSelectionCauseRecord identifies one retained explanation.
func NewSelectionCauseRecord(record SelectionCauseRecord) (SelectionCauseRecord, error) {
	record.Version = artifact.InitialDocumentVersion
	return selectionCauseCodec.New(record)
}

// Content encodes the record for the final gate batch.
func (record SelectionCauseRecord) Content() (artifact.Content, error) {
	return selectionCauseCodec.Content(record)
}

// ParseSelectionCauseRecord decodes one retained explanation.
func ParseSelectionCauseRecord(content []byte) (SelectionCauseRecord, error) {
	return selectionCauseCodec.Parse(content)
}

// Every cause listed on a package is independently sufficient for its
// selection; the kinds are listed in primary-label precedence.
const (
	// SelectionCauseCompiler compiles a changed file.
	SelectionCauseCompiler = "compiler"
	// SelectionCauseRuntime names a changed file at run time.
	SelectionCauseRuntime = "runtime"
	// SelectionCauseReader reaches an opaque reader bound to every root.
	SelectionCauseReader = "opaque-reader"
	// SelectionCauseDependent follows the import graph without a changed input of its own.
	SelectionCauseDependent = "dependent"
)

var selectionCausePrecedence = []string{SelectionCauseCompiler, SelectionCauseRuntime, SelectionCauseReader, SelectionCauseDependent}

// SelectionCause is one sufficient path from the change to a selected package.
type SelectionCause struct {
	Kind   string `json:"kind"`
	Detail string `json:"detail"`
}

// SelectionPackageCauses lists every sufficient cause of one package's
// selection with the primary label the histogram displays it under.
type SelectionPackageCauses struct {
	Package        string           `json:"package"`
	Step           string           `json:"step"`
	Primary        string           `json:"primary"`
	Causes         []SelectionCause `json:"causes"`
	Executed       bool             `json:"executed"`
	Failed         bool             `json:"failed,omitzero"`
	ElapsedSeconds *float64         `json:"elapsed_seconds,omitempty"`
}

// SelectionCauseCount is one histogram bar.
type SelectionCauseCount struct {
	Cause          string  `json:"cause"`
	Packages       int     `json:"packages"`
	Executed       int     `json:"executed"`
	Unstarted      int     `json:"unstarted"`
	ElapsedSeconds float64 `json:"elapsed_seconds"`
}

// SelectionStepSummary partitions one check's packages by observed execution.
type SelectionStepSummary struct {
	Name       string      `json:"name"`
	Outcome    StepOutcome `json:"outcome"`
	DurationNS uint64      `json:"duration_ns"`
	Packages   int         `json:"packages"`
	Executed   int         `json:"executed"`
	Failed     int         `json:"failed"`
	Unstarted  int         `json:"unstarted"`
}

// SelectionHistogram explains one gate's executed cost by selection cause.
// Primary counts partition the packages; sufficient counts overlap.
type SelectionHistogram struct {
	Result      artifact.ID              `json:"result"`
	Changed     []string                 `json:"changed"`
	Steps       []SelectionStepSummary   `json:"steps"`
	Primary     []SelectionCauseCount    `json:"primary"`
	Sufficient  []SelectionCauseCount    `json:"sufficient"`
	Inputs      []SelectionCauseCount    `json:"inputs"`
	Packages    []SelectionPackageCauses `json:"packages"`
	Limitations string                   `json:"limitations"`
}

// SelectionCauses derives every sufficient cause of one package's selection.
func SelectionCauses(entry SelectionPackage) []SelectionCause {
	var causes []SelectionCause
	for _, name := range entry.CompilerInputs {
		causes = append(causes, SelectionCause{Kind: SelectionCauseCompiler, Detail: name})
	}
	for _, name := range entry.RuntimeInputs {
		causes = append(causes, SelectionCause{Kind: SelectionCauseRuntime, Detail: name})
	}
	for _, reader := range slices.Sorted(maps.Keys(entry.RuntimeReaders)) {
		causes = append(causes, SelectionCause{Kind: SelectionCauseReader, Detail: reader + ": " + entry.RuntimeReaders[reader]})
	}
	if len(causes) == 0 {
		causes = append(causes, SelectionCause{Kind: SelectionCauseDependent, Detail: "selected through the import graph without a changed input of its own"})
	}
	return causes
}

// SelectionCauseHistogram reconstructs counts and causal paths from the
// retained gate result and its selection-cause record. Cost is attributed
// only to observed execution; an unobserved package is unstarted work.
func SelectionCauseHistogram(result GateResult, record SelectionCauseRecord) (SelectionHistogram, error) {
	if !result.ID.Valid() || record.Result != result.ID {
		return SelectionHistogram{}, errors.New("run record: selection cause record does not explain this gate result")
	}
	steps := map[string]*SelectionStepSummary{}
	var order []string
	for _, step := range result.Steps {
		if _, known := steps[step.Name]; known {
			return SelectionHistogram{}, fmt.Errorf("run record: gate result repeats step %s", step.Name)
		}
		steps[step.Name] = &SelectionStepSummary{Name: step.Name, Outcome: step.Outcome, DurationNS: step.DurationNS}
		order = append(order, step.Name)
	}
	histogram := SelectionHistogram{
		Result: result.ID, Changed: slices.Clone(record.Changed), Packages: []SelectionPackageCauses{},
		Limitations: "Package elapsed values come from go test and overlap within a step; they are work, not wall. Primary labels partition packages for display only; every listed cause is independently sufficient.",
	}
	primary := map[string]*SelectionCauseCount{}
	sufficient := map[string]*SelectionCauseCount{}
	inputs := map[string]*SelectionCauseCount{}
	count := func(index map[string]*SelectionCauseCount, key string, executed bool, elapsed *float64) {
		bar := index[key]
		if bar == nil {
			bar = &SelectionCauseCount{Cause: key}
			index[key] = bar
		}
		bar.Packages++
		if executed {
			bar.Executed++
			if elapsed != nil {
				bar.ElapsedSeconds += *elapsed
			}
		} else {
			bar.Unstarted++
		}
	}
	for _, entry := range record.Packages {
		step := steps[entry.Step]
		if step == nil {
			return SelectionHistogram{}, fmt.Errorf("run record: package %s names step %s absent from the gate result", entry.Package, entry.Step)
		}
		causes := SelectionCauses(entry)
		executed := entry.Started && (entry.Action == "pass" || entry.Action == "fail")
		failed := entry.Action == "fail"
		step.Packages++
		switch {
		case executed && failed:
			step.Executed++
			step.Failed++
		case executed:
			step.Executed++
		default:
			step.Unstarted++
		}
		var elapsed *float64
		if executed {
			elapsed = entry.ElapsedSeconds
		}
		count(primary, causes[0].Kind, executed, elapsed)
		kinds := map[string]bool{}
		for _, cause := range causes {
			if !kinds[cause.Kind] {
				kinds[cause.Kind] = true
				count(sufficient, cause.Kind, executed, elapsed)
			}
			if cause.Kind == SelectionCauseCompiler || cause.Kind == SelectionCauseRuntime {
				count(inputs, cause.Detail, executed, elapsed)
			}
		}
		histogram.Packages = append(histogram.Packages, SelectionPackageCauses{
			Package: entry.Package, Step: entry.Step, Primary: causes[0].Kind, Causes: causes,
			Executed: executed, Failed: failed, ElapsedSeconds: elapsed,
		})
	}
	for _, name := range order {
		if steps[name].Packages != 0 {
			histogram.Steps = append(histogram.Steps, *steps[name])
		}
	}
	for _, kind := range selectionCausePrecedence {
		if bar := primary[kind]; bar != nil {
			histogram.Primary = append(histogram.Primary, *bar)
		}
		if bar := sufficient[kind]; bar != nil {
			histogram.Sufficient = append(histogram.Sufficient, *bar)
		}
	}
	for _, name := range slices.Sorted(maps.Keys(inputs)) {
		histogram.Inputs = append(histogram.Inputs, *inputs[name])
	}
	slices.SortStableFunc(histogram.Inputs, func(left, right SelectionCauseCount) int { return right.Packages - left.Packages })
	return histogram, nil
}
