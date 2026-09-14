package runrecord

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

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
	Reuse          SelectionReuse    `json:"reuse,omitzero"`
}

// SelectionReuse records the exact obligation and its receipt before execution.
// No receipt means no matching authority, not proof of a particular input change.
type SelectionReuse struct {
	Obligation artifact.ID `json:"obligation"`
	Receipt    artifact.ID `json:"receipt,omitzero"`
	Passed     bool        `json:"passed,omitzero"`
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

// SelectionCauseCodec owns selection-record construction, encoding and queries.
var SelectionCauseCodec = artifact.JSONDocumentCodec(
	"gate selection cause", artifact.KindEvidence, "application/vnd.overgo.gate-selection-cause+json", "overgo/gate-selection-cause/v1",
	canonicalizeSelectionCause, func(value SelectionCauseRecord) artifact.ID { return value.ID },
	func(value *SelectionCauseRecord, id artifact.ID) { value.ID = id },
	func(value SelectionCauseRecord) SelectionCauseRecord {
		value.Changed = slices.Clone(value.Changed)
		value.Packages = slices.Clone(value.Packages)
		for index := range value.Packages {
			entry := &value.Packages[index]
			entry.CompilerInputs = slices.Clone(entry.CompilerInputs)
			entry.RuntimeInputs = slices.Clone(entry.RuntimeInputs)
			entry.UnboundInputs = slices.Clone(entry.UnboundInputs)
			entry.RuntimeReaders = maps.Clone(entry.RuntimeReaders)
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
	slices.SortFunc(value.Packages, func(left, right SelectionPackage) int {
		return strings.Compare(left.Step+"\x00"+left.Package, right.Step+"\x00"+right.Package)
	})
	var previous string
	for index := range value.Packages {
		entry := &value.Packages[index]
		if strings.TrimSpace(entry.Package) == "" || strings.TrimSpace(entry.Step) == "" {
			return errors.New("run record: selection cause package requires its package and step")
		}
		key := entry.Step + "\x00" + entry.Package
		if reuse := entry.Reuse; reuse != (SelectionReuse{}) &&
			(reuse.Obligation.Kind() != artifact.KindEvidence ||
				reuse.Receipt != (artifact.ID{}) && reuse.Receipt.Kind() != artifact.KindEvidence ||
				reuse.Passed && !reuse.Receipt.Valid()) {
			return errors.New("run record: selection reuse requires an obligation and a receipt for a prior pass")
		}
		if previous == key {
			return fmt.Errorf("run record: selection cause repeats %s in %s", entry.Package, entry.Step)
		}
		previous = key
		for _, names := range []*[]string{&entry.CompilerInputs, &entry.RuntimeInputs, &entry.UnboundInputs} {
			slices.Sort(*names)
			*names = slices.Compact(*names)
		}
	}
	return nil
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
	Skipped        bool             `json:"skipped,omitzero"`
	ElapsedSeconds *float64         `json:"elapsed_seconds,omitempty"`
	Reuse          SelectionReuse   `json:"reuse,omitzero"`
}

// SelectionCauseCount is one histogram bar.
type SelectionCauseCount struct {
	Cause          string  `json:"cause"`
	Packages       int     `json:"packages"`
	Executed       int     `json:"executed"`
	Unstarted      int     `json:"unstarted"`
	Skipped        int     `json:"skipped"`
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
	Skipped    int         `json:"skipped"`
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
	for _, step := range result.Steps {
		if _, known := steps[step.Name]; known {
			return SelectionHistogram{}, fmt.Errorf("run record: gate result repeats step %s", step.Name)
		}
		steps[step.Name] = &SelectionStepSummary{Name: step.Name, Outcome: step.Outcome, DurationNS: step.DurationNS}
	}
	histogram := SelectionHistogram{
		Result: result.ID, Changed: slices.Clone(record.Changed), Packages: []SelectionPackageCauses{},
		Limitations: "Package elapsed values come from go test and overlap within a step; they are work, not wall. Primary labels partition packages for display only; every listed cause is independently sufficient.",
	}
	primary := map[string]*SelectionCauseCount{}
	sufficient := map[string]*SelectionCauseCount{}
	inputs := map[string]*SelectionCauseCount{}
	count := func(index map[string]*SelectionCauseCount, key string, executed, skipped bool, elapsed *float64) {
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
		} else if skipped {
			bar.Skipped++
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
		failed := entry.Action == "fail"
		executed := entry.Started && (entry.Action == "pass" || failed)
		skipped := entry.Action == "skip"
		step.Packages++
		if executed {
			step.Executed++
			if failed {
				step.Failed++
			}
		} else if skipped {
			step.Skipped++
		} else {
			step.Unstarted++
		}
		var elapsed *float64
		if executed {
			elapsed = entry.ElapsedSeconds
		}
		count(primary, causes[0].Kind, executed, skipped, elapsed)
		var previousKind string
		for _, cause := range causes {
			if cause.Kind != previousKind {
				previousKind = cause.Kind
				count(sufficient, cause.Kind, executed, skipped, elapsed)
			}
			if cause.Kind == SelectionCauseCompiler || cause.Kind == SelectionCauseRuntime {
				count(inputs, cause.Detail, executed, skipped, elapsed)
			}
		}
		histogram.Packages = append(histogram.Packages, SelectionPackageCauses{
			Package: entry.Package, Step: entry.Step, Primary: causes[0].Kind, Causes: causes,
			Executed: executed, Failed: failed, Skipped: skipped, ElapsedSeconds: elapsed, Reuse: entry.Reuse,
		})
	}
	for _, step := range result.Steps {
		if steps[step.Name].Packages != 0 {
			histogram.Steps = append(histogram.Steps, *steps[step.Name])
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
