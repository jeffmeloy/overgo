package main

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
)

// SurfaceID names one code surface whose change can invalidate a class of
// evidence. Every validation cell is pinned to exactly one surface; a commit
// that moves a cell's surface runs the cell whatever its evidence says.
type SurfaceID string

// ModelValidation is one (model, declared function, validation) obligation:
// the appropriate validation for one declared modality/function of one model,
// with the surface that pins its accepted evidence. Recipe is the
// activation's recipe: it finds the accepted record and is never itself
// evidence that the record holds.
type ModelValidation struct {
	Model          artifact.ID `json:"model"`
	ModelName      string      `json:"model_name,omitzero"`
	Location       string      `json:"location,omitzero"`
	Kind           recipe.Task `json:"kind,omitzero"` // empty: registered without activation
	Modality       string      `json:"modality,omitzero"`
	Validation     string      `json:"validation,omitzero"` // empty: no validation applicable to the kind
	Surface        SurfaceID   `json:"surface,omitzero"`
	Recipe         artifact.ID `json:"recipe,omitzero"`
	Protocol       string      `json:"protocol,omitzero"` // a benchmark's prompting protocol, which its re-acquisition repeats
	Cost           int64       `json:"cost_ns,omitzero"`  // measured wall or declared timeout, for smallest-first
	Stale          string      `json:"stale,omitzero"`    // activation defect, if any
	OperatorOnly   bool        `json:"operator_only,omitzero"`
	OperatorReason string      `json:"operator_reason,omitzero"`
	// Grounding is the committed data a training recipe's objective reads,
	// which names the route whose run verifies it again; Conditioning is the
	// video stack directory a raw DiT checkpoint trains against.
	Grounding    string `json:"grounding,omitzero"`
	Conditioning string `json:"conditioning,omitzero"`
}

// Selected pairs an obligation with the reason it landed in its plan bucket
// and, for a cell that runs, whether promotion waits on it and why.
type Selected struct {
	ModelValidation
	Reason    string `json:"reason"`
	Promotion string `json:"promotion,omitzero"`
}

// AffectRecord records whether a commit's changes moved one surface and why.
type AffectRecord struct {
	Affected bool   `json:"affected"`
	Reason   string `json:"reason"`
}

// Inputs are the resolved facts the selection reduces: the commit range, the
// per-surface affectedness derived from the range's changed paths, every
// registered model's validation obligations, the one current-evidence
// decision (acceptedRecords.current) the inventory takes too, and the
// baseline scorecard the cycle's next record is judged against, nil when
// none measured the promoted commit.
type Inputs struct {
	Baseline         string                      `json:"baseline"`
	Head             string                      `json:"head"`
	ChangedPaths     []string                    `json:"changed_paths"`
	AffectedSurfaces map[SurfaceID]AffectRecord  `json:"affected_surfaces"`
	Registered       []ModelValidation           `json:"-"`
	Current          func(ModelValidation) error `json:"-"`
	Scorecard        *scorecard.Record           `json:"-"`
}

// Plan is the validation decision for one commit: run what the commit's
// changes affect and what holds no current evidence, promotion waiting on
// what it compares while the rest runs in the background, reuse only what
// is current, name the cells no owner judges, and account for every
// registered model and every lane of the improvement cycle. Scorecard names
// the baseline scorecard that scoped the run.
type Plan struct {
	Baseline           string      `json:"baseline"`
	Head               string      `json:"head"`
	Scorecard          string      `json:"scorecard"`
	ChangedPaths       []string    `json:"changed_paths"`
	AffectedSurfaces   []SurfaceID `json:"affected_surfaces"`
	Run                []Selected  `json:"run"`
	Background         []Selected  `json:"background"`
	Reuse              []Selected  `json:"reuse"`
	Unknown            []Selected  `json:"unknown"`
	OperatorOwned      []Selected  `json:"operator_owned"`
	RegisteredInactive []Selected  `json:"registered_inactive"`
	SkipInappropriate  []Selected  `json:"skip_inappropriate"`

	Suites []SelectedSuite `json:"suites"`
}

// SelectedSuite is one lane of the improvement cycle, the real-model suites
// no landing runs, in the plan: whether it runs, why, and the go command's
// arguments and added environment that run it.
type SelectedSuite struct {
	Lane        string   `json:"lane"`
	Run         bool     `json:"run"`
	Reason      string   `json:"reason"`
	Arguments   []string `json:"arguments"`
	Environment []string `json:"environment,omitzero"`
}

// SelectValidation reduces the resolved inputs to a validation plan. It is
// pure: no store, git, filesystem or device access, so the whole decision is
// unit-tested. A cell is inapplicable when its model's own declaration
// excludes it, whatever moved. Otherwise it runs when this commit's changes
// move its surface, its activation is stale, or the current-evidence decision
// refuses its accepted record; it is reused only when that decision finds the
// record current, and it is unknown when no owner judges it. A short commit
// range is a reason to run, never proof that older evidence holds. Everything
// else is operator-owned, registered-inactive, or inapplicable. A cell that
// runs is scheduled by what promotion compares. Every bucket is ordered
// cheapest first so a broken run surfaces on a small model in seconds.
func SelectValidation(in Inputs) Plan {
	plan := Plan{
		Baseline: in.Baseline, Head: in.Head, ChangedPaths: in.ChangedPaths,
		Run: []Selected{}, Background: []Selected{}, Reuse: []Selected{}, Unknown: []Selected{}, OperatorOwned: []Selected{},
		RegisteredInactive: []Selected{}, SkipInappropriate: []Selected{},
	}
	if in.Scorecard != nil {
		plan.Scorecard = in.Scorecard.Name
	}
	for id, rec := range in.AffectedSurfaces {
		if rec.Affected {
			plan.AffectedSurfaces = append(plan.AffectedSurfaces, id)
		}
	}
	slices.Sort(plan.AffectedSurfaces)
	for _, mv := range in.Registered {
		sel := Selected{ModelValidation: mv}
		switch {
		case mv.OperatorOnly:
			sel.Reason = "operator-owned: " + mv.OperatorReason
			plan.OperatorOwned = append(plan.OperatorOwned, sel)
		case mv.Kind == "":
			sel.Reason = "registered without task activation"
			plan.RegisteredInactive = append(plan.RegisteredInactive, sel)
		case mv.Validation == "":
			sel.Reason = "no validation applicable to " + string(mv.Kind)
			plan.SkipInappropriate = append(plan.SkipInappropriate, sel)
		default:
			switch refusal, rec := in.Current(mv), in.AffectedSurfaces[mv.Surface]; {
			case errors.Is(refusal, errCellInapplicable):
				sel.Reason = "skip: " + refusal.Error()
				plan.SkipInappropriate = append(plan.SkipInappropriate, sel)
			case rec.Affected:
				sel.Reason = "run: " + string(mv.Surface) + " surface affected: " + rec.Reason
				plan.schedule(sel, in.Scorecard)
			case mv.Stale != "":
				sel.Reason = "run: the activation is stale: " + mv.Stale
				plan.schedule(sel, in.Scorecard)
			case refusal == nil:
				sel.Reason = "reuse: its accepted record is bound to the " + string(mv.Surface) + " identity the code is at now"
				plan.Reuse = append(plan.Reuse, sel)
			case errors.Is(refusal, errCellNotJudged):
				sel.Reason = "unknown: " + uncheckedEvidence
				plan.Unknown = append(plan.Unknown, sel)
			default:
				sel.Reason = "run: " + refusal.Error()
				plan.schedule(sel, in.Scorecard)
			}
		}
	}
	for _, bucket := range [][]Selected{plan.Run, plan.Background, plan.Reuse, plan.Unknown, plan.OperatorOwned, plan.RegisteredInactive, plan.SkipInappropriate} {
		slices.SortStableFunc(bucket, func(a, b Selected) int {
			return cmp.Or(cmp.Compare(a.Cost, b.Cost), cmp.Compare(a.Model.String(), b.Model.String()), cmp.Compare(a.Validation, b.Validation))
		})
	}
	return plan
}

// schedule files a cell that runs by what promotion compares, saying which
// and why: in Run when promotion waits on its pass, else in Background.
// Promotion waits on every guard, serving cell and proof, and on a benchmark
// that re-measures the held-out contract, suite and protocol, the baseline
// scorecard compares for its model and task. Any other benchmark, one never
// run for the model among them, runs in the background and never blocks
// promotion.
func (plan *Plan) schedule(sel Selected, baseline *scorecard.Record) {
	blocking, why := true, "promotion waits on every "+sel.Validation+" cell"
	if benchmark(sel.ModelValidation) {
		contract, none := comparedContract(baseline, sel.ModelValidation)
		why = fmt.Sprintf("the baseline scorecard compares %s on %s (%s)", contract.Metric, contract.Suite, contract.Protocol)
		switch blocking = none == "" && ofFamily(contract.Suite, sel.Validation) && contract.Protocol == sel.Protocol; {
		case none != "":
			why = none
		case !blocking:
			why += ", not " + sel.Validation + " (" + sel.Protocol + ")"
		}
	}
	if blocking {
		sel.Promotion = "blocking: " + why
		plan.Run = append(plan.Run, sel)
		return
	}
	sel.Promotion = "background: " + why
	plan.Background = append(plan.Background, sel)
}

// comparedContract is the held-out contract the baseline scorecard compares
// for a cell's model and task: the baseline row's headline quality, the one
// cell scorecard's Pair.Compare holds a record to. It says why there is none
// when no baseline measured one.
func comparedContract(baseline *scorecard.Record, cell ModelValidation) (scorecard.Contract, string) {
	if baseline == nil {
		return scorecard.Contract{}, "no baseline scorecard measured the commit " + scorecard.PromotedBranch + " holds, so promotion compares no held-out contract"
	}
	at := slices.IndexFunc(baseline.Rows, func(row scorecard.Row) bool { return row.Model == cell.Model && row.Task == cell.Kind })
	if at < 0 || !baseline.Rows[at].Cells[metric.Quality].Measured() {
		return scorecard.Contract{}, "the baseline scorecard measured no held-out quality for this model and task"
	}
	return baseline.Rows[at].Cells[metric.Quality].Contract, ""
}

// compareContracts gives each inference activation a benchmark cell on the
// held-out contract the baseline scorecard compares for it, under the
// protocol it compares, so the contract is judged and re-acquired as
// compared rather than under whichever protocol ran last. A compared family
// the activation has no cell of joins as one; a set outside the derived
// catalog, named by its dataset alone, is of no family.
func compareContracts(cells []ModelValidation, baseline *scorecard.Record) []ModelValidation {
	for _, cell := range cells {
		contract, _ := comparedContract(baseline, cell)
		if _, family, derived := strings.Cut(contract.Suite, "/"); derived && benchmark(cell) {
			at := slices.IndexFunc(cells, func(other ModelValidation) bool { return other.Recipe == cell.Recipe && other.Validation == family })
			if at < 0 {
				cell.Validation = family
				at, cells = len(cells), append(cells, cell)
			}
			cells[at].Protocol = contract.Protocol
		}
	}
	return cells
}
