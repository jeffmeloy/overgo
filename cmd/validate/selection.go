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
// evidence. Every validation cell is pinned to exactly one surface.
type SurfaceID string

// ModelValidation is one (model, declared function, validation) obligation:
// the appropriate validation for one declared modality/function of one model,
// with the surface that pins its accepted evidence. Recipe is the
// activation's recipe: it finds the accepted record and is never itself
// evidence that the record holds.
type ModelValidation struct {
	Model          artifact.ID `json:"model"`
	ModelName      string      `json:"model_name,omitzero"`
	Location       string      `json:"location,omitzero"`  // the weights file the catalog finds the model by
	Directory      string      `json:"directory,omitzero"` // the root its inventory was registered from, which its capabilities resolve
	Kind           recipe.Task `json:"kind,omitzero"`      // empty: registered without activation
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
	// which names the route whose run verifies it again, or the committed
	// exact suite or oracle a projection recipe's verification ran;
	// Objective is the alias that registers a training recipe's objective
	// no committed data grounds, which train -objective trains under;
	// Conditioning is the video stack directory a raw DiT checkpoint trains
	// against, and Projector the projector an exact projection suite runs on.
	Grounding    string `json:"grounding,omitzero"`
	Objective    string `json:"objective,omitzero"`
	Conditioning string `json:"conditioning,omitzero"`
	Projector    string `json:"projector,omitzero"`
	// Replay is the record a benchmark's run replays first: its newest
	// evaluation, which evaluate carries forward only when the replay
	// reproduces it.
	Replay artifact.ID `json:"replay,omitzero"`
}

// Selected pairs an obligation with the reason it landed in its plan bucket
// and, for a cell that runs, whether promotion waits on it and why.
type Selected struct {
	ModelValidation
	Reason    string `json:"reason"`
	Promotion string `json:"promotion,omitzero"`
}

// Inputs are the resolved facts the selection reduces: every registered
// model's validation obligations, the one current-evidence decision
// (acceptedRecords.current) the inventory takes too, and the baseline
// scorecard the cycle's next record is judged against, nil when none
// measured the promoted commit.
type Inputs struct {
	Registered []ModelValidation
	Current    func(ModelValidation) error
	Scorecard  *scorecard.Record
}

// Plan is the validation decision for one tree: run what holds no current
// evidence, promotion waiting on what it compares while the rest runs in the
// background, reuse only what is current, name the cells no owner judges,
// and account for every registered model and every lane of the improvement
// cycle. Scorecard names the baseline scorecard that scoped the run.
type Plan struct {
	Scorecard          string     `json:"scorecard"`
	Run                []Selected `json:"run"`
	Background         []Selected `json:"background"`
	Reuse              []Selected `json:"reuse"`
	Unknown            []Selected `json:"unknown"`
	OperatorOwned      []Selected `json:"operator_owned"`
	RegisteredInactive []Selected `json:"registered_inactive"`
	SkipInappropriate  []Selected `json:"skip_inappropriate"`

	Suites []SelectedSuite `json:"suites"`
}

// SelectedSuite is one lane of the improvement cycle, the real-model suites
// no landing runs, in the plan: the go command's arguments and added
// environment that run it.
type SelectedSuite struct {
	Lane        string   `json:"lane"`
	Arguments   []string `json:"arguments"`
	Environment []string `json:"environment,omitzero"`
}

// SelectValidation reduces the resolved inputs to a validation plan. It is
// pure: no store, git, filesystem or device access, so the whole decision is
// unit-tested. A cell is inapplicable when its model's own declaration
// excludes it. Otherwise it runs when its activation is stale or the
// current-evidence decision refuses its accepted record; it is reused only
// when that decision finds the record current, and it is unknown when no
// owner judges it. Everything else is operator-owned, registered-inactive,
// or inapplicable. A cell that runs is scheduled by what promotion compares.
// Every bucket is ordered cheapest first so a broken run surfaces on a small
// model in seconds.
func SelectValidation(in Inputs) Plan {
	plan := Plan{
		Run: []Selected{}, Background: []Selected{}, Reuse: []Selected{}, Unknown: []Selected{}, OperatorOwned: []Selected{},
		RegisteredInactive: []Selected{}, SkipInappropriate: []Selected{},
	}
	if in.Scorecard != nil {
		plan.Scorecard = in.Scorecard.Name
	}
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
			switch refusal := in.Current(mv); {
			case errors.Is(refusal, errCellInapplicable):
				sel.Reason = "skip: " + refusal.Error()
				plan.SkipInappropriate = append(plan.SkipInappropriate, sel)
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
// that re-measures a held-out contract, suite and protocol, the baseline
// scorecard compares for its model and task. Any other benchmark, one never
// run for the model among them, runs in the background and never blocks
// promotion.
func (plan *Plan) schedule(sel Selected, baseline *scorecard.Record) {
	blocking, why := true, "promotion waits on every "+sel.Validation+" cell"
	if benchmark(sel.ModelValidation) {
		contracts, none := comparedContracts(baseline, sel.ModelValidation)
		compared := make([]string, len(contracts))
		for index, contract := range contracts {
			compared[index] = fmt.Sprintf("%s on %s (%s)", contract.Metric, contract.Suite, contract.Protocol)
		}
		at := slices.IndexFunc(contracts, func(contract scorecard.Contract) bool {
			return ofFamily(contract.Suite, sel.Validation) && contract.Protocol == sel.Protocol
		})
		switch blocking = at >= 0; {
		case none != "":
			why = none
		case blocking:
			why = "the baseline scorecard compares " + compared[at]
		default:
			why = "the baseline scorecard compares " + strings.Join(compared, ", ") + ", not " + sel.Validation + " (" + sel.Protocol + ")"
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

// comparedContracts are the held-out contracts the baseline scorecard
// compares for a cell's model and task: those of every held-out quality the
// baseline row measured, the headline and each other, the cells scorecard's
// Pair.Compare holds a record to. It says why there are none when no
// baseline measured one.
func comparedContracts(baseline *scorecard.Record, cell ModelValidation) ([]scorecard.Contract, string) {
	if baseline == nil {
		return nil, "no baseline scorecard measured the commit " + scorecard.PromotedBranch + " holds, so promotion compares no held-out contract"
	}
	var contracts []scorecard.Contract
	if at := slices.IndexFunc(baseline.Rows, func(row scorecard.Row) bool { return row.Model == cell.Model && row.Task == cell.Kind }); at >= 0 {
		for _, quality := range baseline.Rows[at].Measured(metric.Quality) {
			contracts = append(contracts, quality.Contract)
		}
	}
	if len(contracts) == 0 {
		return nil, "the baseline scorecard measured no held-out quality for this model and task"
	}
	return contracts, ""
}

// compareContracts gives each inference activation a benchmark cell on each
// held-out contract the baseline scorecard compares for it, under the
// protocol it compares, so every compared contract is judged and re-acquired
// as compared rather than under whichever protocol ran last. A compared
// contract takes the activation's cell of its family that no other compared
// contract took, else joins as a cell of its own, so a family compared under
// two protocols is two cells; a set outside the derived catalog, named by its
// dataset alone, is of no family.
func compareContracts(cells []ModelValidation, baseline *scorecard.Record) []ModelValidation {
	taken := map[int]bool{}
	for _, cell := range cells {
		if !benchmark(cell) {
			continue
		}
		contracts, _ := comparedContracts(baseline, cell)
		for _, contract := range contracts {
			_, family, derived := strings.Cut(contract.Suite, "/")
			if !derived {
				continue
			}
			at := -1
			for index, other := range cells {
				if other.Recipe != cell.Recipe || other.Validation != family {
					continue
				}
				if taken[index] && other.Protocol == contract.Protocol {
					at = index
					break
				}
				if !taken[index] && at < 0 {
					at = index
				}
			}
			if at < 0 {
				at, cells = len(cells), append(cells, cell)
				cells[at].Validation = family
			}
			cells[at].Protocol, taken[at] = contract.Protocol, true
		}
	}
	return cells
}
