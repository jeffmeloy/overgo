package main

import (
	"cmp"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/landing"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
)

// ModelValidation is one (model, declared function, validation) obligation:
// the appropriate validation for one declared modality/function of one model.
// Recipe is the activation's recipe: it finds a benchmark's newest record.
type ModelValidation struct {
	Model          artifact.ID `json:"model"`
	ModelName      string      `json:"model_name,omitzero"`
	Location       string      `json:"location,omitzero"`  // the weights file the catalog finds the model by
	Directory      string      `json:"directory,omitzero"` // the root its inventory was registered from, which its capabilities resolve
	Kind           recipe.Task `json:"kind,omitzero"`      // empty: registered without activation
	Modality       string      `json:"modality,omitzero"`
	Validation     string      `json:"validation,omitzero"` // empty: no validation applicable to the kind
	Recipe         artifact.ID `json:"recipe,omitzero"`
	Protocol       string      `json:"protocol,omitzero"`     // a benchmark's prompting protocol, which its re-acquisition repeats
	Cost           int64       `json:"cost_ns,omitzero"`      // measured wall or declared timeout, for smallest-first
	Stale          string      `json:"stale,omitzero"`        // activation defect, if any
	Inapplicable   string      `json:"inapplicable,omitzero"` // why the model's declared evaluation domains exclude the cell
	OperatorOnly   bool        `json:"operator_only,omitzero"`
	OperatorReason string      `json:"operator_reason,omitzero"`
	// Grounding is the committed data a training recipe's objective reads,
	// which names the route whose run verifies it again, or the committed
	// exact suite or oracle a projection recipe's verification ran;
	// Objective is the alias that registers a training recipe's objective
	// no committed data grounds, which train -objective trains under, or,
	// for an objective that is not text, the route Route names;
	// Conditioning is the video stack directory a raw DiT checkpoint trains
	// against, and Projector the projector an exact projection suite runs on.
	Grounding    string `json:"grounding,omitzero"`
	Objective    string `json:"objective,omitzero"`
	Route        string `json:"route,omitzero"`
	Conditioning string `json:"conditioning,omitzero"`
	Projector    string `json:"projector,omitzero"`
	Consumed     string `json:"consumed,omitzero"` // what binding found the files its command is given to be (consumedFiles)
	// Trained is a model a training run produced, and Measured that its one measurement finished.
	Trained  bool `json:"trained,omitzero"`
	Measured bool `json:"measured,omitzero"`
}

// Selected pairs an obligation with the reason it landed in its plan bucket
// and, for a cell that runs, whether the scorecard waits on it and why.
type Selected struct {
	ModelValidation
	Reason    string `json:"reason"`
	Promotion string `json:"promotion,omitzero"`
	// Owed is how many cells a trained model's one measurement holds, whichever of them a narrower run was asked for.
	Owed int `json:"owed,omitzero"`
}

// Inputs are the resolved facts the selection reduces: every registered
// model's validation obligations, and the scorecard the cycle's next
// record is compared with, nil when none is published.
type Inputs struct {
	Registered []ModelValidation
	Scorecard  *scorecard.Record
}

// Plan is the validation decision for one tree: run every activated
// applicable cell, the scorecard waiting on what it compares while the rest runs
// in the background, and account for every registered model and every lane of
// the improvement cycle. Scorecard names the scorecard that scoped
// the run.
type Plan struct {
	Scorecard          string     `json:"scorecard"`
	Run                []Selected `json:"run"`
	Background         []Selected `json:"background"`
	OperatorOwned      []Selected `json:"operator_owned"`
	RegisteredInactive []Selected `json:"registered_inactive"`
	SkipInappropriate  []Selected `json:"skip_inappropriate"`
	// MeasuredOnce is the cells of trained models whose one measurement finished, which no run of every model repeats.
	MeasuredOnce []Selected `json:"measured_once"`
	// NotChosen is the cells of models a full run was not given (Chosen), whose last readings stand.
	NotChosen []Selected `json:"not_chosen"`

	Suites []SelectedSuite `json:"suites"`
}

// measuredModelsFile lists the models a full run measures again, by the names -model takes: one per runtime path,
// which the lead picks, and those the owner adds.
const measuredModelsFile = "docs/verification/measured_models.json"

// Chosen is the plan of a full run that measures the models it is given: those chosen, each trained model's one
// measurement and every model's first, of which the baseline holds no reading. Another model's cells are not chosen,
// and its last reading stands until a run is given it, by the list or by -model.
func (plan Plan) Chosen(chosen func(ModelValidation) bool, baseline *scorecard.Record) Plan {
	keep := func(cells []Selected) []Selected {
		return slices.DeleteFunc(slices.Clone(cells), func(sel Selected) bool {
			if chosen(sel.ModelValidation) || sel.Trained && !sel.Measured || !readBefore(baseline, sel.ModelValidation) {
				return false
			}
			sel.Reason, sel.Promotion = "not chosen: its last reading stands until "+measuredModelsFile+" or -model names it", ""
			plan.NotChosen = append(plan.NotChosen, sel)
			return true
		})
	}
	plan.Run, plan.Background = keep(plan.Run), keep(plan.Background)
	return plan
}

// readBefore says whether the baseline holds a reading of the cell's model on its task.
func readBefore(baseline *scorecard.Record, cell ModelValidation) bool {
	if baseline == nil {
		return false
	}
	at := slices.IndexFunc(baseline.Rows, func(row scorecard.Row) bool { return row.Model == cell.Model && row.Task == cell.Kind })
	return at >= 0 && slices.ContainsFunc(metric.Columns, func(column metric.Name) bool {
		return slices.ContainsFunc(baseline.Rows[at].Measured(column), func(read scorecard.Cell) bool { return read.Measured() && read.Source.Valid() })
	})
}

// OnlyModel is the plan of the models named alone: their cells, and no lane, so a decline's row is judged by its own
// model measured again (validate -run -model), never by a run of every model, and a trained model measured once before
// runs when it is named. It refuses a name no cell answers to.
func (plan Plan) OnlyModel(name string, named func(ModelValidation) bool) (Plan, error) {
	keep := func(cells []Selected) []Selected {
		return slices.DeleteFunc(slices.Clone(cells), func(sel Selected) bool { return !named(sel.ModelValidation) })
	}
	only := Plan{
		Scorecard: plan.Scorecard, Run: append(keep(plan.Run), keep(plan.MeasuredOnce)...), Background: keep(plan.Background), OperatorOwned: keep(plan.OperatorOwned),
		RegisteredInactive: []Selected{}, SkipInappropriate: []Selected{}, MeasuredOnce: []Selected{},
	}
	if len(only.Run)+len(only.Background)+len(only.OperatorOwned) == 0 {
		return Plan{}, fmt.Errorf("validate -model %s: no cell of the plan is that model's", name)
	}
	return only, nil
}

// OnlyTrained is the plan of the trained models still to be measured: their cells, which SelectValidation holds in Run
// all of them, and no lane.
func (plan Plan) OnlyTrained() Plan {
	return Plan{
		Scorecard: plan.Scorecard, Background: []Selected{}, OperatorOwned: []Selected{}, RegisteredInactive: []Selected{},
		SkipInappropriate: []Selected{}, MeasuredOnce: []Selected{},
		Run: slices.DeleteFunc(slices.Clone(plan.Run), func(sel Selected) bool { return !sel.Trained || sel.Measured }),
	}
}

// SelectedSuite is one lane of the improvement cycle, the real-model suites
// no landing runs, in the plan: the go command's arguments and added
// environment that run it.
type SelectedSuite struct {
	Lane        string   `json:"lane"`
	Arguments   []string `json:"arguments"`
	Environment []string `json:"environment,omitzero"`
}

// cycleSuites lists the improvement cycle's lanes, which run every cycle, each
// go test lane afresh, since its models lie outside the module's cache keys.
func cycleSuites() []SelectedSuite {
	suites := []SelectedSuite{}
	for _, lane := range landing.CycleLanes() {
		suites = append(suites, SelectedSuite{Lane: lane.Name, Arguments: lane.Arguments, Environment: lane.Environment})
	}
	return suites
}

// SelectValidation reduces the resolved inputs to a validation plan. It is
// pure: no store, git, filesystem or device access, so the whole decision is
// unit-tested. A cell is inapplicable when its model's own declaration
// excludes it. Every other activated cell runs, every cycle: nothing here
// asks whether its evidence still holds. A benchmark that runs puts to its
// model only the cases the store holds no answer to by the code that answers
// now, which evaluate decides. Everything else is operator-owned, registered-inactive, or
// inappropriate to its kind. A cell that runs is scheduled by what the scorecard
// compares. Every bucket is ordered cheapest first so a broken run surfaces
// on a small model in seconds.
func SelectValidation(in Inputs) Plan {
	plan := Plan{
		Run: []Selected{}, Background: []Selected{}, OperatorOwned: []Selected{},
		RegisteredInactive: []Selected{}, SkipInappropriate: []Selected{}, MeasuredOnce: []Selected{},
	}
	if in.Scorecard != nil {
		plan.Scorecard = in.Scorecard.Name
	}
	for _, mv := range in.Registered {
		sel := Selected{ModelValidation: mv}
		switch {
		case mv.Trained && mv.Measured:
			sel.Reason = "skip: a trained model is measured once, after its training"
			plan.MeasuredOnce = append(plan.MeasuredOnce, sel)
		case mv.OperatorOnly:
			sel.Reason = "operator-owned: " + mv.OperatorReason
			plan.OperatorOwned = append(plan.OperatorOwned, sel)
		case mv.Kind == "":
			sel.Reason = "registered without task activation"
			plan.RegisteredInactive = append(plan.RegisteredInactive, sel)
		case mv.Validation == "":
			sel.Reason = "no validation applicable to " + string(mv.Kind)
			plan.SkipInappropriate = append(plan.SkipInappropriate, sel)
		case mv.Inapplicable != "":
			sel.Reason = "skip: " + mv.Inapplicable
			plan.SkipInappropriate = append(plan.SkipInappropriate, sel)
		case mv.Trained:
			// The one measurement it has: the scorecard waits on every cell of it, the background ones too.
			sel.Reason, sel.Promotion = "run: a trained model's one measurement", "blocking: a trained model is measured once"
			plan.Run = append(plan.Run, sel)
		case mv.Stale != "":
			sel.Reason = "run: the activation is stale: " + mv.Stale
			plan.schedule(sel, in.Scorecard)
		default:
			sel.Reason = "run: re-acquired every cycle"
			plan.schedule(sel, in.Scorecard)
		}
	}
	owed := map[artifact.ID]int{}
	for _, sel := range plan.Run {
		if sel.Trained && !sel.Measured {
			owed[sel.Model]++
		}
	}
	for index := range plan.Run {
		plan.Run[index].Owed = owed[plan.Run[index].Model]
	}
	for _, bucket := range [][]Selected{plan.Run, plan.Background, plan.OperatorOwned, plan.RegisteredInactive, plan.SkipInappropriate, plan.MeasuredOnce} {
		slices.SortStableFunc(bucket, func(a, b Selected) int {
			return cmp.Or(cmp.Compare(a.Cost, b.Cost), cmp.Compare(a.Model.String(), b.Model.String()), cmp.Compare(a.Validation, b.Validation))
		})
	}
	return plan
}

// schedule files a cell that runs by what the scorecard compares, saying which
// and why: in Run when the scorecard waits on its pass, else in Background.
// The scorecard waits on every guard, serving cell and proof, and on a benchmark
// that re-measures a held-out contract, suite and protocol, the baseline
// scorecard compares for its model and task. Any other benchmark, one never
// run for the model among them, runs in the background and never blocks
// the scorecard.
func (plan *Plan) schedule(sel Selected, baseline *scorecard.Record) {
	blocking, why := true, "the scorecard waits on every "+sel.Validation+" cell"
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
			why = "the scorecard compares " + compared[at]
		default:
			why = "the scorecard compares " + strings.Join(compared, ", ") + ", not " + sel.Validation + " (" + sel.Protocol + ")"
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

// comparedContracts are the held-out contracts the scorecard
// compares for a cell's model and task: those of every held-out quality the
// baseline row measured, the headline and each other, the cells scorecard's
// Pair.Compare holds a record to. It says why there are none when no
// baseline measured one.
func comparedContracts(baseline *scorecard.Record, cell ModelValidation) ([]scorecard.Contract, string) {
	if baseline == nil {
		return nil, "no scorecard is published yet, so no held-out contract is compared"
	}
	var contracts []scorecard.Contract
	if at := slices.IndexFunc(baseline.Rows, func(row scorecard.Row) bool { return row.Model == cell.Model && row.Task == cell.Kind }); at >= 0 {
		for _, quality := range baseline.Rows[at].Measured(metric.Quality) {
			contracts = append(contracts, quality.Contract)
		}
	}
	if len(contracts) == 0 {
		return nil, "the scorecard measured no held-out quality for this model and task"
	}
	return contracts, ""
}

// compareContracts gives each inference activation a benchmark cell on each
// held-out contract the scorecard compares for it, under the
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
