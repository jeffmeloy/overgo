package automationcheck

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// Coverage dimensions a model-regression guard must exercise. Weight type alone
// is insufficient, so an affected change spans operator paths, head geometry,
// attention/cache policy and modality; a proven device change also forces the
// device-path class.
const (
	classWeightType     = "weight-type"
	classOperator       = "operator"
	classHeadGeometry   = "head-geometry"
	classAttentionCache = "attention-cache"
	classModality       = "modality"
	classDevicePath     = "device-path"
)

// coverageClass is one affected dimension value the guard must exercise.
type coverageClass struct{ Dimension, Value string }

// modelFacts is one catalog model's resolved coverage classes, its accepted
// immutable-baseline disposition and its measured cost. The caller resolves
// these from the store, discovery and the manifest; the selector stays pure so
// it is host-only and unit-tested.
type modelFacts struct {
	Model      string
	Classes    []coverageClass
	BaselineOK bool   // baseline present, surface-matched and passing
	Regressed  bool   // the fresh-vs-baseline comparison failed
	Gap        string // per-model defect: untested device path or identity drift
	Cost       int64  // measured wall ns; the guard runs the cheapest cover first
}

// modelRegressionInputs are the resolved facts the selection reduces: the
// affected coverage classes for this commit, whether the change is host-only,
// whether device independence is unproven, and the catalog with per-model facts.
type modelRegressionInputs struct {
	HostOnly   bool
	DeviceFull bool
	Affected   []coverageClass
	Catalog    []modelFacts
}

type selectedModel struct {
	modelFacts
	Reason string
}

// modelRegressionPlan is the guard decision: the minimal cheapest model set to
// run, any rejection reasons that must fail the gate, and coverage classes no
// catalog model exercises.
type modelRegressionPlan struct {
	Inapplicable bool
	Run          []selectedModel
	Reasons      []string
	Uncovered    []coverageClass
}

// selectModelRegression reduces the resolved inputs to a minimal, deterministic
// model-regression guard plan. It is pure: no store, git or device access. A
// host-only change with no affected class is inapplicable; otherwise it greedily
// covers every affected class (and the device-path class when independence is
// unproven) with the cheapest models, rejecting any selected model whose
// baseline is stale, whose device path is untested, or that reproduces a
// regression.
func selectModelRegression(in modelRegressionInputs) modelRegressionPlan {
	required := append([]coverageClass(nil), in.Affected...)
	if in.DeviceFull {
		required = append(required, coverageClass{classDevicePath, "*"})
	}
	if in.HostOnly && len(required) == 0 {
		return modelRegressionPlan{Inapplicable: true, Run: []selectedModel{}}
	}
	catalog := append([]modelFacts(nil), in.Catalog...)
	slices.SortStableFunc(catalog, func(a, b modelFacts) int {
		return cmp.Or(cmp.Compare(a.Cost, b.Cost), cmp.Compare(a.Model, b.Model))
	})
	covers := func(model modelFacts, class coverageClass) bool {
		return slices.ContainsFunc(model.Classes, func(owned coverageClass) bool {
			return owned.Dimension == class.Dimension && (owned.Value == class.Value || class.Value == "*")
		})
	}
	uncovered := map[coverageClass]bool{}
	for _, class := range required {
		uncovered[class] = true
	}
	plan := modelRegressionPlan{Run: []selectedModel{}}
	for _, model := range catalog {
		if !slices.ContainsFunc(required, func(class coverageClass) bool { return uncovered[class] && covers(model, class) }) {
			continue // adds no still-uncovered class: minimize cost by skipping it.
		}
		for _, class := range required {
			if covers(model, class) {
				delete(uncovered, class)
			}
		}
		selected := selectedModel{modelFacts: model, Reason: "covers an affected class"}
		switch {
		case !model.BaselineOK:
			selected.Reason = "stale or mismatched baseline"
			plan.Reasons = append(plan.Reasons, model.Model+": "+selected.Reason)
		case model.Gap != "":
			selected.Reason = model.Gap
			plan.Reasons = append(plan.Reasons, model.Model+": "+model.Gap)
		case model.Regressed:
			selected.Reason = "reproduced throughput, memory or quality regression"
			plan.Reasons = append(plan.Reasons, model.Model+": "+selected.Reason)
		}
		plan.Run = append(plan.Run, selected)
	}
	for class := range uncovered {
		plan.Uncovered = append(plan.Uncovered, class)
		plan.Reasons = append(plan.Reasons, "missing coverage class "+class.Dimension+"="+class.Value)
	}
	slices.SortFunc(plan.Uncovered, func(a, b coverageClass) int {
		return cmp.Compare(a.Dimension+a.Value, b.Dimension+b.Value)
	})
	slices.Sort(plan.Reasons)
	return plan
}

// TestModelRegressSelectsMinimalCoverage selects the cheapest model set that
// covers every affected class, skipping redundant models, deterministically
// under input reordering.
func TestModelRegressSelectsMinimalCoverage(t *testing.T) {
	affected := []coverageClass{
		{classWeightType, "q8"}, {classWeightType, "fp8"}, {classHeadGeometry, "128"}, {classModality, "text"},
	}
	catalog := []modelFacts{
		{Model: "small-q8-text", Classes: []coverageClass{{classWeightType, "q8"}, {classModality, "text"}}, BaselineOK: true, Cost: 30},
		{Model: "geom-128", Classes: []coverageClass{{classHeadGeometry, "128"}}, BaselineOK: true, Cost: 40},
		{Model: "redundant-q8", Classes: []coverageClass{{classWeightType, "q8"}}, BaselineOK: true, Cost: 500},
		{Model: "fp8-only", Classes: []coverageClass{{classWeightType, "fp8"}}, BaselineOK: true, Cost: 90},
	}
	plan := selectModelRegression(modelRegressionInputs{Affected: affected, Catalog: catalog})
	if len(plan.Reasons) != 0 || len(plan.Uncovered) != 0 {
		t.Fatalf("clean selection had reasons=%v uncovered=%v", plan.Reasons, plan.Uncovered)
	}
	got := []string{}
	for _, selected := range plan.Run {
		got = append(got, selected.Model)
	}
	// small-q8-text (30) + geom-128 (40) + fp8-only (90) cover all four classes;
	// redundant-q8 adds nothing and is skipped. Cheapest-first ordering.
	want := []string{"small-q8-text", "geom-128", "fp8-only"}
	if !slices.Equal(got, want) {
		t.Fatalf("selected %v, want %v", got, want)
	}
	// Reordering the catalog must not change the deterministic result.
	slices.Reverse(catalog)
	again := selectModelRegression(modelRegressionInputs{Affected: affected, Catalog: catalog})
	regot := []string{}
	for _, selected := range again.Run {
		regot = append(regot, selected.Model)
	}
	if !slices.Equal(regot, want) {
		t.Fatalf("non-deterministic selection: %v vs %v", regot, want)
	}
}

// TestModelRegressRejects proves each mandated negative fails the guard rather
// than passing silently: a missing coverage class, a stale baseline, an
// untested device path, and a reproduced regression.
func TestModelRegressRejects(t *testing.T) {
	base := func() modelRegressionInputs {
		return modelRegressionInputs{
			Affected: []coverageClass{{classWeightType, "q8"}},
			Catalog:  []modelFacts{{Model: "m", Classes: []coverageClass{{classWeightType, "q8"}}, BaselineOK: true, Cost: 10}},
		}
	}
	for name, mutate := range map[string]func(*modelRegressionInputs){
		"missing coverage class": func(in *modelRegressionInputs) { in.Catalog = nil },
		"stale baseline":         func(in *modelRegressionInputs) { in.Catalog[0].BaselineOK = false },
		"untested device path": func(in *modelRegressionInputs) {
			in.DeviceFull = true
			in.Catalog[0].Classes = append(in.Catalog[0].Classes, coverageClass{classDevicePath, "*"})
			in.Catalog[0].Gap = "device path untested at the current surface"
		},
		"reproduced regression": func(in *modelRegressionInputs) { in.Catalog[0].Regressed = true },
	} {
		t.Run(name, func(t *testing.T) {
			in := base()
			mutate(&in)
			if plan := selectModelRegression(in); len(plan.Reasons) == 0 {
				t.Fatalf("%s did not reject: %+v", name, plan)
			}
		})
	}
}

// TestModelRegressHostOnlyInapplicable proves a host-only change that moves no
// model surface is inapplicable and runs nothing.
func TestModelRegressHostOnlyInapplicable(t *testing.T) {
	plan := selectModelRegression(modelRegressionInputs{HostOnly: true})
	if !plan.Inapplicable || len(plan.Run) != 0 {
		t.Fatalf("host-only change not inapplicable: %+v", plan)
	}
}

// TestModelRegressGateSelectionAndRejection binds the real gate selection and
// rejection path: OwnershipImpact triggers the guard on an inference/kernel
// change and excludes it on a host-only change, Plan admits the triggered check,
// and the check's runner fails when the selector yields rejection reasons.
func TestModelRegressGateSelectionAndRejection(t *testing.T) {
	const fact Fact = "capability:model-regression"
	build := func(in modelRegressionInputs) Check {
		return Check{
			Descriptor: Descriptor{
				Name: "model-regression", Phase: "test", Triggers: []Fact{fact},
				Inapplicable: "host-only change; no inference or kernel implementation moved",
				Resources:    []Resource{{Name: "measured-timing", Exclusive: true}},
				Requirements: Requirements{Candidate: true, Process: ProcessSuite},
				Ownership: Ownership{
					Fact: fact, Packages: []string{"internal/inference", "internal/modelrecipe"}, PackagePrefixes: []string{"internal/cuda"},
				},
			},
			Run: func(context.Context, Invocation) (bool, string, error) {
				plan := selectModelRegression(in)
				if plan.Inapplicable {
					return true, "host-only", nil
				}
				if len(plan.Reasons) > 0 {
					return false, "", errors.New(strings.Join(plan.Reasons, "; "))
				}
				return false, "selected " + strconv.Itoa(len(plan.Run)), nil
			},
		}
	}
	clean := modelRegressionInputs{
		Affected: []coverageClass{{classWeightType, "q8"}},
		Catalog:  []modelFacts{{Model: "m", Classes: []coverageClass{{classWeightType, "q8"}}, BaselineOK: true, Cost: 10}},
	}
	check := build(clean)

	// An inference change triggers the guard and is not excluded; Plan admits it.
	triggered := OwnershipImpact([]Check{check}, Surface{Identity: "candidate", Packages: []string{"internal/inference"}})
	if !slices.Contains(triggered.Facts, fact) {
		t.Fatalf("inference change did not trigger the guard: %+v", triggered)
	}
	if _, excluded := triggered.ExclusionReason("model-regression"); excluded {
		t.Fatal("triggered guard was also excluded")
	}
	planned, err := Plan([]Check{check}, triggered)
	if err != nil || !slices.ContainsFunc(planned, func(i Invocation) bool { return i.Check.Name == "model-regression" }) {
		t.Fatalf("plan did not admit the triggered guard: %v %+v", err, planned)
	}

	// A host-only change disjoint from the ownership excludes the guard.
	hostOnly := OwnershipImpact([]Check{check}, Surface{Identity: "candidate", Packages: []string{"internal/loop"}})
	if _, excluded := hostOnly.ExclusionReason("model-regression"); !excluded {
		t.Fatalf("host-only change did not exclude the guard: %+v", hostOnly)
	}

	// The rejection path: a regressed catalog fails the check's runner.
	regressed := clean
	regressed.Catalog = []modelFacts{{Model: "m", Classes: []coverageClass{{classWeightType, "q8"}}, BaselineOK: true, Regressed: true, Cost: 10}}
	if _, _, err := build(regressed).Run(t.Context(), Invocation{}); err == nil {
		t.Fatal("the guard runner accepted a reproduced regression")
	}
}
