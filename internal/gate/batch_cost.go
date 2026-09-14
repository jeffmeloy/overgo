package gate

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

type packageCostAttribution struct {
	packageInputAttribution
	RuntimeReaders []string `json:"runtime_readers"`
}

// checkpointCost records one acceptance step's outcome and wall.
type checkpointCost struct {
	Name       string                `json:"name"`
	Outcome    runrecord.StepOutcome `json:"outcome"`
	DurationNS uint64                `json:"duration_ns"`
}

// batchCost separates summed step time from acceptance execution and reuse.
type batchCost struct {
	Checkpoints []checkpointCost `json:"checkpoints"`
	StepNS      uint64           `json:"step_ns"`
	Failed      int              `json:"failed"`
	FailedNS    uint64           `json:"failed_ns"`
	Other       int              `json:"other"`
	OtherNS     uint64           `json:"other_ns"`
	Accepted    int              `json:"accepted"`
	Reused      int              `json:"reused"`
	ExecutedNS  uint64           `json:"executed_ns"`
}

func acceptanceStep(name string) bool {
	return name == "acceptance" || strings.HasPrefix(name, "acceptance-")
}

// batchCostOf sums step durations; overlapping steps are not elapsed wall.
// Failures of any phase are counted apart; executed
// non-acceptance steps are the other phases; checkpoints sort by name.
func batchCostOf(steps []runrecord.GateStep) batchCost {
	var cost batchCost
	for _, step := range steps {
		cost.StepNS += step.DurationNS
		if step.Outcome == runrecord.StepFailed {
			cost.Failed++
			cost.FailedNS += step.DurationNS
		}
		if !acceptanceStep(step.Name) {
			if step.Outcome == runrecord.StepSucceeded {
				cost.Other++
				cost.OtherNS += step.DurationNS
			}
			continue
		}
		cost.Checkpoints = append(cost.Checkpoints, checkpointCost{Name: step.Name, Outcome: step.Outcome, DurationNS: step.DurationNS})
		switch step.Outcome {
		case runrecord.StepSucceeded:
			cost.Accepted++
			cost.ExecutedNS += step.DurationNS
		case runrecord.StepReused:
			cost.Accepted++
			cost.Reused++
		}
	}
	slices.SortFunc(cost.Checkpoints, func(left, right checkpointCost) int { return strings.Compare(left.Name, right.Name) })
	return cost
}

// reuseSavings estimates step time avoided, not elapsed wall saved: for each reused
// checkpoint, the lower median executed duration of that checkpoint across
// prior costs (the conservative estimate on an even count); a reused
// checkpoint never executed before is returned as unmeasured.
func reuseSavings(prior []batchCost, current batchCost) (uint64, []string) {
	executed := map[string][]uint64{}
	for _, cost := range prior {
		for _, checkpoint := range cost.Checkpoints {
			if checkpoint.Outcome == runrecord.StepSucceeded {
				executed[checkpoint.Name] = append(executed[checkpoint.Name], checkpoint.DurationNS)
			}
		}
	}
	var saved uint64
	var unmeasured []string
	for _, checkpoint := range current.Checkpoints {
		if checkpoint.Outcome != runrecord.StepReused {
			continue
		}
		durations := executed[checkpoint.Name]
		if len(durations) == 0 {
			unmeasured = append(unmeasured, checkpoint.Name)
			continue
		}
		slices.Sort(durations)
		saved += durations[(len(durations)-1)/2]
	}
	return saved, unmeasured
}

// priorBatchCosts reads the costs of every prior gate result recorded by an
// attempt of the same plan row; attempts whose result is not a gate result
// are skipped.
func priorBatchCosts(ctx context.Context, store *overgodb.Store, planRef string) ([]batchCost, error) {
	item, step, bound := strings.Cut(planRef, "/")
	if !bound {
		return nil, fmt.Errorf("gate: batch cost requires an item/step plan reference, got %q", planRef)
	}
	history, err := runrecord.LoadAttemptHistory(ctx, store, runrecord.AttemptFilter{PlanItem: item})
	if err != nil {
		return nil, err
	}
	var costs []batchCost
	for _, attempt := range history.Attempts {
		if attempt.PlanStep != step || !attempt.Result.Valid() {
			continue
		}
		result, err := runrecord.RequireGateResult(ctx, store, attempt.Result)
		if err != nil {
			continue
		}
		costs = append(costs, batchCostOf(result.Steps))
	}
	return costs, nil
}

// batchCostAudit reports measured wall, summed work and estimated avoided work
// on every gate; without acceptance steps the accepted figures read zero.
func (g *gateContext) batchCostAudit(ctx context.Context, store *overgodb.Store, steps []runrecord.GateStep, wallNS uint64) {
	current := batchCostOf(steps)
	prior, err := priorBatchCosts(ctx, store, g.planRef)
	if err != nil {
		g.note("gate cost: prior attempts unavailable: " + err.Error())
		return
	}
	saved, unmeasured := reuseSavings(prior, current)
	g.note(fmt.Sprintf(
		"gate cost: total_wall=%s summed_step_time=%s failed=%d/%s other_phases=%d/%s accepted=%d reused=%d accepted_executed=%s estimated_step_time_avoided=%s prior_runs=%d unmeasured=%q",
		time.Duration(wallNS), time.Duration(current.StepNS), current.Failed, time.Duration(current.FailedNS), current.Other, time.Duration(current.OtherNS),
		current.Accepted, current.Reused, time.Duration(current.ExecutedNS), time.Duration(saved), len(prior), unmeasured,
	))
}

// dependencyCostAudit explains the costliest executed package group using the
// frozen graph. It reports attribution, never savings or new test evidence.
func (g *gateContext) dependencyCostAudit(steps []runrecord.GateStep) {
	if g.testPlan == nil || g.packageGraph == nil {
		return
	}
	began := time.Now()
	g.auditMutex.Lock()
	batches := g.packageGraph.normalizePackageExecutions(g.testExecutions)
	g.auditMutex.Unlock()
	g.retainSelectionCauses(batches)
	var selected runrecord.GateStep
	for _, step := range steps {
		if step.Name != "test-owners" && step.Name != "test-device" && step.Name != "test" {
			continue
		}
		if step.Outcome != runrecord.StepSucceeded && step.Outcome != runrecord.StepFailed && step.Outcome != runrecord.StepCancelled {
			continue
		}
		if step.DurationNS > selected.DurationNS || step.DurationNS == selected.DurationNS && step.Name < selected.Name {
			selected = step
		}
	}
	if selected.Name == "" {
		return
	}
	packages := slices.Clone(g.testPlan.edited)
	if selected.Name != "test-owners" {
		packages = slices.Concat(g.testPlan.remaining, g.testPlan.dependent)
		device, err := g.packageGraph.devicePackages(packages)
		if err != nil {
			g.note("test input attribution unavailable: " + err.Error())
			return
		}
		packages = slices.DeleteFunc(packages, func(target string) bool {
			return slices.Contains(device, target) != (selected.Name == "test-device")
		})
	}
	for _, batch := range batches {
		for _, execution := range batch.Executions {
			if len(g.packageGraph.byID[execution.Package]) != 0 {
				packages = append(packages, execution.Package)
			}
		}
	}
	slices.Sort(packages)
	packages = slices.Compact(packages)
	report := struct {
		Step       string                   `json:"step"`
		DurationNS uint64                   `json:"duration_ns"`
		AnalysisNS uint64                   `json:"analysis_ns"`
		Packages   []packageCostAttribution `json:"packages"`
		Readers    map[string]string        `json:"readers"`
		Executions []packageExecutionBatch  `json:"executions"`
	}{Step: selected.Name, DurationNS: selected.DurationNS, Readers: map[string]string{}, Executions: batches}
	for _, target := range packages {
		attribution, err := g.packageGraph.attributeInputs(target, g.paths)
		if err != nil {
			g.note("test input attribution unavailable: " + err.Error())
			return
		}
		attribution.Input = g.testPlan.directInputs[target]
		if !attribution.Input.Valid() {
			attribution.Input = g.testPlan.dependentInputs[target]
		}
		entry := packageCostAttribution{packageInputAttribution: attribution}
		for reader, reason := range attribution.RuntimeReaders {
			report.Readers[reader] = reason
			entry.RuntimeReaders = append(entry.RuntimeReaders, reader)
		}
		slices.Sort(entry.RuntimeReaders)
		report.Packages = append(report.Packages, entry)
	}
	report.AnalysisNS = uint64(time.Since(began).Nanoseconds())
	data, err := json.Marshal(report)
	if err != nil {
		g.note("test input attribution unavailable: " + err.Error())
		return
	}
	g.note("test input attribution: " + string(data))
}

// retainSelectionCauses attributes every package a test check requested,
// with its observed execution, for the retained selection-cause record; a
// package retried within one check keeps its observed execution.
func (g *gateContext) retainSelectionCauses(batches []packageExecutionBatch) {
	var packages []runrecord.SelectionPackage
	index := map[string]int{}
	for _, batch := range batches {
		if batch.Step == "" {
			continue
		}
		for _, target := range batch.Requested {
			if len(g.packageGraph.byID[target]) == 0 {
				continue
			}
			entry := runrecord.SelectionPackage{Package: target, Step: batch.Step}
			if position := slices.IndexFunc(batch.Executions, func(execution testevidence.PackageExecution) bool { return execution.Package == target }); position >= 0 {
				execution := batch.Executions[position]
				entry.Action, entry.Started, entry.ElapsedSeconds = execution.Action, execution.Started, execution.Elapsed
			}
			key := batch.Step + "\x00" + target
			if previous, seen := index[key]; seen {
				if entry.Started {
					packages[previous].Action, packages[previous].Started, packages[previous].ElapsedSeconds = entry.Action, entry.Started, entry.ElapsedSeconds
				}
				continue
			}
			attribution, err := g.packageGraph.attributeInputs(target, g.paths)
			if err != nil {
				g.note("selection causes not retained: " + err.Error())
				return
			}
			entry.Input = g.testPlan.directInputs[target]
			if !entry.Input.Valid() {
				entry.Input = g.testPlan.dependentInputs[target]
			}
			entry.CompilerInputs, entry.RuntimeInputs, entry.UnboundInputs = attribution.CompilerInputs, attribution.RuntimeInputs, attribution.UnboundInputs
			if len(attribution.RuntimeReaders) != 0 {
				entry.RuntimeReaders = attribution.RuntimeReaders
			}
			index[key] = len(packages)
			packages = append(packages, entry)
		}
	}
	g.selectionCauses = packages
}

// appendSelectionCauses retains the selection explanation beside the gate
// result so the histogram derives from retained records alone.
func (g *gateContext) appendSelectionCauses(batch *artifact.Batch, result artifact.ID) error {
	if len(g.selectionCauses) == 0 {
		return nil
	}
	record, err := runrecord.NewSelectionCauseRecord(runrecord.SelectionCauseRecord{
		Result: result, Changed: slices.Clone(g.paths), Packages: slices.Clone(g.selectionCauses),
		Limitations: "Attribution is package-level binding on the frozen candidate graph, not function reach. Elapsed values come from go test and overlap within a check.",
	})
	if err != nil {
		return err
	}
	content, err := record.Content()
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, content)
	batch.Lineage = append(batch.Lineage, artifact.Lineage{Child: content.Descriptor.ID, Parent: result, Relation: artifact.RelationDependsOn})
	g.note(fmt.Sprintf("selection causes: evidence=%s result=%s packages=%d; query with plan -history all -phases -result %s", content.Descriptor.ID, result, len(record.Packages), result))
	return nil
}
