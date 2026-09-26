package gate

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

// Invocation wall includes build, execution, drain and receipt publication.
// Package elapsed values come from go test and can overlap; they are not wall.
type packageExecutionBatch struct {
	Step         string                          `json:"step,omitzero"`
	Short        bool                            `json:"short"`
	WallNS       uint64                          `json:"wall_ns"`
	Failed       bool                            `json:"failed"`
	Requested    []string                        `json:"requested"`
	Executions   []testevidence.PackageExecution `json:"executions"`
	Unobserved   []string                        `json:"unobserved"`
	StreamDigest string                          `json:"stream_digest,omitzero"`
	// Skipped names the fixture tests this invocation skipped, so the
	// uncredited evidence is a durable fact the review can act on.
	Skipped   []string        `json:"skipped,omitempty"`
	TestCosts []suiteTestCost `json:"-"`
}

// Costs retain one stream identity per invocation. No raw test output is copied.
type suiteTestCost struct {
	testevidence.PackageExecution
	Tests                 []testevidence.TestExecution `json:"tests"`
	SummedTopLevelSeconds float64                      `json:"summed_top_level_seconds"`
	Unmeasured            int                          `json:"unmeasured"`
	ChildExecutionSeconds *float64                     `json:"child_execution_seconds"`
	AssertionSeconds      *float64                     `json:"assertion_seconds"`
}

// The plan bounds this diagnostic to the two costliest measured packages.
// Parent/child elapsed may overlap. Top-level sums omit parallel-child work.
const suiteCostPackageLimit = 2

func suiteCostRanking(report testevidence.GoTestReport) []suiteTestCost {
	packages := slices.DeleteFunc(slices.Clone(report.Executions), func(execution testevidence.PackageExecution) bool {
		return !execution.Started || execution.Elapsed == nil || execution.Action != "pass" && execution.Action != "fail"
	})
	slices.SortFunc(packages, func(left, right testevidence.PackageExecution) int {
		return cmp.Or(cmp.Compare(*right.Elapsed, *left.Elapsed), strings.Compare(left.Package, right.Package))
	})
	tests := report.TestCosts()
	var result []suiteTestCost
	for _, execution := range packages[:min(len(packages), suiteCostPackageLimit)] {
		cost := suiteTestCost{PackageExecution: execution}
		for _, test := range tests {
			if test.Package != execution.Package {
				continue
			}
			cost.Tests = append(cost.Tests, test)
			if test.Elapsed == nil {
				cost.Unmeasured++
				continue
			}
			if !strings.Contains(test.Name, "/") {
				cost.SummedTopLevelSeconds += *test.Elapsed
			}
		}
		result = append(result, cost)
	}
	return result
}

var suiteCostContract = artifact.DocumentContract{
	Kind:      artifact.KindEvidence,
	MediaType: runrecord.SuiteCostMediaType,
	Schema:    runrecord.SuiteCostSchema,
}

// Retain compact costs in the gate's final atomic batch, including recovery debt.
// No child spans exist in this stream: subprocess/assertion attribution stays null.
func (g *gateContext) appendSuiteCost(batch *artifact.Batch, result artifact.ID) error {
	type invocation struct {
		packageExecutionBatch
		Suites []suiteTestCost `json:"suites"`
	}
	record := struct {
		Result      artifact.ID  `json:"result"`
		Invocations []invocation `json:"invocations"`
		Limitations string       `json:"limitations"`
	}{Result: result, Limitations: "Parent and child elapsed may overlap. Top-level sums are neither package wall nor complete work: parallel children need not appear in parent elapsed. Child-execution and assertion costs are unknown without child spans. Diagnostic observations grant no test or reuse credit."}
	for _, batch := range g.testExecutions {
		record.Invocations = append(record.Invocations, invocation{batch, batch.TestCosts})
	}
	if len(record.Invocations) == 0 {
		return nil
	}
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	content, err := suiteCostContract.ContentBytes(data)
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, content)
	batch.Lineage = append(batch.Lineage, artifact.Lineage{Child: content.Descriptor.ID, Parent: result, Relation: artifact.RelationDependsOn})
	g.suiteCost = content.Descriptor.ID
	g.advise(noteSuiteCost, fmt.Sprintf("suite cost ranking: evidence=%s result=%s invocations=%d bytes=%d; query with overgodb-query -repo <store> -id %s -content", content.Descriptor.ID, result, len(record.Invocations), len(data), content.Descriptor.ID))
	return nil
}

// suiteCostAlias names the suite cost the last finalized gate recorded.
const suiteCostAlias = "gate-suite-cost-current"

// orderByMeasuredCost starts the packages the last recorded suite cost
// measured longest first: go test schedules earlier arguments first, so the
// longest run no longer queues behind short ones. Unmeasured packages lead,
// their cost unknown; with no recorded cost the order is kept.
func (g *gateContext) orderByMeasuredCost(packages []string) []string {
	g.auditMutex.Lock()
	if g.packageCosts == nil {
		g.packageCosts = sync.OnceValue(func() map[string]float64 { return recordedPackageCosts(g.storePath) })
	}
	costs := g.packageCosts
	g.auditMutex.Unlock()
	cost := func(pkg string) float64 {
		if measured, found := costs()[pkg]; found {
			return measured
		}
		return math.Inf(1)
	}
	ordered := slices.Clone(packages)
	slices.SortStableFunc(ordered, func(left, right string) int { return cmp.Compare(cost(right), cost(left)) })
	return ordered
}

// recordedPackageCosts reads each package's longest elapsed from the suite
// cost the store's alias names; an absent or unreadable record is no cost.
func recordedPackageCosts(storePath string) map[string]float64 {
	costs := map[string]float64{}
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		return costs
	}
	defer store.Close()
	ctx := context.Background()
	id, found, err := artifact.ResolveAlias(ctx, store, suiteCostAlias)
	if err != nil || !found {
		return costs
	}
	content, found, err := artifact.ReadContent(ctx, store, id)
	var record struct {
		Invocations []struct {
			Executions []testevidence.PackageExecution
		}
	}
	if err != nil || !found || json.Unmarshal(content.Data, &record) != nil {
		return costs
	}
	for _, invocation := range record.Invocations {
		for _, execution := range invocation.Executions {
			if execution.Elapsed != nil {
				costs[execution.Package] = max(costs[execution.Package], *execution.Elapsed)
			}
		}
	}
	return costs
}

func (g *gateContext) recordPackageExecution(packages []string, short bool, wall time.Duration, report testevidence.GoTestReport, err error) {
	batch := packageExecutionBatch{
		Short: short, WallNS: uint64(wall.Nanoseconds()), Failed: err != nil,
		Requested: slices.Clone(packages), Executions: slices.Clone(report.Executions),
		StreamDigest: report.StreamDigest, Skipped: slices.Clone(report.Skipped), TestCosts: suiteCostRanking(report),
	}
	g.auditMutex.Lock()
	defer g.auditMutex.Unlock()
	batch.Step = g.testStep
	g.testExecutions = append(g.testExecutions, batch)
}

// canonicalPackage resolves a go-list selector to its import path against this
// candidate's package graph, never a guessed module name: an import path the
// graph already owns stays itself, a ./relative or matched selector becomes the
// production node's import path, and an unowned target reports false.
func (graph packageInputGraph) canonicalPackage(target string) (string, bool) {
	if len(graph.byID[target]) != 0 {
		return target, true
	}
	for _, node := range graph.nodes {
		relative, err := filepath.Rel(graph.root, node.Dir)
		if node.ForTest == "" && len(node.Match) != 0 && (slices.Contains(node.Match, target) || err == nil && target == "./"+filepath.ToSlash(relative)) {
			return node.ImportPath, true
		}
	}
	return "", false
}

// normalizePackageExecutions resolves go-list selectors without guessing module names.
func (graph packageInputGraph) normalizePackageExecutions(batches []packageExecutionBatch) []packageExecutionBatch {
	result := slices.Clone(batches)
	for index, batch := range result {
		var requested []string
		for _, target := range batch.Requested {
			if canonical, ok := graph.canonicalPackage(target); ok {
				requested = append(requested, canonical)
			} else {
				requested = append(requested, target)
			}
		}
		slices.Sort(requested)
		batch.Requested = slices.Compact(requested)
		batch.Unobserved = nil
		for _, target := range batch.Requested {
			if !slices.ContainsFunc(batch.Executions, func(execution testevidence.PackageExecution) bool { return execution.Package == target }) {
				batch.Unobserved = append(batch.Unobserved, target)
			}
		}
		result[index] = batch
	}
	return result
}

// Project the retained observations; selection and package-pass receipts stay separate.
func (g *gateContext) packageExecutionAudit() {
	if g.testPlan == nil {
		return
	}
	type profile struct {
		name  string
		short bool
	}
	awaiting := map[profile]bool{}
	for _, group := range []struct {
		packages        []string
		short, mayDefer bool
	}{
		{g.testPlan.edited, true, false}, {g.testPlan.remaining, true, true}, {g.testPlan.dependent, false, true},
	} {
		for _, name := range group.packages {
			awaiting[profile{name, group.short}] = group.mayDefer
		}
	}
	g.auditMutex.Lock()
	batches := slices.Clone(g.testExecutions)
	g.auditMutex.Unlock()
	started, unstarted := 0, 0
	outcomes := map[string]int{}
	for _, batch := range batches {
		for _, execution := range batch.Executions {
			if execution.Started {
				started++
			} else {
				unstarted++
			}
			outcomes[execution.Action]++
			delete(awaiting, profile{execution.Package, batch.Short})
		}
	}
	var deferred []string
	if g.deferLanes && len(awaiting) != 0 {
		if g.packageGraph == nil {
			g.note("package execution accounting: deferred classification requires the input graph")
			return
		}
		var candidates []string
		for profile, mayDefer := range awaiting {
			if mayDefer {
				candidates = append(candidates, profile.name)
			}
		}
		var err error
		deferred, err = g.packageGraph.devicePackages(candidates)
		if err != nil {
			g.note("package execution accounting: " + err.Error())
			return
		}
	}
	g.note(fmt.Sprintf("package test work: %d reused profiles; %d started attempts", g.testPlan.reused, started))
	g.note(fmt.Sprintf("package observations: passed=%d failed=%d skipped=%d interrupted=%d unstarted=%d; awaiting=%d deferred=%d; observations grant no evidence credit", outcomes["pass"], outcomes["fail"], outcomes["skip"], outcomes[""], unstarted, len(awaiting), len(deferred)))
}
