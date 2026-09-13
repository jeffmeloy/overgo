package gate

import (
	"cmp"
	"encoding/json"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/testevidence"
)

// Invocation wall includes build, execution, drain and receipt publication.
// Package elapsed values come from go test and can overlap; they are not wall.
type packageExecutionBatch struct {
	Short        bool                            `json:"short"`
	WallNS       uint64                          `json:"wall_ns"`
	Failed       bool                            `json:"failed"`
	Requested    []string                        `json:"requested"`
	Executions   []testevidence.PackageExecution `json:"executions"`
	Unobserved   []string                        `json:"unobserved"`
	StreamDigest string                          `json:"stream_digest,omitzero"`
	TestCosts    []suiteTestCost                 `json:"-"`
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
	MediaType: "application/vnd.overgo.gate-suite-cost+json",
	Schema:    "overgo/gate-suite-cost/v1",
}

// Retain compact costs in the gate's final atomic batch, including recovery debt.
// No child spans exist in this stream: subprocess/assertion attribution stays null.
func (g *gateContext) appendSuiteCost(batch *artifact.Batch, result artifact.ID) error {
	type invocation struct {
		StreamDigest string          `json:"stream_digest"`
		Short        bool            `json:"short"`
		Failed       bool            `json:"failed"`
		WallNS       uint64          `json:"wall_ns"`
		Suites       []suiteTestCost `json:"suites"`
	}
	record := struct {
		Result      artifact.ID  `json:"result"`
		Invocations []invocation `json:"invocations"`
		Limitations string       `json:"limitations"`
	}{Result: result, Limitations: "Parent and child elapsed may overlap. Top-level sums are neither package wall nor complete work: parallel children need not appear in parent elapsed. Child-execution and assertion costs are unknown without child spans. Diagnostic observations grant no test or reuse credit."}
	for _, batch := range g.testExecutions {
		if len(batch.TestCosts) != 0 {
			record.Invocations = append(record.Invocations, invocation{batch.StreamDigest, batch.Short, batch.Failed, batch.WallNS, batch.TestCosts})
		}
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
	g.note(fmt.Sprintf("suite cost ranking: evidence=%s result=%s invocations=%d bytes=%d; query with overgodb-query -repo <store> -id %s -content", content.Descriptor.ID, result, len(record.Invocations), len(data), content.Descriptor.ID))
	return nil
}

func (g *gateContext) recordPackageExecution(packages []string, short bool, wall time.Duration, report testevidence.GoTestReport, err error) {
	batch := packageExecutionBatch{
		Short: short, WallNS: uint64(wall.Nanoseconds()), Failed: err != nil,
		Requested: slices.Clone(packages), Executions: slices.Clone(report.Executions),
		StreamDigest: report.StreamDigest, TestCosts: suiteCostRanking(report),
	}
	g.auditMutex.Lock()
	defer g.auditMutex.Unlock()
	g.testExecutions = append(g.testExecutions, batch)
}

// normalizePackageExecutions resolves go-list selectors without guessing module names.
func (graph packageInputGraph) normalizePackageExecutions(batches []packageExecutionBatch) []packageExecutionBatch {
	result := slices.Clone(batches)
	for index, batch := range result {
		var requested []string
		for _, target := range batch.Requested {
			if len(graph.byID[target]) != 0 {
				requested = append(requested, target)
				continue
			}
			found := false
			for _, node := range graph.nodes {
				relative, err := filepath.Rel(graph.root, node.Dir)
				if node.ForTest == "" && len(node.Match) != 0 && (slices.Contains(node.Match, target) || err == nil && target == "./"+filepath.ToSlash(relative)) {
					requested = append(requested, node.ImportPath)
					found = true
				}
			}
			if !found {
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
