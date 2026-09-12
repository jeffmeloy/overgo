package gate

import (
	"path/filepath"
	"slices"
	"time"

	"overgo/internal/testevidence"
)

// Invocation wall includes build, execution, drain and receipt publication.
// Package elapsed values come from go test and can overlap; they are not wall.
type packageExecutionBatch struct {
	Short      bool                            `json:"short"`
	WallNS     uint64                          `json:"wall_ns"`
	Failed     bool                            `json:"failed"`
	Requested  []string                        `json:"requested"`
	Executions []testevidence.PackageExecution `json:"executions"`
	Unobserved []string                        `json:"unobserved"`
}

func (g *gateContext) recordPackageExecution(packages []string, short bool, wall time.Duration, report testevidence.GoTestReport, err error) {
	batch := packageExecutionBatch{
		Short: short, WallNS: uint64(wall.Nanoseconds()), Failed: err != nil,
		Requested: slices.Clone(packages), Executions: slices.Clone(report.Executions),
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
