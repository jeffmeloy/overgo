package gate

import (
	"fmt"
	"path/filepath"

	"overgo/internal/jsonfile"
	"overgo/internal/repoanalysis"
)

// The gate and its diagnostic projection use the same reviewed boundaries.
func (g *gateContext) architectureDiagnostics(snapshot repoanalysis.SourceSnapshot) error {
	consolidation, err := repoanalysis.AuditAgentHarnessConsolidation(snapshot)
	if err != nil {
		return err
	}
	if len(consolidation.Findings) != 0 {
		return fmt.Errorf("harness consolidation: %v", consolidation.Findings)
	}
	if _, err := g.harnessSurfaceUpdate(); err != nil {
		return err
	}
	var baseline struct {
		Version  int                         `json:"version"`
		Doc      string                      `json:"doc"`
		Branches []repoanalysis.FamilyBranch `json:"branches"`
	}
	if err := jsonfile.DecodeStrict(filepath.Join(g.repo, "docs/family_branch_baseline.json"), &baseline); err != nil {
		return err
	}
	allowed := map[[2]string]int{}
	for _, branch := range baseline.Branches {
		allowed[[2]string{branch.File, branch.Literal}] = branch.Count
	}
	current, err := repoanalysis.FamilyBranchCensus(snapshot)
	if err != nil {
		return err
	}
	for _, branch := range current {
		limit, known := allowed[[2]string{branch.File, branch.Literal}]
		if !known || branch.Count > limit {
			return fmt.Errorf("family-named branch exceeds reviewed count: %s %q %d -> %d; consult a declared profile policy", branch.File, branch.Literal, limit, branch.Count)
		}
	}
	return nil
}
