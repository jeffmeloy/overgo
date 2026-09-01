package model

import (
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
)

// TestArchitectureFamilyBranchCensus pins the model-abstraction boundary: a
// new model expressible through existing primitives introduces no
// family-named branch in shared execution code. The census enumerates every
// remaining branch outside catalogs, converters, fixtures, and
// presentation, and the reviewed baseline only shrinks -- a branch leaves
// when its check moves into a declared profile policy, and a new one
// refuses here.
func TestArchitectureFamilyBranchCensus(t *testing.T) {
	root := filepath.Join("..", "..")
	var baseline struct {
		Version  int            `json:"version"`
		Doc      string         `json:"doc"`
		Branches []FamilyBranch `json:"branches"`
	}
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs", "family_branch_baseline.json"), &baseline); err != nil {
		t.Fatal(err)
	}
	allowed := map[[2]string]int{}
	for _, branch := range baseline.Branches {
		allowed[[2]string{branch.File, branch.Literal}] = branch.Count
	}
	current, err := FamilyBranchCensus(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, branch := range current {
		limit, known := allowed[[2]string{branch.File, branch.Literal}]
		if !known {
			t.Errorf("new family-named branch: %s compares %q; consult a declared profile policy instead",
				branch.File, branch.Literal)
			continue
		}
		if branch.Count > limit {
			t.Errorf("family-named branch grew: %s %q %d -> %d", branch.File, branch.Literal, limit, branch.Count)
		}
	}
}
