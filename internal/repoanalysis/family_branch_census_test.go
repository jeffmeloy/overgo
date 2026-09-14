package repoanalysis

import (
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/jsonfile"
)

func TestFamilyBranchSnapshot(t *testing.T) {
	source := []byte(`package shared
func Match(a string) bool { switch a { case "gemma3": return true }; return a == "llama3" || a != "qwen2.5" }
`)
	const owner = "internal/shared/value.go"
	snapshot, err := (SourceSnapshot{}).Overlay(map[string][]byte{
		owner: source, "internal/model/architecture_catalog.go": source,
		"internal/testutil/fixture.go": source, "cmd/example/main_test.go": source,
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := FamilyBranchCensus(snapshot)
	want := []FamilyBranch{{owner, "gemma3", 1}, {owner, "llama3", 1}, {owner, "qwen2.5", 1}}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("branches=%+v error=%v", got, err)
	}
}

// Keep the repository's independent family-branch ceiling assertion.
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
	snapshot, err := DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	current, err := FamilyBranchCensus(snapshot)
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
