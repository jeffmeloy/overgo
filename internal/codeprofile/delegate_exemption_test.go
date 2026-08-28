package codeprofile

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

// TestDuplicateExcessExemptsPureDelegates proves the ratchet counts
// only consolidatable duplication: a clone group of pure delegates --
// each body one statement forwarding to a shared owner -- is listed
// with the Delegate mark and contributes zero excess, while a group
// with real duplicated bodies still counts -- so the metric
// measures debt, never the mandated forwarding idiom.
func TestDuplicateExcessExemptsPureDelegates(t *testing.T) {
	root := t.TempDir()
	for relative, content := range map[string]string{
		"internal/delegate_a.go": "package p\nfunc alphaOwner(v int) int { return v }\nfunc DelegateAlpha(v int) int { return alphaOwner(v) }\n",
		"internal/delegate_b.go": "package p\nfunc DelegateBeta(v int) int { return alphaOwner(v) }\n",
		"internal/dupe_a.go":     "package p\nfunc dupeFirst(v int) int { if v > 0 { v++ }; if v > 2 { v-- }; return v + 1 }\n",
		"internal/dupe_b.go":     "package p\nfunc dupeSecond(n int) int { if n > 0 { n++ }; if n > 2 { n-- }; return n + 1 }\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	profile, err := Build(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	var delegateGroup, duplicateGroup *Clone
	for index := range profile.Clones {
		for _, function := range profile.Clones[index].Functions {
			switch {
			case function == "internal/delegate_a.go:DelegateAlpha":
				delegateGroup = &profile.Clones[index]
			case function == "internal/dupe_a.go:dupeFirst":
				duplicateGroup = &profile.Clones[index]
			}
		}
	}
	if delegateGroup == nil || !delegateGroup.Delegate {
		t.Fatalf("delegate group missing or unmarked: %+v", profile.Clones)
	}
	if duplicateGroup == nil || duplicateGroup.Delegate {
		t.Fatalf("duplicated-body group must not be exempt: %+v", duplicateGroup)
	}
	if profile.DuplicateExcessNodes != duplicateGroup.Nodes {
		t.Fatalf("excess %d, want only the duplicated group's %d",
			profile.DuplicateExcessNodes, duplicateGroup.Nodes)
	}
}
