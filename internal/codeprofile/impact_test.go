package codeprofile

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestFunctionImpactChangedBodyReverseReachability(t *testing.T) {
	base, candidate, selection := impactSnapshots(t, `package p
func leaf() int { return 1 }
func middle() int { return leaf() }
func top() int { return middle() }
func unrelated() int { return 7 }
`, `package p
func leaf() int { return 2 }
func middle() int { return leaf() }
func top() int { return middle() }
func unrelated() int { return 7 }
`)
	impact, err := DeriveFunctionImpact(base, candidate, selection, selection)
	if err != nil {
		t.Fatal(err)
	}
	if impact.BaseIdentity == impact.CandidateIdentity || !symbolNamesEqual(impact.Seeds, []string{"leaf"}) ||
		!symbolNamesEqual(impact.Reachable, []string{"leaf", "middle", "top"}) {
		t.Fatalf("impact = %+v", impact)
	}
}

func TestFunctionImpactInterfaceNameOverApproximation(t *testing.T) {
	base, candidate, selection := impactSnapshots(t, `package p
type Runner interface { Run() int }
type A struct{}
type B struct{}
func (A) Run() int { return 1 }
func (B) Run() int { return 2 }
func invoke(value Runner) int { return value.Run() }
func other(value B) int { return value.Run() }
func top(value B) int { return other(value) }
`, `package p
type Runner interface { Run() int }
type A struct{}
type B struct{}
func (A) Run() int { return 3 }
func (B) Run() int { return 2 }
func invoke(value Runner) int { return value.Run() }
func other(value B) int { return value.Run() }
func top(value B) int { return other(value) }
`)
	impact, err := DeriveFunctionImpact(base, candidate, selection, selection)
	if err != nil {
		t.Fatal(err)
	}
	if len(impact.Seeds) != 1 || impact.Seeds[0].Receiver != "A" || impact.Seeds[0].Name != "Run" ||
		!symbolNamesEqual(impact.Reachable, []string{"Run", "invoke", "other", "top"}) {
		t.Fatalf("interface impact = %+v", impact)
	}
}

func impactSnapshots(t *testing.T, baseSource, candidateSource string) (repoanalysis.SourceSnapshot, repoanalysis.SourceSnapshot, repoanalysis.BuildSelection) {
	t.Helper()
	root := t.TempDir()
	const sourcePath = "p/p.go"
	absPath := filepath.Join(root, filepath.FromSlash(sourcePath))
	if err := os.MkdirAll(filepath.Dir(absPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(absPath, []byte(baseSource), 0o644); err != nil {
		t.Fatal(err)
	}
	base, err := repoanalysis.LoadGo(root, []string{sourcePath})
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := base.Overlay(map[string][]byte{sourcePath: []byte(candidateSource)})
	if err != nil {
		t.Fatal(err)
	}
	selection := repoanalysis.BuildSelection{
		Root: root, Files: map[string]bool{sourcePath: true},
		Packages: map[string]string{sourcePath: "example/p"},
	}
	return base, candidate, selection
}

func symbolNamesEqual(symbols []FunctionSymbol, names []string) bool {
	if len(symbols) != len(names) {
		return false
	}
	wanted := make(map[string]int, len(names))
	for _, name := range names {
		wanted[name]++
	}
	for _, symbol := range symbols {
		if wanted[symbol.Name] == 0 {
			return false
		}
		wanted[symbol.Name]--
	}
	return true
}
