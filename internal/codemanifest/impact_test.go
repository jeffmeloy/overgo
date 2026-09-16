package codemanifest

import (
	"testing"

	"overgo/internal/gosource"
)

func TestReverseClosure(t *testing.T) {
	base := completeFixtureManifest(t)
	candidateValue := fixtureManifest()
	candidateValue.Uncertainty = nil
	candidateValue.Symbols[1].BodySHA256 = "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	candidate, err := codec.New(candidateValue)
	if err != nil {
		t.Fatal(err)
	}
	delta, err := Diff(base, candidate)
	if err != nil {
		t.Fatal(err)
	}
	impact, err := Close(base, candidate, delta)
	if err != nil {
		t.Fatal(err)
	}
	if len(impact.Seeds) != 1 || len(impact.Reachable) != 2 || impact.Reachable[0].Name != "Run" || impact.Reachable[1].Name != "Apply" {
		t.Fatalf("impact seeds=%+v reachable=%+v", impact.Seeds, impact.Reachable)
	}
}

func TestInterfaceOverApproximation(t *testing.T) {
	root := t.TempDir()
	name := "internal/example/run.go"
	baseSource := "package example\ntype Worker interface { Apply() int }\ntype A struct{}\nfunc (A) Apply() int { return 1 }\ntype B struct{}\nfunc (B) Apply() int { return 2 }\nfunc Run(worker Worker) int { return worker.Apply() }\n"
	writeGeneratorFixture(t, root, name, baseSource)
	selection := gosource.BuildSelection{
		Context: "linux/amd64", Root: root, Files: map[string]bool{name: true},
		Packages: map[string]string{name: "overgo/internal/example"},
	}
	base := generateForDelta(t, root, selection)
	writeGeneratorFixture(t, root, name, "package example\ntype Worker interface { Apply() int }\ntype A struct{}\nfunc (A) Apply() int { return 3 }\ntype B struct{}\nfunc (B) Apply() int { return 2 }\nfunc Run(worker Worker) int { return worker.Apply() }\n")
	candidate := generateForDelta(t, root, selection)
	delta, err := Diff(base, candidate)
	if err != nil {
		t.Fatal(err)
	}
	impact, err := Close(base, candidate, delta)
	if err != nil {
		t.Fatal(err)
	}
	foundRun := false
	for _, symbol := range impact.Reachable {
		foundRun = foundRun || symbol.Name == "Run"
	}
	if !foundRun {
		t.Fatalf("interface caller absent from conservative closure: %+v", impact.Reachable)
	}
}

func TestUnknownClosureCannotExclude(t *testing.T) {
	base := completeFixtureManifest(t)
	candidateValue := fixtureManifest()
	candidateValue.Symbols[1].BodySHA256 = "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	candidateValue.Uncertainty = []Uncertainty{{
		Kind: UncertaintyCgo, Path: candidateValue.Symbols[1].File,
		Context: candidateValue.Symbols[1].ID.Context, Symbol: &candidateValue.Symbols[1].ID,
		Reason: "fixture cgo boundary",
	}}
	candidate, err := codec.New(candidateValue)
	if err != nil {
		t.Fatal(err)
	}
	delta, err := Diff(base, candidate)
	if err != nil {
		t.Fatal(err)
	}
	impact, err := Close(base, candidate, delta)
	if err != nil {
		t.Fatal(err)
	}
	if err := impact.ExclusionAuthority(); err == nil {
		t.Fatal("reachable cgo uncertainty authorized exclusion")
	}
}
