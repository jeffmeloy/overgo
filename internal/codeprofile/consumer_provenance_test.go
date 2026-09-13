package codeprofile

import (
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestConsumerReferenceProvenanceAcceptance(t *testing.T) {
	const name = "internal/example/probe.go"
	const source = `package example
func Target() {}
func Generic[T any]() {}
func GenericValue() func() { return Generic[int] }
func Store() func() { return Target }
func Invoke() { Target(); Target(); (Target)(); Generic[int]() }
func Later() func() { return func() { Target() } }
func Async() { go Target(); defer Target() }
func Immediate() { func() { Target() }() }
type A struct{}
func (A) Target() {}
func Method(a A) { a.Target() }
type Number int
func Convert() Number { return Number(2) }
`
	snapshot, err := (repoanalysis.SourceSnapshot{}).Overlay(map[string][]byte{name: []byte(source)})
	if err != nil {
		t.Fatal(err)
	}
	selection := repoanalysis.BuildSelection{Context: "windows/amd64", Files: map[string]bool{name: true}, Packages: map[string]string{name: "overgo/internal/example"}}
	_, edges, _, err := ProductionConsumerGraph(snapshot, selection)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]struct {
		kind  string
		count int
	}{
		"Store": {"use", 1}, "Invoke": {"call", 4}, "Later": {"use", 1},
		"Async": {"use", 2}, "Immediate": {"use", 1}, "Method": {"interface", 1},
		"GenericValue": {"use", 1},
	}
	seen := map[string]int{}
	sites := map[int]bool{}
	for _, edge := range edges {
		if edge.To.Kind != "function" && edge.To.Kind != "method" {
			if edge.Kind != "use" {
				t.Fatalf("type use became an invocation: %+v", edge)
			}
			continue
		}
		expected, found := want[edge.From.Name]
		if !found || edge.Kind != expected.kind {
			t.Fatalf("unexpected relation: %+v", edge)
		}
		if edge.From.File != name || edge.Offset < 0 || edge.Offset >= len(source) || sites[edge.Offset] {
			t.Fatalf("source site missing or coalesced: %+v", edge)
		}
		sites[edge.Offset] = true
		prefix := edge.To.Name
		if edge.From.Name == "Method" {
			prefix = "a.Target"
		}
		if !strings.HasPrefix(source[edge.Offset:], prefix) || edge.Line != 1+strings.Count(source[:edge.Offset], "\n") {
			t.Fatalf("source site does not locate its reference: %+v", edge)
		}
		seen[edge.From.Name]++
	}
	for caller, expected := range want {
		if seen[caller] != expected.count {
			t.Errorf("%s references=%d want=%d", caller, seen[caller], expected.count)
		}
	}
}
