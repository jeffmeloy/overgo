package codemanifest

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

func TestReferenceProvenanceAcceptance(t *testing.T) {
	const name = "internal/example/probe.go"
	const source = "package example\nfunc Target() {}\nfunc Hold() func() { return Target }\nfunc Invoke() { Target(); Target() }\n"
	snapshot, err := (repoanalysis.SourceSnapshot{}).Overlay(map[string][]byte{name: []byte(source)})
	if err != nil {
		t.Fatal(err)
	}
	selection := gosource.BuildSelection{Context: "windows/amd64", Files: map[string]bool{name: true}, Packages: map[string]string{name: "overgo/internal/example"}}
	base, err := Generate(snapshot, []gosource.BuildSelection{selection}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(base.References) != 3 {
		t.Fatalf("reference sites collapsed: %+v", base.References)
	}
	for _, edge := range base.References {
		want := ReferenceCall
		if edge.From.Name == "Hold" {
			want = ReferenceUse
		}
		if edge.Kind != want || edge.Line < 1 || edge.Offset < 0 || edge.Offset >= len(source) || !strings.HasPrefix(source[edge.Offset:], "Target") {
			t.Fatalf("reference provenance lost: %+v", edge)
		}
	}
	stored, err := base.Content()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := codec.Parse(stored.Data)
	if err != nil || restored.ID != base.ID || !slices.Equal(restored.References, base.References) {
		t.Fatalf("source provenance lost in storage: %v", err)
	}
	// Value references still propagate impact: a returned function can run later.
	changed, err := snapshot.Overlay(map[string][]byte{name: []byte(strings.Replace(source, "func Target() {}", "func Target() { panic(\"changed\") }", 1))})
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := Generate(changed, []gosource.BuildSelection{selection}, nil)
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
	for _, name := range []string{"Target", "Hold", "Invoke"} {
		if !slices.ContainsFunc(impact.Reachable, func(symbol SymbolID) bool { return symbol.Name == name }) {
			t.Fatalf("%s lost conservative impact: %+v", name, impact)
		}
	}
	for _, mutation := range []struct {
		name         string
		line, offset int
	}{
		{"negative line", -1, 0}, {"negative offset", 1, -1}, {"offset without line", 0, 3},
	} {
		t.Run(mutation.name, func(t *testing.T) {
			value := clone(base)
			value.References[0].Line, value.References[0].Offset = mutation.line, mutation.offset
			if _, err := codec.New(value); err == nil {
				t.Fatal("malformed source provenance accepted")
			}
		})
	}
	legacy := clone(base)
	legacy.Analyzer.Version = "consumer-graph-v1"
	legacy.References = legacy.References[:1]
	legacy.References[0].Line, legacy.References[0].Offset = 0, 0
	legacy, err = codec.New(legacy)
	if err != nil {
		t.Fatalf("historical edge rejected: %v", err)
	}
	content, err := legacy.Content()
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := codec.Parse(content.Data)
	if err != nil || replayed.ID != legacy.ID {
		t.Fatalf("historical edge replay: %v", err)
	}
	if legacy.Analyzer == base.Analyzer {
		t.Fatal("changed graph semantics reused analyzer identity")
	}
	delta, err = Diff(legacy, base)
	if err != nil {
		t.Fatal(err)
	}
	impact, err = Close(legacy, base, delta)
	if err != nil {
		t.Fatal(err)
	}
	if err := impact.ExclusionAuthority(); err == nil {
		t.Fatal("analyzer change silently authorized exclusion")
	}
}
