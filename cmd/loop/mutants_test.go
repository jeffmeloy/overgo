package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// mutantFixture is a module whose history lands two steps, each with one Go test as its check: strong adds Above,
// whose test holds its boundary on both sides, and weak adds Within, whose test calls it and asserts nothing of its
// comparison. It returns the module's root.
func mutantFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	git := func(arguments ...string) {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", arguments...)
		command.Dir = root
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", arguments, err, output)
		}
	}
	write := func(path, text string) {
		t.Helper()
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(filepath.Join(root, path)), 0o700))
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, path), []byte(text), 0o600))
	}
	steps := []plan.Step{
		{ID: "strong", Title: "Above holds its boundary", Status: plan.StatusOpen, Verify: "go test ./p -count=1 -run '^TestAbove$'"},
		{ID: "weak", Title: "Within is called", Status: plan.StatusOpen, Verify: "go test ./p -count=1 -run '^TestWithin$'"},
	}
	document := plan.Plan{Campaign: "mutants", Lane: "master-lead", Items: []plan.Item{{
		ID: "fixture", Title: "two steps", Owner: "master-lead", Status: plan.StatusOpen, Budget: plan.Budget{Nodes: 10, Reason: "a fixture"}, Steps: steps,
	}}}
	data := testcheck.Must(json.MarshalIndent(document, "", "  "))(t)
	testcheck.Must(plan.Parse(data))(t)
	git("init", "-q")
	git("config", "user.email", "fixture@example.invalid")
	git("config", "user.name", "fixture")
	write("go.mod", "module fixture\n\ngo 1.26\n")
	write("local-models.json", "{}\n")
	write("p/doc.go", "// Package p is a fixture.\npackage p\n")
	write("docs/plan.json", string(data))
	git("add", "-A")
	git("commit", "-q", "-m", "the plan")
	write("p/above.go", "package p\n\n// Above says v is above ten.\nfunc Above(v int) bool { return v > 10 }\n")
	write("p/above_test.go", "package p\n\nimport \"testing\"\n\nfunc TestAbove(t *testing.T) {\n\tif !Above(11) || Above(10) {\n\t\tt.Fatal(Above(11), Above(10))\n\t}\n}\n")
	git("add", "-A")
	git("commit", "-q", "-m", "fixture/strong: Above\n\n"+landedRowTrailer+": fixture/strong")
	write("p/within.go", "package p\n\n// Within says v is not zero.\nfunc Within(v int) bool { return v != 0 }\n")
	write("p/within_test.go", "package p\n\nimport \"testing\"\n\nfunc TestWithin(t *testing.T) {\n\t_ = Within(5)\n}\n")
	git("add", "-A")
	git("commit", "-q", "-m", "fixture/weak: Within\n\n"+landedRowTrailer+": fixture/weak")
	return root
}

// TestTheGateRefusesBadAlternatives puts landed steps to bad alternatives of their own changes: each comparison on a
// line a landing wrote in production Go, inverted in turn, at the commit that landed it, where its check passes as it
// stands. A check that holds the behaviour it landed refuses the alternative; one that holds nothing of it passes it,
// and the record names the alternative it passed, so the owner's question, whether the gate catches deliberately bad
// alternatives, is measured per step: here the strong step's check refuses 1 of 1, the weak step's 0 of 1, and the
// run prints the share. A step whose landing wrote no comparison, or whose check needs the card, is passed over.
func TestTheGateRefusesBadAlternatives(t *testing.T) {
	t.Parallel()
	root := mutantFixture(t)
	var out bytes.Buffer
	if err := mutateSteps(t.Context(), root, "2026-11-01", defaultReplayLimit, &out); err != nil {
		t.Fatalf("the run failed with %v, its records:\n%s", err, out.String())
	}
	var records []mutantRecord
	for line := range strings.Lines(out.String()) {
		var record mutantRecord
		testcheck.NoError(t, json.Unmarshal([]byte(line), &record))
		records = append(records, record)
	}
	at := func(step string) mutantRecord {
		index := slices.IndexFunc(records, func(record mutantRecord) bool { return record.Step == step })
		if index < 0 {
			t.Fatalf("the run recorded no %s: %+v", step, records)
		}
		return records[index]
	}
	if strong := at("fixture/strong"); strong.Mutants != 1 || strong.Refused != 1 || strong.Skipped != "" {
		t.Errorf("the strong step's check met %+v, want its one alternative refused", strong)
	}
	if weak := at("fixture/weak"); weak.Mutants != 1 || weak.Refused != 0 || len(weak.Survived) != 1 || !strings.Contains(weak.Survived[0], "p/within.go:4 != to ==") {
		t.Errorf("the weak step's check met %+v, want its one alternative passed and named", weak)
	}
	if !onCard("OVERGO_CUDA_TEST=1 go test -tags modeltest ./cmd/evaluate -count=1 -run '^TestA$'") || onCard("go test ./p -count=1 -run '^TestAbove$'") {
		t.Error("a check that needs the card is not told from one that does not")
	}
}
