package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestCheckingAStepJudgesItsCheckAsTheAttemptDoes holds plan -verify to judging a step's check as the commit attempt
// does (verify-as-gate): it ran the check once in the working folder, so a check that passes without the change passed
// it, and only an attempt, minutes to an hour of gate in, refused it. Over a change whose test passes with the change's
// production Go restored, -verify fails saying the attempt would refuse it; over one whose test catches the change, it
// passes, the undo run said.
func TestCheckingAStepJudgesItsCheckAsTheAttemptDoes(t *testing.T) {
	t.Parallel()
	const (
		value = "// Package p is a fixture.\npackage p\n\n// Value is the fixture's value.\nfunc Value() int { return 1 }\n"
		check = "go test ./p -run '^TestValue$' -count=1"
		asked = "package p\n\nimport \"testing\"\n\nfunc TestValue(t *testing.T) {\n\tt.Parallel()\n\tif Value() != %s {\n\t\tt.Fatal(Value())\n\t}\n}\n"
	)
	root, _ := editFixture(t, plan.Plan{Campaign: "fixture", Items: []plan.Item{
		{ID: "row", Title: "row", Steps: []plan.Step{{ID: "do", Title: "row", Verify: check}}},
	}})
	write := func(name, content string) {
		path := filepath.Join(root, filepath.FromSlash(name))
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	}
	write("go.mod", "module fixture\n\ngo 1.26\n")
	write("p/p.go", value)
	for _, arguments := range [][]string{{"init", "-q"}, {"add", "-A"},
		{"-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "plan"}} {
		testcheck.Must(commandInput(root, nil, "git", arguments...))(t)
	}

	write("p/p.go", value+"\n// Twice is twice the value.\nfunc Twice() int { return 2 }\n")
	write("p/p_test.go", strings.Replace(asked, "%s", "1", 1))
	said, err := planCommand(t, root, "-verify")
	if err == nil || !strings.Contains(said, "the commit attempt would refuse it") || !strings.Contains(said, "proves nothing") {
		t.Fatalf("plan -verify over a check that passes without the change ended %v, saying:\n%s\nwant it failed as the attempt refuses it", err, said)
	}

	write("p/p.go", strings.Replace(value, "return 1", "return 2", 1))
	write("p/p_test.go", strings.Replace(asked, "%s", "2", 1))
	said, err = planCommand(t, root, "-verify")
	if err != nil || !strings.Contains(said, "with p/p.go restored to ") || !strings.Contains(said, "judged as the commit attempt judges it") {
		t.Fatalf("plan -verify over a check that catches the change ended %v, saying:\n%s\nwant it passed, the undo run said", err, said)
	}
}
