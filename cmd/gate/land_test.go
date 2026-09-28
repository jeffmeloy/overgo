package main

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/landing"
	"overgo/internal/plan"
)

// TestSmallGateLandsAChange holds the small gate to its contract on a fixture
// module: a change whose checks pass lands as one commit that names its row
// and removes the row from the plan; a failing test, a message without its
// sections, and a new inline number are refused with HEAD left where it was.
func TestSmallGateLandsAChange(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		full := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git := func(arguments ...string) string {
		t.Helper()
		command := exec.Command("git", arguments...)
		command.Dir = repo
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", arguments, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	write("go.mod", "module fixture\n\ngo 1.26\n")
	write("internal/p/p.go", "// Package p is the fixture.\npackage p\n\n// Value is the fixture's value.\nfunc Value() int { return 1 }\n")
	write("internal/p/p_test.go", "package p\n\nimport \"testing\"\n\nfunc TestValue(t *testing.T) {\n\tif Value() != 1 {\n\t\tt.Fatal(Value())\n\t}\n}\n")
	// An unchanged file of the changed package that go fix rewrites.
	const legacy = "internal/p/legacy.go"
	write(legacy, "package p\n\n// Has reports whether values holds value.\nfunc Has(values []string, value string) bool {\n"+
		"\tfor _, candidate := range values {\n\t\tif candidate == value {\n\t\t\treturn true\n\t\t}\n\t}\n\treturn false\n}\n")
	row := func(id string) plan.Item {
		return plan.Item{ID: id, Title: id, Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Title: id, Status: plan.StatusOpen, Verify: "go test ./internal/p -run '^TestValue$' -count=1",
		}}}
	}
	document := plan.Plan{Campaign: "fixture", Doctrine: "fixture", Items: []plan.Item{row("first"), row("second")}}
	if err := os.MkdirAll(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := plan.Save(filepath.Join(repo, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	git("init", "-q")
	git("config", "user.name", "Fixture")
	git("config", "user.email", "fixture@example.invalid")
	git("add", "-A")
	git("commit", "-q", "-m", "base")
	message := []byte("Change the fixture\n\nCause: the fixture needs a change.\nPredicted effect: none.\n")
	land := func(reference string, message []byte) (string, error) {
		return landing.Run(t.Context(), landing.Options{Repo: repo, Ref: reference, Message: message, Log: io.Discard})
	}

	// The change only adds a test file: a package rule must still see the
	// package doc the unchanged source carries.
	write("internal/p/twice_test.go", "package p\n\nimport \"testing\"\n\nfunc TestValueTwice(t *testing.T) {\n\tif Value() != Value() {\n\t\tt.Fatal(Value())\n\t}\n}\n")
	if _, err := land("first/do", []byte("Change the fixture\n\nPredicted effect: none.\n")); err == nil || !strings.Contains(err.Error(), "Cause:") {
		t.Fatalf("a message without its cause landed: %v", err)
	}
	commit, err := land("first/do", message)
	if err != nil {
		t.Fatalf("a passing change was refused: %v", err)
	}
	if head := git("rev-parse", "HEAD"); head != commit {
		t.Fatalf("landed %s, HEAD is %s", commit, head)
	}
	if body := git("log", "-1", "--format=%B"); !strings.Contains(body, plan.LandedRowTrailer+": first/do") || !strings.Contains(body, "Measured by the gate:") {
		t.Fatalf("landing message = %q", body)
	}
	landed, err := plan.Load(filepath.Join(repo, filepath.FromSlash(plan.Path)))
	if err != nil || len(landed.Items) != 1 || landed.Items[0].ID != "second" {
		t.Fatalf("plan after the landing = %+v, %v", landed.Items, err)
	}
	if status := git("status", "--porcelain"); status != "" {
		t.Fatalf("the landing left the tree dirty: %q", status)
	}
	if fixed := git("show", "HEAD:"+legacy); !strings.Contains(fixed, "slices.Contains") {
		t.Fatalf("the file go fix rewrote did not land with the change:\n%s", fixed)
	}

	write("internal/p/p_test.go", "package p\n\nimport \"testing\"\n\nfunc TestValue(t *testing.T) {\n\tif Value() == 1 {\n\t\tt.Fatal(Value())\n\t}\n}\n")
	if _, err := land("second/do", message); err == nil || !strings.Contains(err.Error(), "test") {
		t.Fatalf("a failing test landed: %v", err)
	}
	git("checkout", "--", "internal/p/p_test.go")
	write("internal/p/p.go", "// Package p is the fixture.\npackage p\n\n// Value is the fixture's value.\nfunc Value() int { return 1 + 2 - 2 }\n")
	if _, err := land("second/do", message); err == nil || !strings.Contains(err.Error(), "new inline number") {
		t.Fatalf("a new inline number landed: %v", err)
	}
	if head := git("rev-parse", "HEAD"); head != commit {
		t.Fatalf("a refused landing moved HEAD to %s", head)
	}

	// A change to the plan alone runs only the row's verify.
	git("checkout", "--", "internal/p")
	edited, err := plan.Load(filepath.Join(repo, filepath.FromSlash(plan.Path)))
	if err != nil {
		t.Fatal(err)
	}
	edited.Items[0].Title = "second, retitled"
	if err := plan.Save(filepath.Join(repo, filepath.FromSlash(plan.Path)), edited); err != nil {
		t.Fatal(err)
	}
	if _, err := land("second/do", message); err != nil {
		t.Fatalf("a plan-only change was refused: %v", err)
	}
	if body := git("log", "-1", "--format=%B"); !strings.Contains(body, "Measured by the gate: verify ") || strings.Contains(body, "test ") {
		t.Fatalf("a plan-only landing ran more than the verify: %q", body)
	}
}
