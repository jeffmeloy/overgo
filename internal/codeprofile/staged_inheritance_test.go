package codeprofile

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestStagedSurfaceInheritedLaneObligation(t *testing.T) {
	root := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		command := exec.Command("git", args...)
		command.Dir = root
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	git("init", "-q", "--initial-branch=main")
	git("config", "user.name", "fixture")
	git("config", "user.email", "fixture@example.invalid")
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(root, "docs", name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	plan := func(lane, scope, owner, state string) string {
		return fmt.Sprintf(`{"lane":%q,"scope":%q,"items":[{"id":"upstream","owner":%q,"status":"open","steps":[{"id":"consume","status":%q}]}]}`, lane, scope, owner, state)
	}
	manifest := func(reason, reference, inherited string) string {
		return fmt.Sprintf(`{"version":2,"inherited_from":%q,"staged":[{"package":"example/p","name":"API","reason":%q,"retire_with":%q}]}`, inherited, reason, reference)
	}
	commit := func() string {
		t.Helper()
		git("add", "docs")
		git("-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
		return git("rev-parse", "HEAD")
	}
	write("plan.json", plan("", "", "master", "open"))
	write("staged_surface.json", manifest("future consumer", "upstream/consume", ""))
	source := commit()
	write("plan.json", plan("", "", "master", "blocked: waiting"))
	blocked := commit()
	write("plan.json", plan("colibri", "", "colibri", "open"))
	owned := commit()
	write("plan.json", plan("", "", "", "open"))
	unassigned := commit()
	live := `{"lane":"colibri","scope":"lane","items":[]}`
	write("plan.json", live)
	write("staged_surface.json", manifest("future consumer", "upstream/consume", source))
	path := filepath.Join(root, "docs", "staged_surface.json")
	if _, err := LoadStagedSurface(path); err != nil {
		t.Fatalf("exact inherited foreign obligation refused: %v", err)
	}
	for _, test := range []struct{ name, plan, manifest string }{
		{"no proof", live, manifest("future consumer", "upstream/consume", "")},
		{"movable reference", live, manifest("future consumer", "upstream/consume", "HEAD")},
		{"changed declaration", live, manifest("different reason", "upstream/consume", source)},
		{"absent source symbol", live, strings.ReplaceAll(manifest("future consumer", "upstream/consume", source), `"API"`, `"Other"`)},
		{"wrong retirement", live, manifest("future consumer", "upstream/missing", source)},
		{"blocked source", live, manifest("future consumer", "upstream/consume", blocked)},
		{"same lane", live, manifest("future consumer", "upstream/consume", owned)},
		{"unassigned source", live, manifest("future consumer", "upstream/consume", unassigned)},
		{"not projected", `{"lane":"colibri","items":[]}`, manifest("future consumer", "upstream/consume", source)},
		{"ordinary plan", `{"items":[]}`, manifest("future consumer", "upstream/consume", source)},
		{"blocked current", plan("colibri", "lane", "colibri", "blocked: waiting"), manifest("future consumer", "upstream/consume", source)},
	} {
		t.Run(test.name, func(t *testing.T) {
			write("plan.json", test.plan)
			write("staged_surface.json", test.manifest)
			if _, err := LoadStagedSurface(path); err == nil {
				t.Fatal("invalid inherited declaration accepted")
			}
		})
	}
	// A later ordinary plan owns its canonical open task again. The retained
	// source pin grants no historical exception on that integration branch.
	write("plan.json", plan("", "", "master", "open"))
	write("staged_surface.json", manifest("future consumer", "upstream/consume", source))
	if _, err := LoadStagedSurface(path); err != nil {
		t.Fatalf("ordinary live canonical obligation refused: %v", err)
	}
}
