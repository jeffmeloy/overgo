package landing

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// hookedRepository is a repository on master with the trunk's hook on, a commit behind its tip for reset and revert, and a
// branch side to merge from.
func hookedRepository(t *testing.T) (run func(environment []string, arguments ...string) (string, error), repo string) {
	t.Helper()
	repo = t.TempDir()
	run = func(environment []string, arguments ...string) (string, error) {
		command := exec.CommandContext(t.Context(), "git", arguments...)
		command.Dir, command.Env = repo, append(os.Environ(), environment...)
		output, err := command.CombinedOutput()
		return strings.TrimSpace(string(output)), err
	}
	setup := func(arguments ...string) {
		t.Helper()
		if output, err := run(nil, arguments...); err != nil {
			t.Fatalf("git %v: %v: %s", arguments, err, output)
		}
	}
	setup("init", "-q", "-b", "master")
	setup("config", "user.name", "fixture")
	setup("config", "user.email", "fixture@example.invalid")
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("code.go", "first\n")
	setup("add", "-A")
	setup("commit", "-q", "-m", "first")
	write("code.go", "second\n")
	setup("commit", "-q", "-a", "-m", "second")
	setup("branch", "side")
	setup("checkout", "-q", "side")
	write("side.go", "side\n")
	setup("add", "-A")
	setup("commit", "-q", "-m", "side")
	setup("checkout", "-q", "master")
	hook, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), filepath.FromSlash(trunkHookDirectory), trunkHook))
	if err != nil {
		t.Fatal(err)
	}
	write(filepath.Join(trunkHookDirectory, trunkHook), string(hook))
	setup("update-index", "--refresh")
	setup("config", "core.hooksPath", trunkHookDirectory)
	return run, repo
}

// TestOnlyALandingMovesTheTrunkInGit: every verb that moves the trunk meets the hook, --no-verify among them, for a worker
// on any harness. A marked landing, another branch and a commit of the register alone pass.
func TestOnlyALandingMovesTheTrunkInGit(t *testing.T) {
	t.Parallel()
	for name, arguments := range map[string][]string{
		"commit":     {"commit", "-q", "--allow-empty", "-m", "raw"},
		"no-verify":  {"commit", "-q", "--no-verify", "--allow-empty", "-m", "raw"},
		"merge":      {"merge", "-q", "--no-ff", "--no-edit", "side"},
		"cherrypick": {"cherry-pick", "side"},
		"revert":     {"revert", "--no-edit", "HEAD"},
		"rebase":     {"rebase", "side"},
		"reset":      {"reset", "-q", "--hard", "HEAD~1"},
		"update-ref": {"update-ref", "refs/heads/master", "side"},
		"delete":     {"branch", "-D", "master"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			run, _ := hookedRepository(t)
			before, err := run(nil, "rev-parse", "refs/heads/master")
			if err != nil {
				t.Fatal(err)
			}
			if name == "delete" {
				if output, err := run(nil, "checkout", "-q", "--detach"); err != nil {
					t.Fatalf("%v: %s", err, output)
				}
			}
			if output, err := run(nil, arguments...); err == nil {
				t.Fatalf("git %v moved the trunk: %s", arguments, output)
			}
			if after, err := run(nil, "rev-parse", "refs/heads/master"); err != nil || after != before {
				t.Fatalf("the trunk went from %s to %s, %v", before, after, err)
			}
		})
	}
	run, repo := hookedRepository(t)
	allowed := func(what string, environment []string, arguments ...string) {
		t.Helper()
		if output, err := run(environment, arguments...); err != nil {
			t.Fatalf("%s was refused: %v: %s", what, err, output)
		}
	}
	register := filepath.Join(repo, "docs", "sqa_findings.json")
	if err := os.MkdirAll(filepath.Dir(register), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(register, []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	allowed("staging the register", nil, "add", "docs/sqa_findings.json")
	allowed("a commit of the register", nil, "commit", "-q", "-m", "sqa: a finding")
	allowed("a new branch", nil, "checkout", "-q", "-b", "elsewhere")
	allowed("a commit on another branch", nil, "commit", "-q", "--allow-empty", "-m", "elsewhere")
	allowed("a marked landing", []string{trunkMoveEnvironment + "=1"}, "update-ref", "refs/heads/master", "side")
}

// TestALandingRefusesATrunkWithoutItsHook: nothing in a checkout turns the hook on, so a landing in a repository that
// carries it says how, and one that carries none is not asked.
func TestALandingRefusesATrunkWithoutItsHook(t *testing.T) {
	t.Parallel()
	run, repo := hookedRepository(t)
	check := func() error {
		l := &landing{Options: Options{Repo: repo, Log: io.Discard}}
		return l.hooked(context.Background())
	}
	if err := check(); err != nil {
		t.Fatalf("with the hook on: %v", err)
	}
	if output, err := run(nil, "config", "--unset", "core.hooksPath"); err != nil {
		t.Fatalf("%v: %s", err, output)
	}
	if err := check(); err == nil || !strings.Contains(err.Error(), "git config core.hooksPath "+trunkHookDirectory) {
		t.Fatalf("with the hook off: %v", err)
	}
	if err := os.RemoveAll(filepath.Join(repo, filepath.FromSlash(trunkHookDirectory))); err != nil {
		t.Fatal(err)
	}
	if err := check(); err != nil {
		t.Fatalf("a repository with no hook: %v", err)
	}
}
