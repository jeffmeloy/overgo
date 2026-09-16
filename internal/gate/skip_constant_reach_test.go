package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestSkipConstantReach pins two cuts of the process launcher's reach: a
// program the caller hands to a launcher is the caller's reach, so the
// launcher and its importers are not Go-source readers, and the shared
// skip reason lives in a leaf package, so a test that only names it inherits
// no launcher import. The fixture pins the rule; the live counts are the
// measured witness.
func TestSkipConstantReach(t *testing.T) {
	t.Parallel()
	t.Run("fixture", callerNamedProgramFixture)
	live := liveRepositoryFixture(t)
	g := live.context("internal/gate/preflight.go")
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range graph.nodes {
		if node.ForTest != "" || len(node.Match) == 0 {
			continue
		}
		switch node.ImportPath {
		case "overgo/internal/processcontrol", "overgo/internal/testevidence", "overgo/internal/cuda/driver":
			if node.sourceReader {
				t.Errorf("%s runs only programs its callers name yet is classed as a source reader", node.ImportPath)
			}
		case "overgo/internal/testskip":
			if len(node.Imports) != 0 {
				t.Errorf("testskip imports %v; it must stay a leaf", node.Imports)
			}
		}
	}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	total := len(scope.selected()) + scope.excluded
	// Measured 2026-09-14: a gate-only change excluded 52 packages before
	// the two cuts and 87 after; the ratchet holds a third.
	if scope.excluded < total/3 {
		t.Fatalf("gate-only change excluded %d of %d packages; the launcher reach is back", scope.excluded, total)
	}
	if slices.Contains(scope.selected(), "overgo/internal/gguf") {
		t.Fatalf("gguf, whose tests name only the skip reason, is selected by a gate-only change")
	}
	t.Logf("gate-only change: direct=%d uncertain=%d dependent=%d excluded=%d of %d", len(scope.direct), len(scope.uncertain), len(scope.dependent), scope.excluded, total)
}

// callerNamedProgramFixture: launcher runs the program its caller hands in,
// runner runs a program the environment names. An unrelated source change
// selects runner, whose program may be any repository command, and not
// launcher or its importer; the launcher stays an unnamed reach over data.
func callerNamedProgramFixture(t *testing.T) {
	g := scopeCompilerFixture(t)
	write := func(name, content string) {
		path := filepath.Join(g.repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/launcher/launcher.go", "package launcher\nimport \"os/exec\"\ntype Command struct { Path string; Args []string }\nfunc Run(command Command) error { return exec.Command(command.Path, command.Args...).Run() }\n")
	write("internal/launcher/launcher_test.go", "package launcher\nimport \"testing\"\nfunc TestRun(t *testing.T) { if err := Run(Command{Path: \"go\", Args: []string{\"version\"}}); err != nil { t.Skip(err) } }\n")
	write("internal/launchclient/client.go", "package launchclient\nimport \"overgo/internal/launcher\"\nfunc Run() error { return launcher.Run(launcher.Command{Path: \"go\", Args: []string{\"version\"}}) }\n")
	write("internal/launchclient/client_test.go", "package launchclient\nimport \"testing\"\nfunc TestRun(t *testing.T) { if err := Run(); err != nil { t.Skip(err) } }\n")
	write("internal/runner/runner.go", "package runner\nimport (\"os\"; \"os/exec\")\nfunc Run() error { return exec.Command(os.Getenv(\"TOOL\")).Run() }\n")
	write("internal/runner/runner_test.go", "package runner\nimport \"testing\"\nfunc TestRun(t *testing.T) { if err := Run(); err != nil { t.Skip(err) } }\n")
	runGitFixture(t, g.repo, "add", ".")
	g.paths = []string{"internal/recipe/recipe.go"}
	g.packageGraph = nil
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range graph.nodes {
		if node.ForTest != "" {
			continue
		}
		switch node.ImportPath {
		case "overgo/internal/launcher":
			if !node.opaqueReader || node.sourceReader {
				t.Errorf("launcher: opaque=%v source=%v; want an unnamed reach that is not a source reader", node.opaqueReader, node.sourceReader)
			}
		case "overgo/internal/runner":
			if !node.sourceReader {
				t.Error("runner runs a program the environment names yet is not a source reader")
			}
		}
	}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(scope.selected(), "overgo/internal/runner") {
		t.Fatalf("unnamed program runner left the scope of a source change: %v", scope.selected())
	}
	for _, pkg := range []string{"overgo/internal/launcher", "overgo/internal/launchclient"} {
		if slices.Contains(scope.selected(), pkg) {
			t.Fatalf("%s selected by a source change it neither compiles nor runs: %v", pkg, scope.selected())
		}
	}
}
