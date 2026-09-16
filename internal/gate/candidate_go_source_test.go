package gate

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestCandidateGoPhasesExcludeIgnoredSource(t *testing.T) {
	if isolatedProcess(t) {
		return
	}
	t.Setenv("OVERGO_DATA_ROOT", "")
	t.Setenv("OVERGO_AUDIO_REFERENCE_STORE", "")
	repo := t.TempDir()
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
	for name, content := range map[string]string{
		"go.mod":            "module candidatefixture\n\ngo 1.25\n",
		".gitignore":        "scratch/\npkg/ignored.go\nexternal.txt\n",
		"pkg/value.go":      "package pkg\nconst Value = 1\n",
		"pkg/value_test.go": "package pkg\nimport (\"os\"; \"testing\")\nfunc TestCandidateValue(t *testing.T) { data,err := os.ReadFile(os.Getenv(\"OVERGO_DATA_ROOT\")+\"/external.txt\"); if err != nil || string(data) != \"declared fixture\" { t.Fatalf(\"external input: %s %v\",data,err) }; if Value != 2 { t.Fatal(Value) }; for _, path := range []string{\"ignored.go\", \"../scratch/poison.go\"} { if _,err := os.Stat(path); !os.IsNotExist(err) { t.Fatalf(\"ambient input %s: %v\", path,err) } } }\n",
	} {
		write(name, content)
	}
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "candidate@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Candidate Test")
	runGitFixture(t, repo, "add", ".")
	runGitFixture(t, repo, "commit", "-q", "-m", "source fixture")
	write("pkg/value.go", "package pkg\nconst Value = 2\n")
	for _, name := range []string{"scratch/poison.go", "pkg/ignored.go"} {
		write(name, "not Go source\n")
	}
	write("external.txt", "declared fixture")
	g := &gateContext{repo: repo, paths: []string{"pkg/value.go"}}
	tree, err := g.plannedTree()
	if err != nil {
		t.Fatal(err)
	}
	var firstIdentity string
	err = g.withCandidateWorktree(tree, func(root string) error {
		if _, err := g.stepBuild(); err != nil {
			return fmt.Errorf("candidate build: %w", err)
		}
		if _, err := g.stepVet(); err != nil {
			return fmt.Errorf("candidate vet: %w", err)
		}
		graph, err := g.inputGraph()
		if err != nil {
			return err
		}
		if graph.root != root {
			return fmt.Errorf("compiler graph uses ambient repository %s", graph.root)
		}
		identity, err := graph.identity("candidatefixture/pkg")
		if err != nil {
			return err
		}
		firstIdentity = identity.String()
		report, err := g.runGoTests(t.Context(), []string{"candidatefixture/pkg"}, false, nil)
		if err != nil {
			return err
		}
		if report.PassedTests != 1 {
			return fmt.Errorf("candidate test denominator = %d", report.PassedTests)
		}
		if _, err := g.executeCandidateVerifier(tree, "go test ./pkg -run '^TestCandidateValue$' -count=1"); err != nil {
			return err
		}
		return requireCandidateWorktreeUnchanged(root, tree)
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := g.withCandidateWorktree(tree, func(root string) error {
		graph, err := g.inputGraph()
		if err != nil {
			return err
		}
		identity, err := graph.identity("candidatefixture/pkg")
		if err != nil {
			return err
		}
		if identity.String() != firstIdentity {
			return fmt.Errorf("equivalent candidate lost input identity")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"scratch/poison.go", "pkg/ignored.go"} {
		data, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(name)))
		if err != nil || string(data) != "not Go source\n" {
			t.Fatalf("ambient scratch changed: %s, %v", name, err)
		}
	}
}
