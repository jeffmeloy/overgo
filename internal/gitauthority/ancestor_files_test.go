package gitauthority

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAncestorFiles(t *testing.T) {
	root := t.TempDir()
	gitTestCommand(t, root, "init", "-q", "--initial-branch=main")
	gitTestCommand(t, root, "config", "user.name", "fixture")
	gitTestCommand(t, root, "config", "user.email", "fixture@example.invalid")
	if err := os.Mkdir(filepath.Join(root, "directory"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "directory", "entry"), []byte("tree fixture"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitTestCommand(t, root, "add", "directory")
	write := func(value string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(root, "contract.json"), []byte(value), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	commit := func() string {
		t.Helper()
		gitTestCommand(t, root, "add", "contract.json")
		gitTestCommand(t, root, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
		return gitTestCommand(t, root, "rev-parse", "HEAD")
	}
	write("source")
	source := commit()
	write("later")
	later := commit()
	files, err := AncestorFiles(t.Context(), root, source, "contract.json")
	if err != nil || len(files) != 1 || string(files[0]) != "source" {
		t.Fatalf("ancestor read = %q, %v", files, err)
	}
	gitTestCommand(t, root, "tag", "-a", "source-tag", "-m", "fixture", source)
	tag := gitTestCommand(t, root, "rev-parse", "source-tag")
	for _, revision := range []string{"HEAD", source[:12], tag, gitTestCommand(t, root, "rev-parse", source+":contract.json")} {
		if _, err := AncestorFiles(t.Context(), root, revision, "contract.json"); err == nil {
			t.Fatalf("accepted non-exact commit %q", revision)
		}
	}
	for _, name := range []string{"", ".", "../contract.json", "/contract.json", "docs/../contract.json", "C:/contract.json", "missing.json", "directory"} {
		if _, err := AncestorFiles(t.Context(), root, source, name); err == nil {
			t.Fatalf("accepted invalid or missing path %q", name)
		}
	}
	gitTestCommand(t, root, "checkout", "--orphan", "unrelated")
	write("unrelated")
	unrelated := commit()
	gitTestCommand(t, root, "checkout", "main")
	if _, err := AncestorFiles(t.Context(), root, unrelated, "contract.json"); err == nil {
		t.Fatal("accepted a non-ancestor")
	}
	gitTestCommand(t, root, "replace", source, later)
	if _, err := AncestorFiles(t.Context(), root, source, "contract.json"); err == nil {
		t.Fatal("accepted rewritten history")
	}
	gitTestCommand(t, root, "replace", "-d", source)
	foreign := t.TempDir()
	gitTestCommand(t, foreign, "init", "-q")
	t.Setenv("GIT_DIR", filepath.Join(foreign, ".git"))
	files, err = AncestorFiles(t.Context(), root, source, "contract.json")
	if err != nil || string(files[0]) != "source" {
		t.Fatalf("ambient Git override changed authority: %q, %v", files, err)
	}
}
