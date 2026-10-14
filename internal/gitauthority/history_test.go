package gitauthority

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestRepositoryEnvironmentRemovesGitOverrides(t *testing.T) {
	t.Setenv("OVERGO_GIT_AUTHORITY_SENTINEL", "preserved")
	t.Setenv("GIT_DIR", t.TempDir())
	t.Setenv("GIT_WORK_TREE", t.TempDir())
	t.Setenv("GIT_COMMON_DIR", t.TempDir())
	t.Setenv("GIT_OBJECT_DIRECTORY", t.TempDir())
	t.Setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", t.TempDir())
	t.Setenv("GIT_CONFIG_COUNT", "1")

	preserved := false
	for _, entry := range RepositoryEnvironment() {
		name, value, _ := strings.Cut(entry, "=")
		if strings.HasPrefix(strings.ToUpper(name), "GIT_") {
			t.Fatalf("Git override survived authority environment: %q", entry)
		}
		if name == "OVERGO_GIT_AUTHORITY_SENTINEL" && value == "preserved" {
			preserved = true
		}
	}
	if !preserved {
		t.Fatalf("non-Git environment was removed from authority command; process environment has %d entries", len(os.Environ()))
	}
}

func TestReaderEnvironmentDisablesOptionalLocksAfterRemovingGitOverrides(t *testing.T) {
	t.Setenv("OVERGO_GIT_AUTHORITY_SENTINEL", "preserved")
	t.Setenv("GIT_DIR", t.TempDir())
	t.Setenv("GIT_OPTIONAL_LOCKS", "1")

	preserved := false
	optionalLocks := 0
	for _, entry := range ReaderEnvironment() {
		name, value, _ := strings.Cut(entry, "=")
		switch {
		case strings.EqualFold(name, "GIT_OPTIONAL_LOCKS"):
			optionalLocks++
			if value != "0" {
				t.Fatalf("reader optional-lock policy = %q, want 0", value)
			}
		case strings.HasPrefix(strings.ToUpper(name), "GIT_"):
			t.Fatalf("Git override survived reader environment: %q", entry)
		case name == "OVERGO_GIT_AUTHORITY_SENTINEL" && value == "preserved":
			preserved = true
		}
	}
	if optionalLocks != 1 {
		t.Fatalf("reader optional-lock policy count = %d, want 1", optionalLocks)
	}
	if !preserved {
		t.Fatal("reader environment removed a non-Git variable")
	}
}

func TestRepositoryRootUsesExplicitRepository(t *testing.T) {
	repository := t.TempDir()
	foreign := t.TempDir()
	gitTestCommand(t, repository, "init", "-q")
	gitTestCommand(t, foreign, "init", "-q")
	t.Setenv("GIT_DIR", filepath.Join(foreign, ".git"))
	t.Setenv("GIT_WORK_TREE", foreign)
	if _, err := RepositoryRoot(t.Context(), repository); err != nil {
		t.Fatalf("explicit repository root was redirected by ambient Git state: %v", err)
	}
	subdirectory := filepath.Join(repository, "nested")
	if err := os.Mkdir(subdirectory, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := RepositoryRoot(t.Context(), subdirectory); err == nil ||
		!strings.Contains(err.Error(), "exact repository root") {
		t.Fatalf("repository subdirectory result = %v", err)
	}
}

func gitTestCommand(t *testing.T, repository string, arguments ...string) string {
	t.Helper()
	command := exec.Command("git", append([]string{"--no-replace-objects"}, arguments...)...)
	command.Dir = repository
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", arguments, err, output)
	}
	return strings.TrimSpace(string(output))
}
