package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCandidateCheckoutOwnsIndexRefresh(t *testing.T) {
	t.Parallel()
	for _, refresh := range []string{"true", "false"} {
		t.Run(refresh, func(t *testing.T) {
			repo := t.TempDir()
			runGitFixture(t, repo, "init", "-q")
			runGitFixture(t, repo, "config", "user.email", "gate@example.invalid")
			runGitFixture(t, repo, "config", "user.name", "Gate Test")
			runGitFixture(t, repo, "config", "diff.autoRefreshIndex", refresh)
			if err := os.WriteFile(filepath.Join(repo, "planned.flag"), []byte("accepted"), 0o644); err != nil {
				t.Fatal(err)
			}
			runGitFixture(t, repo, "add", "--", "planned.flag")
			runGitFixture(t, repo, "commit", "-q", "-m", "base")
			g := gateContext{repo: repo, paths: []string{"planned.flag"}}
			tree, err := g.plannedTree()
			if err != nil {
				t.Fatal(err)
			}
			if _, err = g.executeCandidateVerifier(tree, `test "$(cat planned.flag)" = accepted`); err != nil {
				t.Fatalf("unchanged candidate depends on ambient index refresh: %v", err)
			}
			for _, verify := range []string{`printf changed > planned.flag`, `printf created > extra.flag`} {
				if _, err = g.executeCandidateVerifier(tree, verify); err == nil || !strings.Contains(err.Error(), "mutated its immutable candidate worktree") {
					t.Fatalf("mutation accepted: %v", err)
				}
			}
		})
	}
}
