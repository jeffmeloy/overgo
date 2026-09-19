package gate

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// preparedManifestFixture builds a valid, canonical manifest plan for one check.
func preparedManifestFixture(t *testing.T) automationcheck.ManifestPlan {
	t.Helper()
	check := automationcheck.Check{
		Descriptor: automationcheck.Descriptor{Name: "node", Phase: runrecord.PhaseTest, Always: true},
		Run:        func(context.Context, automationcheck.Invocation) (bool, string, error) { return false, "", nil },
	}
	invocations, err := automationcheck.Plan([]automationcheck.Check{check}, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	base, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("base"))
	candidate, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("candidate"))
	manifest, err := automationcheck.BindManifestPlan(base, candidate, strings.Repeat("a", 64), strings.Repeat("b", 64),
		automationcheck.Surface{Identity: "base:candidate"}, automationcheck.Impact{}, invocations)
	if err != nil {
		t.Fatal(err)
	}
	return manifest
}

// TestPreparedResumeAcceptance holds the prepared-selection persistence and the
// tree-mapping that governs its reuse: the compact selection round-trips through
// its file and re-verifies, an absent file reconstructs, an advanced-plan-only
// difference between the composed candidate and the landed tree preserves the
// selection, and any source difference forces reconstruction.
func TestPreparedResumeAcceptance(t *testing.T) {
	t.Parallel()

	t.Run("selection round trips and re-verifies", func(t *testing.T) {
		repo := t.TempDir()
		if err := os.MkdirAll(filepath.Join(repo, "tmp"), 0o755); err != nil {
			t.Fatal(err)
		}
		manifest := preparedManifestFixture(t)
		selection := preparedSelection{
			Version: artifact.InitialDocumentVersion, Preparation: testutil.ArtifactID(t, artifact.KindEvidence, "prep"),
			CandidateTree: strings.Repeat("c", 40), ManifestID: manifest.ID, Manifest: manifest,
		}
		if err := writeJSON(repo, gatePreparedSelectionFile, selection, clioptions.OutputFileMode); err != nil {
			t.Fatal(err)
		}
		got, ok, err := readPreparedSelection(repo)
		if err != nil || !ok {
			t.Fatalf("read prepared selection: ok=%v err=%v", ok, err)
		}
		got.Manifest.ID = got.ManifestID
		if got.Manifest.Validate() != nil || got.Manifest.ID != manifest.ID || got.Preparation != selection.Preparation {
			t.Fatalf("round trip lost the selection: %+v", got)
		}
	})

	t.Run("absent selection reconstructs", func(t *testing.T) {
		if _, ok, err := readPreparedSelection(t.TempDir()); ok || err != nil {
			t.Fatalf("absent selection: ok=%v err=%v", ok, err)
		}
	})

	t.Run("advanced plan reuses, source change reconstructs", func(t *testing.T) {
		repo := t.TempDir()
		write := func(name, content string) {
			path := filepath.Join(repo, filepath.FromSlash(name))
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		tree := func() string {
			runGitFixture(t, repo, "add", "-A")
			runGitFixture(t, repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "step")
			out, err := command(repo, "git", "rev-parse", "HEAD^{tree}")
			if err != nil {
				t.Fatal(err)
			}
			return strings.TrimSpace(out)
		}
		runGitFixture(t, repo, "init", "-q")
		write("docs/plan.json", "{\"campaign\":\"c\"}\n")
		write("internal/pkg/pkg.go", "package pkg\n")
		candidate := tree()

		write("docs/plan.json", "{\"campaign\":\"c\",\"advanced\":true}\n")
		g := &gateContext{repo: repo, fixedTree: tree()}
		if !g.landedTreeMatchesCandidate(candidate) {
			t.Fatal("an advanced-plan-only difference forced reconstruction")
		}

		write("internal/pkg/pkg.go", "package pkg\nfunc Added() {}\n")
		g.fixedTree = tree()
		if g.landedTreeMatchesCandidate(candidate) {
			t.Fatal("a source difference reused a stale selection")
		}
	})
}
