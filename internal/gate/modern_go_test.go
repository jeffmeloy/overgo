package gate

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/repoanalysis"
)

func TestModernGoPreflightBindsCandidateCensus(t *testing.T) {
	repo, candidate := t.TempDir(), t.TempDir()
	if err := os.Mkdir(filepath.Join(repo, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, directory := range []string{"docs", "cmd/example", "internal/example"} {
		if err := os.MkdirAll(filepath.Join(candidate, directory), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for name, content := range map[string]string{
		"go.mod":                      "module example\n\ngo 1.26\n",
		"cmd/example/main.go":         "package main\nfunc main() {}\n",
		"internal/example/example.go": "package example\n",
	} {
		if err := os.WriteFile(filepath.Join(candidate, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	census, err := repoanalysis.BuildModernGoCensus(candidate, repoanalysis.ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := repoanalysis.BuildModernGoBaseline(census)
	if err != nil {
		t.Fatal(err)
	}
	published, err := repoanalysis.BuildModernGoPublishedCensus(census, baseline)
	if err != nil {
		t.Fatal(err)
	}
	// Correct working-tree authorities must not hide stale candidate files.
	if err := jsonfile.Write(filepath.Join(repo, repoanalysis.ModernGoBaselineFile), baseline, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := jsonfile.Write(filepath.Join(repo, repoanalysis.ModernGoPublishedCensusFile), published, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"valid", "stale", "missing", "malformed", "baseline"} {
		t.Run(mode, func(t *testing.T) {
			candidateBaseline, candidateCensus := baseline, published
			if mode == "stale" {
				candidateCensus.SourceIdentity = "stale"
			}
			if mode == "baseline" {
				candidateBaseline.SourceIdentity = "stale"
			}
			if err := jsonfile.Write(filepath.Join(candidate, repoanalysis.ModernGoBaselineFile), candidateBaseline, 0o644); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(candidate, repoanalysis.ModernGoPublishedCensusFile)
			if err := jsonfile.Write(path, candidateCensus, 0o644); err != nil {
				t.Fatal(err)
			}
			if mode == "missing" {
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "malformed" {
				if err := os.WriteFile(path, []byte("{"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			g := &gateContext{repo: repo, candidateRoot: candidate}
			if mode == "valid" {
				runGitFixture(t, candidate, "init")
				runGitFixture(t, candidate, "add", ".")
				runGitFixture(t, candidate, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "baseline")
				g.repo = candidate
			}
			skipped, err := g.stepModernGoRatchet()
			if skipped || (err != nil) != (mode != "valid") {
				t.Fatalf("candidate %s: skipped=%v err=%v", mode, skipped, err)
			}
		})
	}
}

func TestGateAlwaysRunsModernGoRatchet(t *testing.T) {
	checks := (&gateContext{}).pipelineChecks()
	for _, check := range checks {
		if check.Descriptor.Name == "modern-go" {
			if !check.Descriptor.Always {
				t.Fatal("modern-Go ratchet is not an always-run gate check")
			}
			return
		}
	}
	t.Fatal("modern-Go ratchet is absent from the gate pipeline")
}
