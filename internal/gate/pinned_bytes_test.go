package gate

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestPreflightNamesPinnedBytes holds the surface check to naming, before the
// commit, a planned change to bytes a record pins -- a coverage input, a
// measured harness (read with CRLF line ends normalized), a document whose
// digest a compatibility test passes as a literal -- with the pinning record,
// and to naming nothing for unpinned or unchanged files.
func TestPreflightNamesPinnedBytes(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	digest := func(text string) string { sum := sha256.Sum256([]byte(text)); return hex.EncodeToString(sum[:]) }
	coverage := `{"Scope": "fixture", "Inputs": {"internal/a/a_test.go": "` + digest("package a // pinned\n") + `"}}`
	for name, text := range map[string]string{
		"internal/a/a_test.go":                       "package a // edited\n",
		"internal/b/b_test.go":                       "package b\n",
		"internal/c/harness_test.go":                 "package c\r\n",
		"docs/verification/fixture-coverage.json":    coverage,
		"docs/image_video_capacity.json":             `{"source_files": {"internal/c/harness_test.go": {"evidence": "e", "sha256": "` + digest("package c\n") + `"}}}`,
		"cmd/compatibility/accepted_fixture_test.go": "package main\nfunc TestFixture(t *testing.T) { requireAcceptedDocument(t, root, \"docs/verification/fixture-coverage.json\", \"" + digest("the reviewed document") + "\", nil) }\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	stale, err := stalePins(root, []string{"internal/a/a_test.go", "internal/b/b_test.go", "internal/c/harness_test.go", "docs/verification/fixture-coverage.json"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"docs/verification/fixture-coverage.json (pinned by cmd/compatibility/accepted_fixture_test.go)",
		"internal/a/a_test.go (pinned by docs/verification/fixture-coverage.json)",
	}
	if !slices.Equal(stale, want) {
		t.Fatalf("stale pins %q, want %q", stale, want)
	}
	if stale, err := stalePins(root, []string{"internal/b/b_test.go"}); err != nil || len(stale) != 0 {
		t.Fatalf("an unpinned change named %q (%v)", stale, err)
	}
}
