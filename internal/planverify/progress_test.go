package planverify

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestACheckShowsItsProgress holds a check that reports where its caller watches: each package's verdict and each failed
// test of a go test, and each line a command prints of its own, marked as the check's, with none of go test's JSON
// itself. A cycle's lanes ran six hours as a landing's check and printed nothing until a page answered a journey with a
// refusal (2026-11-14). The fixture has a package that passes and one whose test fails. A caller timing the tests apart
// from their build is told as each test binary starts (undo-overrun/do), and a command running none tells it nothing.
func TestACheckShowsItsProgress(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	for name, content := range map[string]string{
		"go.mod":                "module example.com/probe\n\ngo 1.26\n",
		"passes/passes_test.go": "package passes\n\nimport \"testing\"\n\nfunc TestShown(t *testing.T) {}\n",
		"fails/fails_test.go":   "package fails\n\nimport \"testing\"\n\nfunc TestShown(t *testing.T) { t.Fatal(\"the page answered with a refusal\") }\n",
	} {
		path := filepath.Join(directory, filepath.FromSlash(name))
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(content), 0o600))
	}
	var shown bytes.Buffer
	began := 0
	if _, err := ExecuteShowing(t.Context(), directory, "go test ./... -count=1 -p=1 -run '^TestShown$'", nil, &shown, func() { began++ }); err == nil {
		t.Fatal("a check whose test fails passed")
	}
	if began != 2 {
		t.Errorf("the check's two test binaries were said to begin %d times; want twice, once as each started", began)
	}
	said := shown.String()
	for _, want := range []string{"verify: pass\texample.com/probe/passes\t", "verify: fail\texample.com/probe/fails\tTestShown", "verify: fail\texample.com/probe/fails\t"} {
		if !strings.Contains(said, want) {
			t.Errorf("the check showed %q; want it to show %q", said, want)
		}
	}
	if strings.Contains(said, `"Action"`) {
		t.Errorf("the check showed go test's JSON itself: %q", said)
	}
	shown.Reset()
	if _, err := ExecuteShowing(t.Context(), directory, "go version", nil, &shown, func() { began++ }); err != nil {
		t.Fatal(err)
	}
	if began != 2 {
		t.Error("a command that runs no test binary was said to begin one")
	}
	if !strings.HasPrefix(shown.String(), "verify: go version go") {
		t.Errorf("a command's own line was shown as %q; want it marked as the check's", shown.String())
	}
}
