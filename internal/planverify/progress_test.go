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
	if _, err := ExecuteShowing(t.Context(), directory, "go version", nil, &shown, nil); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(shown.String(), "verify: go version go") {
		t.Errorf("a command's own line was shown as %q; want it marked as the check's", shown.String())
	}
}

// TestACheckRunningNoGoTestIsTimedFromItsStart holds the overrun notice to every kind of check (overrun-every-check/do):
// a check that runs no go test of its own, as a browser check through cmd/webui-lane does, tells its caller it began as
// it starts, once, before its command has run, so the caller's clock times it; one that runs go test still tells its
// caller as each test binary starts, not before. began had been called only on go test's JSON start events, so a
// browser check's undo run hung 32 minutes on 2026-11-24 and its overrun notice never started.
func TestACheckRunningNoGoTestIsTimedFromItsStart(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	marker := filepath.Join(directory, "ran")
	began, ranBefore := 0, false
	if _, err := ExecuteShowing(t.Context(), directory, "touch ran", nil, nil, func() {
		began++
		_, err := os.Stat(marker)
		ranBefore = err == nil
	}); err != nil {
		t.Fatal(err)
	}
	if began != 1 || ranBefore {
		t.Errorf("a check running no go test told its caller it began %d times, after its command had run %t; want once, as it starts", began, ranBefore)
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "go.mod"), []byte("module example.com/probe\n\ngo 1.26\n"), 0o600))
	testcheck.NoError(t, os.MkdirAll(filepath.Join(directory, "p"), 0o755))
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "p", "p_test.go"), []byte("package p\n\nimport \"testing\"\n\nfunc TestA(t *testing.T) {}\n"), 0o600))
	began = 0
	if _, err := ExecuteShowing(t.Context(), directory, "go test ./p -count=1", nil, nil, func() { began++ }); err != nil {
		t.Fatal(err)
	}
	if began != 1 {
		t.Errorf("a go test check of one package told its caller it began %d times; want once, as its test binary started", began)
	}
}
