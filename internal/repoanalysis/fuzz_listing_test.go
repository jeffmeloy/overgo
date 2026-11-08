package repoanalysis

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/testcheck"
)

// failingListingGo is a stub go: its listing of ./cmd/a names FuzzA, its listing of ./cmd/b fails as a package that
// does not build, and a fuzz run passes.
const failingListingGo = `#!/usr/bin/env bash
case "$*" in
  *-list*./cmd/a*) echo FuzzA; echo ok ;;
  *-list*./cmd/b*) echo "cmd/b: build failed" >&2; exit 1 ;;
  *) exit 0 ;;
esac
`

// TestTheFuzzJobFailsWhenAPackageCannotBeListed holds the scheduled fuzz job to every package it finds (SQA-318): the
// script listed each package's targets in a pipeline ending || true, so a package that failed to build listed nothing,
// the others kept the count above zero, and the job printed every target clean. Run on a tree of two fuzz packages,
// the second of which cannot be listed, it ends nonzero with no all-clean line.
func TestTheFuzzJobFailsWhenAPackageCannotBeListed(t *testing.T) {
	t.Parallel()
	shell := testcheck.Must(clioptions.POSIXShell())(t)
	tree, stub := t.TempDir(), t.TempDir()
	write := func(path, text string) {
		t.Helper()
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(text), 0o755))
	}
	write(filepath.Join(tree, filepath.FromSlash(fuzzScript)), string(testcheck.Must(os.ReadFile(filepath.Join(checkoutRoot, filepath.FromSlash(fuzzScript))))(t)))
	write(filepath.Join(tree, "cmd", "a", "a_test.go"), "package a\n\nfunc FuzzA(f *testing.F) {}\n")
	write(filepath.Join(tree, "cmd", "b", "b_test.go"), "package b\n\nfunc FuzzB(f *testing.F) {}\n")
	write(filepath.Join(stub, "go"), failingListingGo)
	script := exec.Command(shell, filepath.Join(tree, filepath.FromSlash(fuzzScript)), "1s")
	script.Env = append(os.Environ(), "PATH="+stub+string(os.PathListSeparator)+os.Getenv("PATH"))
	output, err := script.CombinedOutput()
	if err == nil || strings.Contains(string(output), "targets clean") {
		t.Fatalf("a fuzz package that cannot be listed ended %v with:\n%s\nwant the job to fail with no all-clean line", err, output)
	}
}
