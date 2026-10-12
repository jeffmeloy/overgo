package repoanalysis

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// pythonPathspecs select every Python file, wherever it is.
var pythonPathspecs = []string{"*" + PythonSuffix}

// validatePythonPlace refuses every Python file git tracks or would track if
// added. The repository holds no Python: a golden reference is data, and the
// script that wrote it stays on the machine that ran it, in the one ignored
// place.
func validatePythonPlace(ctx context.Context, root string) error {
	scripts, err := checkoutFiles(ctx, root, pythonPathspecs)
	if err != nil {
		return err
	}
	var findings []error
	for _, script := range scripts {
		findings = append(findings, fmt.Errorf("python file %s is tracked, or would be if added: the one place for Python is %s, which git ignores; move it there, or write the tool in Go", script, PythonPlace))
	}
	return errors.Join(findings...)
}

// TestPythonLivesOnlyInItsIgnoredPlace holds the checkout to the rule that the
// repository holds no Python: no Python file is tracked, and none lies where
// adding it would track it. The scripts that write a golden reference from a
// Python original live in the one directory git ignores, and a golden
// reference reaches the repository as data. A script committed beside the
// test it serves fails here, with the place it belongs.
//
// The rule is also held on a repository made for the purpose, so that it is
// known to refuse: a tracked script and an untracked one are both refused, and
// one in the ignored place is not.
func TestPythonLivesOnlyInItsIgnoredPlace(t *testing.T) {
	t.Parallel()
	if err := validatePythonPlace(t.Context(), "../.."); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	git := func(arguments ...string) {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", append([]string{"-C", root}, arguments...)...)
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(arguments, " "), err, output)
		}
	}
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	git("init", "--quiet")
	write(".gitignore", "/"+PythonPlace+"\n")
	write(PythonPlace+"golden.py", "print('a golden reference')\n")
	write("internal/model/model.go", "package model\n")
	if err := validatePythonPlace(t.Context(), root); err != nil {
		t.Fatalf("a script in the ignored place was refused: %v", err)
	}
	write("internal/model/testdata/golden.py", "print('tracked')\n")
	git("add", "internal/model/testdata/golden.py")
	write("tools/stray.py", "print('not yet added')\n")
	err := validatePythonPlace(t.Context(), root)
	if err == nil {
		t.Fatal("a tracked script and an untracked one outside the ignored place both passed")
	}
	for _, refused := range []string{"internal/model/testdata/golden.py", "tools/stray.py"} {
		if !strings.Contains(err.Error(), "python file "+refused+" ") {
			t.Errorf("%s was not refused: %v", refused, err)
		}
	}
	if strings.Contains(err.Error(), PythonPlace+"golden.py") {
		t.Errorf("the script in the ignored place was refused: %v", err)
	}
}
