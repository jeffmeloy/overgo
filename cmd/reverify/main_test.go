package main

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// TestReverifyInvokesLiveCommands closes the rename-residue finding: a
// helper command reverify shells out to must exist in this tree, so a
// rename that deletes a command directory fails here instead of at
// evidence-recovery time.
func TestReverifyInvokesLiveCommands(t *testing.T) {
	source, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	invoked := regexp.MustCompile(`"\./(cmd/[a-z0-9-]+)"`).FindAllStringSubmatch(string(source), -1)
	if len(invoked) == 0 {
		t.Fatal("no invoked commands found; the recovery path changed shape")
	}
	for _, match := range invoked {
		path := filepath.Join("..", "..", filepath.FromSlash(match[1]))
		if info, err := os.Stat(path); err != nil || !info.IsDir() {
			t.Fatalf("reverify invokes %q which does not exist: %v", match[1], err)
		}
	}
}
