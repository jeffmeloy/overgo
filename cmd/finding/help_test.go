package main

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"overgo/internal/dataroot"
)

// TestFindingHelpContract drives the real runArgs entrypoint: explicit help
// prints the purpose and the actual create and disposition flags, exits
// successfully, opens no store and writes no finding, while an unknown flag
// still fails with guidance.
func TestFindingHelpContract(t *testing.T) {
	root := t.TempDir()
	t.Setenv(dataroot.Env, root)
	for _, arg := range []string{"-h", "-help"} {
		var out bytes.Buffer
		if err := runArgs([]string{arg}, &out); err != nil {
			t.Fatalf("%s exited non-zero: %v", arg, err)
		}
		text := out.String()
		for _, want := range []string{"finding:", "-title", "-severity", "-owner", "-evidence", "-closure", "-check", "-close", "-resolution"} {
			if !strings.Contains(text, want) {
				t.Fatalf("%s help missing %q in:\n%s", arg, want, text)
			}
		}
	}
	// Help must not have opened or created a store under the data root.
	if entries, err := os.ReadDir(root); err != nil || len(entries) != 0 {
		t.Fatalf("help touched the data root: %v %v", entries, err)
	}
	var out bytes.Buffer
	if err := runArgs([]string{"-nonexistent"}, &out); err == nil {
		t.Fatal("unknown flag accepted")
	}
}
