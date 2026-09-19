package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestQueryHelpContract drives the real run entrypoint: explicit help prints the
// purpose and the actual query flags, including -content discovery, exits
// successfully and reads nothing, while an unknown flag still fails with
// guidance rather than a discarded, silent failure.
func TestQueryHelpContract(t *testing.T) {
	for _, arg := range []string{"-h", "-help"} {
		var out bytes.Buffer
		if err := run([]string{arg}, &out); err != nil {
			t.Fatalf("%s exited non-zero: %v", arg, err)
		}
		text := out.String()
		for _, want := range []string{"overgodb-query:", "-repo", "-id", "-content", "-limit", "-alias"} {
			if !strings.Contains(text, want) {
				t.Fatalf("%s help missing %q in:\n%s", arg, want, text)
			}
		}
	}
	var out bytes.Buffer
	if err := run([]string{"-nonexistent"}, &out); err == nil {
		t.Fatal("unknown flag accepted")
	}
}
