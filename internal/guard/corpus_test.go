package guard

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestBuildAdviceNamesBin holds the guard's advice to the place an executable
// belongs. A go build of one command with no output is refused, and the
// refusal says to build into bin/ and never into the scratch folder, which
// the next confirmed commit clears of whatever nothing holds, an executable
// among it.
func TestBuildAdviceNamesBin(t *testing.T) {
	t.Parallel()
	call := ToolCall{ToolName: "Bash"}
	call.ToolInput.Command = "go build ./cmd/plan"
	message, _ := Verdict(call, t.TempDir())
	if !strings.Contains(message, "-o bin/") || strings.Contains(message, "tmp/") {
		t.Fatalf("a go build with no output is answered with %q, want a refusal that says to build with -o bin/ and names no tmp/", message)
	}
}

// The corpus is the guard's non-vacuity check: every rule proves both a DENY
// on the dangerous shape and an ALLOW on the neighbouring safe command. A
// guard that blocks ordinary work gets worked around, and a worked-around
// guard protects nothing. Cases are commands actually issued in the lineage
// repos or minimal variants of one.

type corpusCase struct {
	Name    string `json:"name"`
	Tool    string `json:"tool,omitzero"`
	Command string `json:"command"`
	Expect  string `json:"expect"`
	Rule    string `json:"rule"`
}

type corpus struct {
	Note  string       `json:"_note"`
	Cases []corpusCase `json:"cases"`
}

// TestARawCommitDuringAMergeIsRefused: a merge lands through the gate like any change, so a merge in progress admits no raw
// commit, and the refusal says how to land it.
func TestARawCommitDuringAMergeIsRefused(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testcheck.NoError(t, os.Mkdir(filepath.Join(root, ".git"), 0o755))
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, ".git", "MERGE_HEAD"), []byte("deadbeef\n"), 0o644))
	call := ToolCall{ToolName: "Bash"}
	call.ToolInput.Command = "git commit --no-edit"
	if message, _ := Verdict(call, root); !strings.Contains(message, "cmd/loop -land") {
		t.Fatalf("a raw commit during a merge: %q", message)
	}
}

func TestGuardCorpus(t *testing.T) {
	raw, err := os.ReadFile("testdata/corpus.json")
	if err != nil {
		t.Fatalf("corpus missing: the guard has no non-vacuity check: %v", err)
	}
	var c corpus
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatalf("corpus unreadable: %v", err)
	}
	perRule := map[string]map[string]bool{}
	for _, tc := range c.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			root := t.TempDir()
			call := ToolCall{ToolName: tc.Tool}
			if call.ToolName == "" {
				call.ToolName = "Bash"
			}
			call.ToolInput.Command = tc.Command
			message, _ := Verdict(call, root)
			got := "ALLOW"
			if message != "" {
				got = "DENY"
			}
			if got != tc.Expect {
				t.Fatalf("expect %s got %s (message %q) for command %q", tc.Expect, got, message, tc.Command)
			}
		})
		if perRule[tc.Rule] == nil {
			perRule[tc.Rule] = map[string]bool{}
		}
		perRule[tc.Rule][tc.Expect] = true
	}
	for rule, seen := range perRule {
		if rule == "baseline" {
			continue
		}
		if !seen["ALLOW"] || !seen["DENY"] {
			t.Fatalf("rule %q lacks both verdicts: a one-directional corpus is vacuous", rule)
		}
	}
}
