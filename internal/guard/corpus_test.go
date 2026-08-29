package guard

import (
	"encoding/json"
	"os"
	"testing"
)

// The corpus is the guard's non-vacuity check: every rule proves both a DENY
// on the dangerous shape and an ALLOW on the neighbouring safe command. A
// guard that blocks ordinary work gets worked around, and a worked-around
// guard protects nothing. Cases are commands actually issued in the lineage
// repos or minimal variants of one.

type corpusCase struct {
	Name            string `json:"name"`
	Tool            string `json:"tool,omitzero"`
	Command         string `json:"command"`
	Expect          string `json:"expect"`
	Rule            string `json:"rule"`
	GateEnv         bool   `json:"gate_env,omitzero"`
	MergeInProgress bool   `json:"merge_in_progress,omitzero"`
}

type corpus struct {
	Note  string       `json:"_note"`
	Cases []corpusCase `json:"cases"`
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
			gitDir := root + "/.git"
			if err := os.Mkdir(gitDir, 0o755); err != nil {
				t.Fatal(err)
			}
			if tc.MergeInProgress {
				if err := os.WriteFile(gitDir+"/MERGE_HEAD", []byte("deadbeef\n"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if tc.GateEnv {
				t.Setenv(GateEnv, "1")
			} else {
				t.Setenv(GateEnv, "")
			}
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
