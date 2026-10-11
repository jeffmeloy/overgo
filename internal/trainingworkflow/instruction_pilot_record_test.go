package trainingworkflow

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

// TestFirstRewardRunsRecordIsComplete reads the record of the first reward
// run, the instruction pilot on Qwen2.5-0.5B, beside the record of the
// acceptance it was judged by, and holds it to being whole.
//
// The run is of the acceptance that record fixed. Every arm of its training
// specification is there with every checkpoint the specification names, each
// with what it passed of the selection groups; the checkpoint frozen for an
// arm is one of them, and passed as many selection groups as any of the arm's,
// the earliest of equals. Every comparison the acceptance named is there with
// a verdict, the confirmation it is recorded under and the charge made for it,
// and the promotion part was left one confirmation and then none. And what the
// record says of adoption is what the adoption rule says of those verdicts:
// a record that is complete and does not adopt is a complete record.
func TestFirstRewardRunsRecordIsComplete(t *testing.T) {
	t.Parallel()
	read := func(name string) (lines []string, fields map[string]string) {
		raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs", "verification", name))
		if err != nil {
			t.Fatal(err)
		}
		fields = map[string]string{}
		for line := range strings.Lines(string(raw)) {
			line = strings.TrimSpace(line)
			lines = append(lines, line)
			if key, value, found := strings.Cut(line, ": "); found && !strings.Contains(key, " ") {
				fields[key] = value
			}
		}
		return lines, fields
	}
	_, fixed := read("instruction-pilot-qwen25-05b.txt")
	lines, run := read("instruction-pilot-run-qwen25-05b.txt")
	var acceptance evaluation.InstructionAcceptance
	var training InstructionTraining
	if err := strictjson.DecodeBytes([]byte(fixed["document"]), &acceptance); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(fixed["specification"]), &training); err != nil {
		t.Fatal(err)
	}
	if run["acceptance"] != fixed["acceptance"] || run["acceptance"] == "" {
		t.Fatalf("the run is of acceptance %q, and the acceptance fixed for it is %q", run["acceptance"], fixed["acceptance"])
	}
	if acceptance.ID, _ = artifact.ParseID(run["acceptance"]); !acceptance.ID.Valid() {
		t.Fatalf("the run names acceptance %q", run["acceptance"])
	}
	var selection, baseSelected int
	if _, err := fmt.Sscanf(run["base"], "passed %d of %d selection groups", &baseSelected, &selection); err != nil || selection == 0 || baseSelected > selection {
		t.Fatalf("the base's selection reading %q: %v", run["base"], err)
	}
	for _, arm := range training.Arms {
		passed := map[int]int{}
		for _, line := range lines {
			var groups, selected, of int
			if _, err := fmt.Sscanf(line, arm+" after %d groups: passed %d of %d selection groups;", &groups, &selected, &of); err == nil && of == selection {
				passed[groups] = selected
			}
		}
		var frozen int
		if _, err := fmt.Sscanf(run[arm], "frozen after %d groups,", &frozen); err != nil {
			t.Fatalf("the %s arm's frozen checkpoint %q: %v", arm, run[arm], err)
		}
		for _, groups := range training.Checkpoints {
			selected, read := passed[groups]
			if !read || selected > passed[frozen] || groups < frozen && selected == passed[frozen] {
				t.Fatalf("the %s arm's checkpoint after %d groups is read %t and passed %d, with the one after %d frozen at %d: want every checkpoint read and the first of the best frozen", arm, groups, read, selected, frozen, passed[frozen])
			}
		}
		if _, held := passed[frozen]; !held || len(passed) != len(training.Checkpoints) {
			t.Fatalf("the %s arm has %d checkpoints and froze the one after %d groups, want the specification's %v", arm, len(passed), frozen, training.Checkpoints)
		}
	}
	var answered []PilotComparison
	for index, named := range acceptance.Comparisons {
		for _, line := range lines {
			rest, found := strings.CutPrefix(line, fmt.Sprintf("comparison %q: ", named.Name))
			if !found {
				continue
			}
			verdict, identities, _ := strings.Cut(rest, "; ")
			var confirmation, charge string
			var left uint64
			if _, err := fmt.Sscanf(strings.ReplaceAll(identities, ",", ""), "confirmation %s charge %s %d left", &confirmation, &charge, &left); err != nil || left != uint64(len(acceptance.Comparisons)-index-1) {
				t.Fatalf("comparison %q is recorded as %q (%v), want its confirmation, its charge and %d left", named.Name, rest, err, len(acceptance.Comparisons)-index-1)
			}
			made := PilotComparison{Name: named.Name, Left: left, Confirmation: evaluation.PoolConfirmation{Acceptance: acceptance.ID, Verdict: verdict}}
			if made.Confirmation.ID, _ = artifact.ParseID(confirmation); !made.Confirmation.ID.Valid() {
				t.Fatalf("comparison %q names confirmation %q", named.Name, confirmation)
			}
			made.Confirmation.Charge, _ = artifact.ParseID(charge)
			answered = append(answered, made)
		}
	}
	if len(answered) != len(acceptance.Comparisons) {
		t.Fatalf("the run records %d of the acceptance's %d comparisons", len(answered), len(acceptance.Comparisons))
	}
	adopted, reason := pilotAdopted(acceptance, answered)
	if want := strings.TrimSpace(fmt.Sprintf("%t %s", adopted, reason)); run["adopted"] != want {
		t.Fatalf("the record says adopted %q, and the rule says %q of its verdicts", run["adopted"], want)
	}
	t.Logf("the first reward run: base passed %d of %d selection groups; %d comparisons; adopted %s", baseSelected, selection, len(answered), run["adopted"])
}
