package densecausal

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// TestDenseBlockRecordIsWhole reads the record of the dense block program,
// one block run and trained from one definition, measured beside the
// trainers it would replace and not accepted, and holds it to being whole.
//
// Every figure of the measurement is there, a count above nothing, the
// program's weights within its peak and its forward and backward within its
// step, for all seven projections and for two. Each finding the record
// states, of the program's figure beside a trainer's, is what the record's
// own figures say: a trainer that trains whole models is read on one block
// and on five, and a block's share is a quarter of the difference.
//
// The acceptance is what the findings and the sizes give. Time and memory
// are no worse only if neither like-for-like finding, the program's step on
// the host beside the LoRA trainer's and its passes on the device beside the
// resident trainer's, says longer; production code is smaller only if the
// program replaced more than it is; and the program is accepted only if
// every condition holds. And the record's word that the program was taken
// out is the tree's: neither the rules' package nor the block adapter is in
// it.
func TestDenseBlockRecordIsWhole(t *testing.T) {
	root := testutil.RepoRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "docs", "verification", "dense-block-program.txt"))
	if err != nil {
		t.Fatal(err)
	}
	fields := map[string]string{}
	for line := range strings.Lines(string(raw)) {
		if key, value, found := strings.Cut(strings.TrimSpace(line), ": "); found && !strings.Contains(key, " ") {
			fields[key] = value
		}
	}
	count := func(name string) int64 {
		t.Helper()
		value, err := strconv.ParseInt(fields[name], 10, 64)
		if err != nil || value < 0 {
			t.Fatalf("the record's %s is %q (%v), want a count", name, fields[name], err)
		}
		return value
	}
	figure := func(name string) int64 {
		t.Helper()
		value := count(name)
		if value == 0 {
			t.Fatalf("the record's %s is nothing", name)
		}
		return value
	}
	const one, many = 1, 5
	share := func(name, unit string) int64 {
		t.Helper()
		block := (figure(name+"-"+strconv.Itoa(many)+"-blocks-"+unit) - figure(name+"-"+strconv.Itoa(one)+"-blocks-"+unit)) / (many - one)
		if block <= 0 {
			t.Fatalf("the record's %s gives a block a share of %d", name, block)
		}
		return block
	}
	step, passes := figure("program-device-step-microseconds"), figure("program-device-passes-microseconds")
	narrowStep, narrowPasses := figure("program-device-step-query-value-microseconds"), figure("program-device-passes-query-value-microseconds")
	peak, weights := figure("program-device-peak-bytes"), figure("program-device-weights-bytes")
	if passes > step || narrowPasses > narrowStep || weights > peak {
		t.Fatalf("the program's forward and backward take %d of a step's %d, and %d of %d on two projections, and its weights are %d of a peak of %d: want each a part of the other", passes, step, narrowPasses, narrowStep, weights, peak)
	}
	if share("resident-step", "microseconds") < share("resident-passes", "microseconds") || share("resident-peak", "bytes") <= 0 {
		t.Fatal("the resident trainer's step for a block is shorter than its own forward and backward")
	}
	for _, finding := range []struct {
		name         string
		program, set int64
		under, over  string
	}{
		{"device-step-against-lora-block", step, share("lora-host-step", "microseconds"), "shorter", "longer"},
		{"host-step-against-lora-block", figure("program-host-step-microseconds"), share("lora-host-step", "microseconds"), "shorter", "longer"},
		{"device-passes-against-resident-block", passes, share("resident-passes", "microseconds"), "shorter", "longer"},
		{"device-passes-query-value-against-resident-block", narrowPasses, share("resident-passes", "microseconds"), "shorter", "longer"},
		{"device-step-against-resident-block", step, share("resident-step", "microseconds"), "shorter", "longer"},
		{"device-peak-against-resident-one-block", peak, figure("resident-peak-" + strconv.Itoa(one) + "-blocks-bytes"), "less", "more"},
	} {
		says := finding.over
		if finding.program < finding.set {
			says = finding.under
		}
		if fields[finding.name] != says {
			t.Fatalf("the record finds %s %q, and its figures, %d beside %d, say %q", finding.name, fields[finding.name], finding.program, finding.set, says)
		}
	}

	// The acceptance: what the findings and the sizes give.
	word := func(holds bool) string {
		if holds {
			return "yes"
		}
		return "no"
	}
	accepted := true
	for name, given := range map[string]string{
		"condition-block-is-the-models-own":                      fields["condition-block-is-the-models-own"],
		"condition-gradients-agree-with-directional-derivatives": fields["condition-gradients-agree-with-directional-derivatives"],
		"condition-adapter-learns-and-resumes":                   fields["condition-adapter-learns-and-resumes"],
		"condition-time-and-memory-no-worse":                     word(fields["host-step-against-lora-block"] != "longer" && fields["device-passes-against-resident-block"] != "longer"),
		"condition-production-code-smaller":                      word(count("replaced-production-nodes") > figure("program-production-nodes")),
	} {
		if fields[name] != given || given != "yes" && given != "no" {
			t.Fatalf("the record reads %s as %q, and what it records gives %q", name, fields[name], given)
		}
		accepted = accepted && given == "yes"
	}
	if fields["accepted"] != word(accepted) {
		t.Fatalf("the record says the program is accepted %q, and its conditions say %q", fields["accepted"], word(accepted))
	}

	// Taken out: neither the rules' package nor the block adapter is in the tree.
	out := true
	for _, path := range []string{filepath.Join("internal", "tensorgrad"), filepath.Join("internal", "densecausal", "block_program.go")} {
		if _, err := os.Stat(filepath.Join(root, path)); !errors.Is(err, fs.ErrNotExist) {
			out = false
		}
	}
	if fields["taken-out"] != word(out) || accepted == out {
		t.Fatalf("the record says the program is taken out %q and accepted %q, and the tree has it out %q: want a program that is not accepted out, and one that is accepted in", fields["taken-out"], fields["accepted"], word(out))
	}
}
