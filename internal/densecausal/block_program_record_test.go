// Package densecausal_test holds the dense model's tests that stand beside
// the trainers importing it, which a test inside the package cannot name.
package densecausal_test

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// TestDenseBlockDeviceRecordIsWhole reads the record of the block program's
// measurement on the device, beside the trainers it would replace, and holds
// it to being whole. Every figure the device test logs is there, a count
// above nothing, the program's weights within its peak and its forward and
// backward within its step. And each finding the record states, of the
// program's figure beside a trainer's, is what the record's own figures say:
// a trainer that trains whole models is read on one block and on five, and a
// block's share is a quarter of the difference.
func TestDenseBlockDeviceRecordIsWhole(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "docs", "verification", "dense-block-program-device.txt"))
	if err != nil {
		t.Fatal(err)
	}
	fields := map[string]string{}
	for line := range strings.Lines(string(raw)) {
		if key, value, found := strings.Cut(strings.TrimSpace(line), ": "); found && !strings.Contains(key, " ") {
			fields[key] = value
		}
	}
	figure := func(name string) int64 {
		t.Helper()
		value, err := strconv.ParseInt(fields[name], 10, 64)
		if err != nil || value <= 0 {
			t.Fatalf("the record's %s is %q (%v), want a count above nothing", name, fields[name], err)
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
	peak, weights := figure("program-device-peak-bytes"), figure("program-device-weights-bytes")
	if passes > step || weights > peak {
		t.Fatalf("the program's forward and backward take %d of a step's %d, and its weights are %d of a peak of %d: want each a part of the other", passes, step, weights, peak)
	}
	for _, finding := range []struct {
		name         string
		program, set int64
		under, over  string
	}{
		{"device-step-against-lora-block", step, share("lora-host-step", "microseconds"), "shorter", "longer"},
		{"host-step-against-lora-block", figure("program-host-step-microseconds"), share("lora-host-step", "microseconds"), "shorter", "longer"},
		{"device-passes-against-resident-block", passes, share("resident-passes", "microseconds"), "shorter", "longer"},
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
	if share("resident-step", "microseconds") < share("resident-passes", "microseconds") || share("resident-peak", "bytes") <= 0 {
		t.Fatal("the resident trainer's step for a block is shorter than its own forward and backward")
	}
}
