package main

import (
	"bufio"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheE4BProducerRunsUnderItsTestsTags holds image-presets: the E4B projection cell's producer runs its test under
// every tag the test's file builds with, so the test runs; the full measurement of 2026-11-24 ran it without modeltest,
// no test ran, and the cell was refused, "overgo/internal/server is absent: no test the run selected passed there".
func TestTheE4BProducerRunsUnderItsTestsTags(t *testing.T) {
	t.Parallel()
	file := testcheck.Must(os.Open(filepath.Join(testutil.RepoRoot(t), "internal", "server", "e4b_modalities_windows_test.go")))(t)
	defer file.Close()
	line := bufio.NewScanner(file)
	if !line.Scan() {
		t.Fatal("the producer's test file is empty")
	}
	constraint, built := strings.CutPrefix(line.Text(), "//go:build ")
	if !built {
		return
	}
	tags := ""
	if at := slices.Index(protocolProducer, "-tags"); at >= 0 && at+1 < len(protocolProducer) {
		tags = "," + protocolProducer[at+1] + ","
	}
	for _, term := range strings.FieldsFunc(constraint, func(r rune) bool { return r == ' ' || r == '&' || r == '(' || r == ')' }) {
		if term != "windows" && !strings.HasPrefix(term, "!") && !strings.Contains(tags, ","+term+",") {
			t.Errorf("the producer %q does not build its test's tag %q (//go:build %s)", protocolProducer, term, constraint)
		}
	}
}
