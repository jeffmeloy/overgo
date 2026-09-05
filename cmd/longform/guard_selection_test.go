package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestModelRegressionSelection(t *testing.T) {
	entry := func(name string, cost int64) guardCoverageEntry {
		return guardCoverageEntry{Model: testutil.ArtifactID(t, artifact.KindModel, name),
			Baseline: testutil.ArtifactID(t, artifact.KindEvidence, name).String(), Location: name + ".gguf", MeasuredWallNS: cost}
	}
	cheap, costly := entry("cheap", 20), entry("costly", 90)
	coverage := guardCoverageReport{Surface: "candidate surface", Selected: 2, Explicit: 2, Covered: 2, Entries: []guardCoverageEntry{costly, cheap}}
	selection, err := selectGuardChecks(coverage, "runtime changed", nil)
	if err != nil || selection.Status != "ready" || len(selection.Requests) != 2 ||
		selection.Requests[0].Model != cheap.Model || selection.Requests[0].Baseline != cheap.Baseline || selection.Requests[1].Baseline != costly.Baseline {
		t.Fatalf("selection changed coverage or baseline: %+v %v", selection, err)
	}
	if coverage.Entries[0].Model != costly.Model {
		t.Fatal("selection mutated the coverage authority")
	}
	t.Run("stable ties do not depend on enumeration", func(t *testing.T) {
		first := coverage
		first.Entries = slices.Clone(coverage.Entries)
		first.Entries[0].MeasuredWallNS = cheap.MeasuredWallNS
		second := first
		second.Entries = slices.Clone(first.Entries)
		slices.Reverse(second.Entries)
		a, err := selectGuardChecks(first, "runtime changed", nil)
		if err != nil {
			t.Fatal(err)
		}
		b, err := selectGuardChecks(second, "runtime changed", nil)
		if err != nil || !slices.Equal(a.Requests, b.Requests) {
			t.Fatalf("unstable tie: %+v %+v %v", a, b, err)
		}
	})
	for name, mutate := range map[string]func(*guardCoverageReport){
		"missing class":           func(r *guardCoverageReport) { r.Covered--; r.Uncovered++ },
		"missing row":             func(r *guardCoverageReport) { r.Entries = r.Entries[:1] },
		"missing source":          func(r *guardCoverageReport) { r.Surface = "" },
		"empty":                   func(r *guardCoverageReport) { r.Selected = 0 },
		"missing explicit record": func(r *guardCoverageReport) { r.Explicit-- },
		"unused record":           func(r *guardCoverageReport) { r.Unused = []string{"unused"} },
		"unmeasured cost":         func(r *guardCoverageReport) { r.Entries[1].MeasuredWallNS = 0 },
		"device refusal":          func(r *guardCoverageReport) { r.Entries[1].Gap = "device evidence absent" },
		"invalid record":          func(r *guardCoverageReport) { r.Entries[1].Baseline = "missing" },
		"duplicate model":         func(r *guardCoverageReport) { r.Entries[1].Model = r.Entries[0].Model },
	} {
		t.Run(name, func(t *testing.T) {
			changed := coverage
			changed.Entries = slices.Clone(coverage.Entries)
			mutate(&changed)
			selection, err := selectGuardChecks(changed, "runtime changed", nil)
			if err == nil || selection.Status != "incomplete" || len(selection.Requests) != 0 {
				t.Fatalf("partial or invalid coverage selected: %+v %v", selection, err)
			}
		})
	}
	if selection, err := selectGuardChecks(coverage, "runtime changed", errors.New("stale recipe")); err == nil || len(selection.Requests) != 0 {
		t.Fatalf("upstream refusal lost: %+v %v", selection, err)
	}
	t.Run("paths preserve spaces and missing scope remains unknown", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "paths.txt")
		if err := os.WriteFile(path, []byte("docs/a file.md\r\ninternal/model/spec.go\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		paths, err := readGuardPaths(path)
		if err != nil || !slices.Equal(paths, []string{"docs/a file.md", "internal/model/spec.go"}) {
			t.Fatalf("paths=%v error=%v", paths, err)
		}
		if _, err := readGuardPaths(path + ".absent"); err == nil {
			t.Fatal("missing path list accepted")
		}
	})
	t.Run("CLI requires complete catalog scope", func(t *testing.T) {
		args := []string{"-guard-select", "-all", "-paths-file", "paths.txt", "-budget", "1m", "-corpus", "fixed", "-baseline", cheap.Baseline}
		if options, err := parseOptions(args); err != nil || !options.GuardSelect {
			t.Fatalf("selection options=%+v error=%v", options, err)
		}
		for _, flag := range []string{"-check", "-publish", "-validate-baselines", "-guard-coverage", "-guard"} {
			if _, err := parseOptions(append([]string{flag}, args...)); err == nil {
				t.Fatalf("mixed mode %s accepted", flag)
			}
		}
		if _, err := parseOptions([]string{"-guard-select", "-paths-file", "paths.txt", "-budget", "1m", "-corpus", "fixed", "-baseline", cheap.Baseline, "one.gguf"}); err == nil {
			t.Fatal("partial caller-selected catalog accepted")
		}
	})
	t.Run("host-only CLI avoids opening a missing store", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "paths.txt")
		if err := os.WriteFile(path, []byte("docs/plan.json\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		var output bytes.Buffer
		err := run([]string{"-root", "../..", "-repo", filepath.Join(t.TempDir(), "missing-store"), "-guard-select", "-all", "-paths-file", path, "-budget", "1m", "-corpus", "testdata/guard-corpus.txt", "-baseline", cheap.Baseline}, &output)
		if err != nil || !bytes.Contains(output.Bytes(), []byte(`"status": "inapplicable"`)) {
			t.Fatalf("host-only command read catalog or failed: %s %v", output.String(), err)
		}
	})
}
