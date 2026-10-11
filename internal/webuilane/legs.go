package webuilane

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"

	"overgo/internal/testevidence"
)

// LegRecordsEnvironment names the directory a lane run collects its legs'
// records in; outside a lane run a leg only logs.
const LegRecordsEnvironment = "OVERGO_WEBUI_LANE_LEGS"

// WaitBudgetEnvironment names how long a browser a leg opens may wait for
// one answer from its page, as time.ParseDuration reads it. The lane declares
// it for the page acceptances; with none declared a wait is bounded by its
// caller alone.
const WaitBudgetEnvironment = "OVERGO_WEBUI_LANE_WAIT"

// LegRecord is one browser leg's typed outcome: the package and test that
// ran it, the leg's name (its identity, which row checks name) and what it
// proved.
type LegRecord struct {
	Package string `json:"package"`
	Test    string `json:"test"`
	Leg     string `json:"leg"`
	Detail  string `json:"detail"`
}

// legTest is the part of a test a leg reports through.
type legTest interface {
	Helper()
	Name() string
	Log(...any)
	Fatal(...any)
}

// Leg records that the running test completed the named leg, logging it
// and, in a lane run, writing its record where the lane reads it. The
// record names the test binary's package, as go test's events do.
func Leg(t legTest, name, format string, args ...any) {
	t.Helper()
	detail := fmt.Sprintf(format, args...)
	t.Log(name + ": " + detail)
	directory := os.Getenv(LegRecordsEnvironment)
	if directory == "" {
		return
	}
	build, ok := debug.ReadBuildInfo()
	if !ok {
		t.Fatal("webui lane: the test binary carries no build information naming its package")
	}
	encoded, err := json.Marshal(LegRecord{Package: strings.TrimSuffix(build.Path, ".test"), Test: t.Name(), Leg: name, Detail: detail})
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.CreateTemp(directory, "leg-*.json")
	if err != nil {
		t.Fatal(err)
	}
	_, writeErr := file.Write(encoded)
	if err := errors.Join(writeErr, file.Close()); err != nil {
		t.Fatal(err)
	}
}

// ReadLegRecords reads every record a lane run's legs wrote.
func ReadLegRecords(directory string) ([]LegRecord, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	records := make([]LegRecord, 0, len(entries))
	for _, entry := range entries {
		encoded, err := os.ReadFile(filepath.Join(directory, entry.Name()))
		if err != nil {
			return nil, err
		}
		var record LegRecord
		if err := json.Unmarshal(encoded, &record); err != nil {
			return nil, fmt.Errorf("webui lane: leg record %s: %w", entry.Name(), err)
		}
		records = append(records, record)
	}
	return records, nil
}

// LaneVerdict judges a lane run by its typed test report: a failed or
// unfinished test refuses it with each one's retained output, a named test
// that skipped proves nothing, a run with no passing test proves nothing,
// and each required leg needs a record from a test that passed: its own
// package's evidence complete and the test not skipped, so a same-named
// test of another package never lends it an outcome.
func LaneVerdict(report testevidence.GoTestReport, records []LegRecord, required []string) error {
	if len(report.Failed) != 0 || len(report.Unfinished) != 0 {
		return fmt.Errorf("webui lane: %s\nfailed=%v unfinished=%v", strings.Join(report.Diagnostics, "\n"), report.Failed, report.Unfinished)
	}
	skipped := slices.Concat(report.Skipped, report.ClassifiedSkipped)
	for _, test := range skipped {
		if _, name, _ := strings.Cut(test, ": "); !strings.Contains(name, "/") {
			return fmt.Errorf("webui lane: %s was skipped, so it proves nothing", test)
		}
	}
	if report.PassedTests == 0 {
		return errors.New("webui lane: no browser test passed")
	}
	for _, leg := range required {
		if !slices.ContainsFunc(records, func(record LegRecord) bool {
			return record.Leg == leg && report.PackagePassed(record.Package) && !slices.Contains(skipped, record.Package+": "+record.Test)
		}) {
			return fmt.Errorf("webui lane: no passing test recorded the required leg %q", leg)
		}
	}
	return nil
}
