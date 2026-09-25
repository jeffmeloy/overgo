package webuilane

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// LegRecordsEnvironment names the directory a lane run collects its legs'
// records in; outside a lane run a leg only logs.
const LegRecordsEnvironment = "OVERGO_WEBUI_LANE_LEGS"

// LegRecord is one browser leg's typed outcome: the test that ran it, the
// leg's name (its identity, which row checks name) and what it proved.
type LegRecord struct {
	Test   string `json:"test"`
	Leg    string `json:"leg"`
	Detail string `json:"detail"`
}

// legTest is the part of a test a leg reports through.
type legTest interface {
	Helper()
	Name() string
	Log(...any)
	Fatal(...any)
}

// Leg records that the running test completed the named leg, logging it
// and, in a lane run, writing its record where the lane reads it.
func Leg(t legTest, name, format string, args ...any) {
	t.Helper()
	detail := fmt.Sprintf(format, args...)
	t.Log(name + ": " + detail)
	directory := os.Getenv(LegRecordsEnvironment)
	if directory == "" {
		return
	}
	encoded, err := json.Marshal(LegRecord{Test: t.Name(), Leg: name, Detail: detail})
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

// TestEvent is the part of a go test -json event the lane judges by.
type TestEvent struct {
	Action string
	Test   string
	Output string
}

// TestRun is a go test -json run as the lane reads it: each test's
// outcome, the tests started and not ended, and each test's output.
type TestRun struct {
	Outcomes map[string]string
	Running  []string
	Output   map[string][]string
}

// ReadTestRun follows a go test -json stream, passing each output line on
// to out as it arrives.
func ReadTestRun(stream io.Reader, out io.Writer) (TestRun, error) {
	run := TestRun{Outcomes: map[string]string{}, Output: map[string][]string{}}
	decoder := json.NewDecoder(stream)
	for {
		var event TestEvent
		if err := decoder.Decode(&event); errors.Is(err, io.EOF) {
			return run, nil
		} else if err != nil {
			return run, fmt.Errorf("webui lane: test event stream: %w", err)
		}
		switch event.Action {
		case "output":
			if _, err := io.WriteString(out, event.Output); err != nil {
				return run, err
			}
			if event.Test != "" {
				run.Output[event.Test] = append(run.Output[event.Test], strings.TrimSpace(event.Output))
			}
		case "run":
			run.Running = append(run.Running, event.Test)
		case "pass", "fail", "skip":
			if event.Test == "" {
				continue
			}
			run.Outcomes[event.Test] = event.Action
			run.Running = slices.DeleteFunc(run.Running, func(test string) bool { return test == event.Test })
		}
	}
}

// Failures names the failed and the unfinished top-level tests, each failed
// test with the last source line it wrote before its verdict and that
// line's continuation: the failing step and its page state, which a
// caller's bounded tail of the whole run would lose.
func (run TestRun) Failures() string {
	var failed, unfinished []string
	for test, outcome := range run.Outcomes {
		if outcome != "fail" || strings.Contains(test, "/") {
			continue
		}
		var last string
		for _, line := range run.Output[test] {
			switch {
			case strings.Contains(line, "_test.go:"):
				last = line
			case last != "" && line != "" && !strings.HasPrefix(line, "---") && len(last) < failureLineBytes:
				last += " " + line
			}
		}
		if len(last) > failureLineBytes {
			last = strings.ToValidUTF8(last[:failureLineBytes], "") + "…"
		}
		failed = append(failed, test+": "+last)
	}
	for _, test := range run.Running {
		if !strings.Contains(test, "/") {
			unfinished = append(unfinished, test)
		}
	}
	slices.Sort(failed)
	return "failed=[" + strings.Join(failed, "; ") + "] unfinished=[" + strings.Join(unfinished, ",") + "]"
}

// failureLineBytes bounds one reported failure so several fit a caller's
// diagnostic tail beside one another.
const failureLineBytes = 800

// LaneVerdict judges a lane run by its typed outcomes: a named test that
// skipped proves nothing, a run with no passing test proves nothing, and
// each required leg needs a record from a test that passed.
func LaneVerdict(run TestRun, records []LegRecord, required []string) error {
	passed := 0
	for test, outcome := range run.Outcomes {
		switch {
		case outcome == "skip" && !strings.Contains(test, "/"):
			return fmt.Errorf("webui lane: %s was skipped, so it proves nothing", test)
		case outcome == "pass":
			passed++
		}
	}
	if passed == 0 {
		return errors.New("webui lane: no browser test passed")
	}
	for _, leg := range required {
		if !slices.ContainsFunc(records, func(record LegRecord) bool {
			return record.Leg == leg && run.Outcomes[record.Test] == "pass"
		}) {
			return fmt.Errorf("webui lane: no passing test recorded the required leg %q", leg)
		}
	}
	return nil
}
