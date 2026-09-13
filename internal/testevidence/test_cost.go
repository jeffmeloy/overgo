package testevidence

import (
	"cmp"
	"slices"
	"strings"
	"time"
)

// TestExecution is an observation, never pass or reuse authority.
// Elapsed excludes testing.T pauses. Parent and child elapsed may overlap;
// parallel children need not appear in their parent's elapsed.
type TestExecution struct {
	Package string   `json:"package"`
	Name    string   `json:"name"`
	Action  string   `json:"action"`
	Start   string   `json:"start"`
	End     string   `json:"end"`
	Elapsed *float64 `json:"elapsed_seconds,omitempty"`
	Unknown string   `json:"unknown,omitzero"`
}

// TestCosts ranks terminal measurements, retaining unmeasured observations last.
// Missing timestamps, skips, interrupted and duplicate lifecycles earn no cost.
func (report GoTestReport) TestCosts() []TestExecution {
	result := make([]TestExecution, 0, len(report.tests))
	for _, test := range report.tests {
		entry := TestExecution{Package: test.Package, Name: test.Name, Action: test.Action, Start: test.costStart, End: test.costEnd}
		start, startErr := time.Parse(time.RFC3339Nano, test.costStart)
		end, endErr := time.Parse(time.RFC3339Nano, test.costEnd)
		switch {
		case !test.started || test.Action != "pass" && test.Action != "fail":
			entry.Unknown = "no completed test execution"
		case test.costInvalid:
			entry.Unknown = "ambiguous repeated lifecycle"
		case startErr != nil || endErr != nil || end.Before(start):
			entry.Unknown = "missing or inconsistent event timestamps"
		case test.elapsed == nil:
			entry.Unknown = "missing terminal elapsed measurement"
		default:
			entry.Elapsed = new(*test.elapsed)
		}
		result = append(result, entry)
	}
	slices.SortFunc(result, func(left, right TestExecution) int {
		var elapsed int
		if left.Elapsed != nil && right.Elapsed != nil {
			elapsed = cmp.Compare(*right.Elapsed, *left.Elapsed)
		}
		return cmp.Or(strings.Compare(left.Unknown, right.Unknown), elapsed, strings.Compare(left.Package, right.Package), strings.Compare(left.Name, right.Name))
	})
	return result
}
