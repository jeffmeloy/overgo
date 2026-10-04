package main

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestReplayReportSeparatesQualityFromMeasuredScope(t *testing.T) {
	t.Parallel()
	record := suiteRow{metrics: map[string]float64{"accuracy": 0.8}, record: "current-evaluation", commit: "current-commit",
		cases: 100, measuredCases: 32, wall: 320 * time.Millisecond,
		carried:       testutil.ArtifactID(t, artifact.KindEvaluation, "prior-evaluation"),
		qualityReport: testutil.ArtifactID(t, artifact.KindEvaluation, "prior-full-report"),
		replayReport:  testutil.ArtifactID(t, artifact.KindEvaluation, "fresh-sample-report")}
	if got := perCaseCell(record); got != "10ms" {
		t.Fatalf("replay per case = %s, want320ms/32 =10ms", got)
	}
	text := string(render([]modelRow{{name: "fixture", suites: map[string]suiteRow{"suite": record}}}, map[string]evaluation.SuiteDescriptor{}))
	for _, fact := range []string{"carried", "100", "32", "10ms", record.qualityReport.String(), record.replayReport.String()} {
		if !strings.Contains(text, fact) {
			t.Errorf("report omits %q: %s", fact, text)
		}
	}
	if got := perCaseCell(suiteRow{cases: 100, wall: time.Second}); got != "—" {
		t.Fatalf("unknown measured denominator = %s", got)
	}
	if got := perCaseCell(suiteRow{cases: 100, measuredCases: 100, wall: time.Second}); got != "10ms" {
		t.Fatalf("full per case = %s", got)
	}
	if got := perCaseCell(suiteRow{cases: 100, measuredCases: 132, wall: 1320 * time.Millisecond}); got != "10ms" {
		t.Fatalf("replay plus full fallback per case = %s", got)
	}
	fallback := record
	fallback.carried, fallback.fallback, fallback.measuredCases = artifact.ID{}, true, 132
	text = string(render([]modelRow{{name: "fixture", suites: map[string]suiteRow{"suite": fallback}}}, nil))
	if !strings.Contains(text, "fresh full report") || !strings.Contains(text, "plus full fallback") || strings.Contains(text, "carried quality") {
		t.Fatalf("fallback report loses fresh scope: %s", text)
	}
	fallback.measuredCases, fallback.replayReport = 0, artifact.ID{}
	text = string(render([]modelRow{{name: "fixture", suites: map[string]suiteRow{"suite": fallback}}}, nil))
	if !strings.Contains(text, "replay case count unknown") || perCaseCell(fallback) != "—" {
		t.Fatalf("partial replay invents a denominator: %s", text)
	}
}

func TestReplayScopeRequiresItsBoundReceipt(t *testing.T) {
	t.Parallel()
	plan := testutil.ArtifactID(t, artifact.KindProfile, "full-plan")
	id := testutil.ArtifactID(t, artifact.KindEvaluation, "replay-receipt")
	base := evaluation.ReplayReceipt{ID: id, Plan: plan, Cases: []int{0, 99},
		Carried: testutil.ArtifactID(t, artifact.KindEvaluation, "previous-evaluation"),
		Report:  testutil.ArtifactID(t, artifact.KindEvaluation, "full-quality"),
		Replay:  testutil.ArtifactID(t, artifact.KindEvaluation, "sample")}
	for _, scenario := range []struct {
		name    string
		receipt evaluation.ReplayReceipt
		count   uint64
	}{
		{"carried", base, 2},
		{"fallback", evaluation.ReplayReceipt{ID: id, Plan: plan, Report: base.Report, Replay: base.Replay, Cases: base.Cases, Fallback: true}, 102},
		{"partial", evaluation.ReplayReceipt{ID: id, Plan: plan, Report: base.Report, Fallback: true}, 0},
		{"foreign plan", evaluation.ReplayReceipt{ID: id, Plan: testutil.ArtifactID(t, artifact.KindProfile, "foreign"), Report: base.Report, Fallback: true}, 0},
		{"foreign output", evaluation.ReplayReceipt{ID: base.Report, Plan: plan, Report: base.Report, Replay: base.Replay, Cases: base.Cases, Fallback: true}, 0},
		{"outside full scope", evaluation.ReplayReceipt{ID: id, Plan: plan, Report: base.Report, Replay: base.Replay, Cases: []int{100}, Fallback: true}, 0},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Parallel()
			record := suiteRow{cases: 100, wall: time.Second}
			record.replayScope(runrecord.Run{Inputs: []artifact.ID{plan}, Outputs: []artifact.ID{id}}, scenario.receipt)
			if record.measuredCases != scenario.count {
				t.Fatalf("measured scope=%d, want%d", record.measuredCases, scenario.count)
			}
		})
	}
}
