package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/landing"
	"overgo/internal/testcheck"
)

// TestTheStatusSaysTheDaysFiveMeasures holds the status to the five measures the owner asked to see at session start
// (review-fixes/status-measures, SQA-376), read from Git and the gate's run records alone: the day's commits, plan-only
// ones apart; its refused commit attempts by the stage that refused them, preflights and runs of unknown kind apart; the
// gate's hours over its runs, waits named; the steps landed, added and retired; and the landings of the report's rows.
// A run or commit before the day counts for nothing.
func TestTheStatusSaysTheDaysFiveMeasures(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	now := time.Now()
	day, before := now.Add(-time.Hour), now.Add(-48*time.Hour)
	git := func(at time.Time, arguments ...string) string {
		t.Helper()
		command := exec.Command("git", append([]string{"-C", root, "-c", "user.name=test", "-c", "user.email=test@example.com"}, arguments...)...)
		stamp := at.Format(time.RFC3339)
		command.Env = append(os.Environ(), "GIT_AUTHOR_DATE="+stamp, "GIT_COMMITTER_DATE="+stamp)
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", arguments, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	write := func(name, text string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(text), 0o644))
	}
	planOf := func(items string) string { return `{"items": [` + items + `]}` + "\n" }
	base := `{"id": "base", "steps": [{"id": "one"}]}`
	report := `{"id": "report-row", "steps": [{"id": "a"}, {"id": "b"}]}`
	commit := func(at time.Time, message string) {
		git(at, "add", "-A")
		git(at, "commit", "-q", "-m", message)
	}
	git(before, "init", "-q")
	write("docs/plan.json", planOf(base))
	write("code.go", "package code\n")
	commit(before, "start")
	write("docs/plan.json", planOf(base+", "+report))
	commit(before, "report: file its rows")
	filing := git(before, "rev-parse", "HEAD")
	write("code.go", "package code\n\n// A is report-row/a's work.\nconst A = 1\n")
	write("docs/plan.json", planOf(base+`, {"id": "report-row", "steps": [{"id": "b"}]}`))
	commit(day, "report-row/a: its work\n\nLanded-Row: report-row/a")
	write("docs/plan.json", planOf(`{"id": "base", "steps": [{"id": "one"}, {"id": "two"}]}, {"id": "report-row", "steps": [{"id": "b"}]}`))
	commit(day, "base/two: filed")
	write("docs/plan.json", planOf(`{"id": "base", "steps": [{"id": "two"}]}, {"id": "report-row", "steps": [{"id": "b"}]}`))
	commit(day, "plan: retire base/one\n\nRetired-Row: base/one")

	run := func(at time.Time, minutes int, outcome, kind string) map[string]any {
		return map[string]any{"reference": "row/step", "started": at, "ended": at.Add(time.Duration(minutes) * time.Minute),
			"outcome": outcome, "kind": kind, "queue_wait": int64(time.Minute)}
	}
	runs := []map[string]any{
		run(before, 60, "landed", "commit"),
		run(day, 10, "landed", "commit"), run(day, 8, "rules", "commit"), run(day, 6, "verify", "commit"),
		run(day, 2, "preflight", "preflight"), run(day, 1, "rules", "preflight"), run(day, 3, "test", ""),
	}
	records := testcheck.Must(json.Marshal(runs))(t)
	write(".overgo-runtime/landing_waits.json", string(records))

	delivery := testcheck.Must(landing.DayDelivery(t.Context(), root, now, filing))(t)
	if delivery.Commits != 3 || delivery.PlanOnly != 2 {
		t.Errorf("the day's commits read %d, %d of them plan-only; want 3 and 2", delivery.Commits, delivery.PlanOnly)
	}
	if !slices.Equal(delivery.Landed, []string{"report-row/a"}) || !slices.Equal(delivery.Added, []string{"base/two"}) || !slices.Equal(delivery.Retired, []string{"base/one"}) {
		t.Errorf("the day's steps read landed %v, added %v, retired %v; want [report-row/a], [base/two], [base/one]", delivery.Landed, delivery.Added, delivery.Retired)
	}
	if delivery.ReportLandings != 1 {
		t.Errorf("the report's rows were landed %d times; want 1", delivery.ReportLandings)
	}
	if delivery.Refused["rules"] != 1 || delivery.Refused["verify"] != 1 || delivery.UnknownKind["test"] != 1 || delivery.Preflights != 2 || delivery.PreflightsRefused != 1 {
		t.Errorf("the day's runs read refused %v, of unknown kind %v, preflights %d (%d refused); want rules 1 and verify 1, test 1, 2 and 1",
			delivery.Refused, delivery.UnknownKind, delivery.Preflights, delivery.PreflightsRefused)
	}
	if want := 30.0 / 60; delivery.GateHours < want-0.01 || delivery.GateHours > want+0.01 {
		t.Errorf("the gate's hours read %.2f; want %.2f, the day's runs and no earlier one", delivery.GateHours, want)
	}
	line := delivery.String()
	for _, said := range []string{"3 commits", "2 plan-only", "1 landed", "1 added", "1 retired", "report", "rules 1", "verify 1"} {
		if !strings.Contains(line, said) {
			t.Errorf("the status line %q does not say %q", line, said)
		}
	}
}
