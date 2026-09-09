package gate

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"overgo/internal/runrecord"
)

// TestPhaseWallTableOrdersByCost pins: the table lists every recorded step
// by wall, longest first, ties by name, unmeasured steps last; reused,
// inapplicable and skipped steps carry their outcome mark and executed
// successes carry none; the summary prints the table between the GATE
// line and the audit lines.
func TestPhaseWallTableOrdersByCost(t *testing.T) {
	steps := []runrecord.GateStep{
		{Name: "build", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: uint64(30 * time.Second)},
		{Name: "test", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: uint64(90 * time.Second)},
		{Name: "claims", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepSkipped},
		{Name: "device", Phase: runrecord.PhaseTest, Outcome: runrecord.StepReused, DurationNS: uint64(5 * time.Second)},
		{Name: "magics", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepInapplicable, DurationNS: uint64(12 * time.Second)},
		{Name: "docs", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepSucceeded, DurationNS: uint64(12 * time.Second)},
	}
	rows := PhaseWallTable(steps)
	var order []string
	for _, row := range rows {
		order = append(order, row.Name)
	}
	if got := strings.Join(order, ","); got != "test,build,docs,magics,device,claims" {
		t.Fatalf("phase wall order = %s", got)
	}
	lines := FormatPhaseWallTable(rows, 149*time.Second)
	if len(lines) != len(steps)+1 || !strings.HasPrefix(lines[0], "phase wall: total=2m29s steps=6") {
		t.Fatalf("phase wall lines = %q", lines)
	}
	for index, want := range []string{"1m30s", "30s", "12s", "12s", "5s", "0s"} {
		if !strings.HasPrefix(strings.TrimSpace(lines[index+1]), want) {
			t.Fatalf("row %d = %q, want wall %s", index, lines[index+1], want)
		}
	}
	for _, want := range []string{"magics       validate inapplicable", "device       test reused", "claims       validate skipped"} {
		if !strings.Contains(strings.Join(lines, "\n"), want) {
			t.Fatalf("phase wall lacks %q:\n%s", want, strings.Join(lines, "\n"))
		}
	}
	if strings.Contains(lines[1], "succeeded") {
		t.Fatalf("executed success carries a mark: %q", lines[1])
	}

	gate := &gateContext{start: time.Now(), steps: steps, audit: []string{"code profile delta vs HEAD: production=+0 files/-20 nodes duplicate_excess=-12"}}
	var output bytes.Buffer
	gate.printSummary(&output, runrecord.OutcomeSucceeded, "")
	text := output.String()
	gateAt, tableAt, auditAt := strings.Index(text, "GATE SUCCEEDED"), strings.Index(text, "phase wall: total="), strings.Index(text, "advisory: delta:")
	if gateAt < 0 || tableAt < gateAt || auditAt < tableAt {
		t.Fatalf("summary order gate=%d table=%d audit=%d:\n%s", gateAt, tableAt, auditAt, text)
	}
}
