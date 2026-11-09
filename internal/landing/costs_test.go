package landing

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"overgo/internal/testcheck"
)

// TestALandingRecordsItsFullCost holds a landing to saying what it cost beside what it measured, as the owner's review
// (2026-11-09) asked: the benefit must exceed the full cost of proposing, checking and recovering. The driver recorded
// three attempts at row/do in the process state, the first by a local worker that ran 300 s and failed, the second by
// the Claude worker still running, the one that lands, and one at another step. The landing's message says 2 worker
// attempts by both workers, 1 before the one that landed, and its trailers carry the attempts, the workers' wall (the
// running attempt counted to now) and the gate's, and each worker. A landing the driver recorded no attempt of says
// it was an interactive session's and carries 0 attempts. The record is written as the driver writes it, its fields
// spelled out, so the test builds with the landing before it.
func TestALandingRecordsItsFullCost(t *testing.T) {
	t.Parallel()
	landed := func(attempts []map[string]any) string {
		t.Helper()
		fixture := newRowFixture(t, map[string]string{"p/p.go": undoneValue})
		l := fixture.candidate()
		l.spent.Started = time.Now().Add(-90 * time.Second)
		if attempts != nil {
			state := filepath.Join(fixture.repo, ".overgo-runtime")
			testcheck.NoError(t, os.MkdirAll(state, 0o700))
			testcheck.NoError(t, os.WriteFile(filepath.Join(state, "worker_attempts.json"), testcheck.Must(json.Marshal(attempts))(t), 0o600))
		}
		l.composed = "subject\n\nChange:\n- a change\n"
		return l.message()
	}
	first, running := time.Now().Add(-10*time.Minute), time.Now().Add(-2*time.Minute)
	message := landed([]map[string]any{
		{"step": "row/do", "worker": "local qwen", "started": first},
		{"step": "row/do", "worker": "local qwen", "started": first, "wall": 300 * time.Second},
		{"step": "row/do", "worker": "claude-master-lead", "started": running},
		{"step": "other/step", "worker": "elsewhere", "started": first},
	})
	trailer := func(message, name string) int {
		_, value, found := strings.Cut(message, "\n"+name+": ")
		value, _, _ = strings.Cut(value, "\n")
		number, err := strconv.Atoi(value)
		if !found || err != nil {
			return -1
		}
		return number
	}
	if !strings.Contains(message, "Cost: 2 worker attempts by local qwen, claude-master-lead, 1 before the one that landed") ||
		trailer(message, "Worker-Attempts") != 2 || trailer(message, "Worker-Wall-Seconds") < 420 || trailer(message, "Gate-Wall-Seconds") < 90 ||
		!strings.Contains(message, "\nWorker: local qwen\n") || !strings.Contains(message, "\nWorker: claude-master-lead\n") || strings.Contains(message, "elsewhere") {
		t.Errorf("a landing the driver recorded attempts at says:\n%s", message)
	}
	if interactive := landed(nil); !strings.Contains(interactive, "Cost: no worker attempt recorded, the work an interactive session's") || trailer(interactive, "Worker-Attempts") != 0 {
		t.Errorf("a landing the driver recorded no attempt at says:\n%s", interactive)
	}
}
