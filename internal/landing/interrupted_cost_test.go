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

// TestAnInterruptedAttemptIsNotChargedItsDowntime: an attempt the driver's next start found interrupted is counted and
// charged no wall, since how long it ran before the machine went down is unknown, and the landing's last named worker is
// the one whose attempt lands (SQA-356). The Claude worker's attempt 17 hours ago was interrupted (its start, then the
// interruption the next start recorded), a local worker's ran 60 s three hours ago, and the Claude worker's running
// now lands: three attempts, one interrupted, about three minutes of wall rather than seventeen hours, and the Claude
// worker named last. The record is spelled out as the driver writes it, so the test builds with the landing before it.
func TestAnInterruptedAttemptIsNotChargedItsDowntime(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{"p/p.go": undoneValue})
	l := fixture.candidate()
	l.spent.Started = time.Now().Add(-30 * time.Second)
	interrupted, local, running := time.Now().Add(-17*time.Hour), time.Now().Add(-3*time.Hour), time.Now().Add(-2*time.Minute)
	attempts := []map[string]any{
		{"step": "row/do", "worker": "claude-master-lead", "started": interrupted},
		{"step": "row/do", "worker": "local qwen", "started": local},
		{"step": "row/do", "worker": "local qwen", "started": local, "wall": 60 * time.Second},
		{"step": "row/do", "worker": "", "started": interrupted, "interrupted": true},
		{"step": "row/do", "worker": "claude-master-lead", "started": running},
	}
	state := filepath.Join(fixture.repo, ".overgo-runtime")
	testcheck.NoError(t, os.MkdirAll(state, 0o700))
	testcheck.NoError(t, os.WriteFile(filepath.Join(state, "worker_attempts.json"), testcheck.Must(json.Marshal(attempts))(t), 0o600))
	l.composed = "subject\n\nChange:\n- a change\n"
	message := l.message()
	_, wall, _ := strings.Cut(message, "\nWorker-Wall-Seconds: ")
	wall, _, _ = strings.Cut(wall, "\n")
	seconds, err := strconv.Atoi(wall)
	workers := strings.Split(message, "\nWorker: ")
	last, _, _ := strings.Cut(workers[len(workers)-1], "\n")
	if err != nil || seconds > int(time.Hour/time.Second) || seconds < int((3*time.Minute)/time.Second) || last != "claude-master-lead" ||
		!strings.Contains(message, "Cost: 3 worker attempts by local qwen, claude-master-lead, 2 before the one that landed, 1 interrupted") {
		t.Errorf("a landing after an interrupted attempt says:\n%s", message)
	}
}
