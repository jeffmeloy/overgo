package server

import (
	"runtime"
	"testing"
	"time"
)

func waitForServerSessionWaiter(t *testing.T, handler *Handler) {
	t.Helper()
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	for handler.sessions.Snapshot().Waiting == 0 {
		select {
		case <-timer.C:
			t.Fatal("request did not enter session parking")
		default:
			runtime.Gosched()
		}
	}
}
