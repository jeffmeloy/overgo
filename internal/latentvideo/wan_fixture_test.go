package latentvideo

import (
	"os"
	"testing"

	"overgo/internal/testskip"
)

func wanModelDir(t testing.TB) string {
	t.Helper()
	dir := os.Getenv("OVERGO_WAN_MODEL")
	if dir == "" {
		testskip.NotApplicable(t, "set OVERGO_WAN_MODEL to run real Wan integration tests")
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("Wan model directory %s: %v", dir, err)
	}
	return dir
}
