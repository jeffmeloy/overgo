package latentvideo

import (
	"os"
	"testing"
)

func wanModelDir(t testing.TB) string {
	t.Helper()
	dir := os.Getenv("OVERGO_WAN_MODEL")
	if dir == "" {
		t.Skip("set OVERGO_WAN_MODEL to run real Wan integration tests")
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("Wan model directory %s: %v", dir, err)
	}
	return dir
}
