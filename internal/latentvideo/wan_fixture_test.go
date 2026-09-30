package latentvideo

import (
	"testing"

	"overgo/internal/modeltest"
)

// wanModelDir resolves the Wan 2.1 checkpoint through the store's
// registrations; a checkout whose store has not registered it cannot apply.
func wanModelDir(t testing.TB) string {
	t.Helper()
	return modeltest.Directory(t, "Wan2.1-T2V-1.3B")
}
