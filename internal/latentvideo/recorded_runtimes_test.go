package latentvideo

import (
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// recordedRuntimes points the test's data root at a directory of its own whose record of the clip runtimes is
// runtimes, written as it stands, so a runtime the machine does not hold can be named. Its caller runs serially.
func recordedRuntimes(t *testing.T, runtimes media.ClipRuntimes) {
	t.Helper()
	t.Setenv("OVERGO_DATA_ROOT", t.TempDir())
	_, record, err := media.ReadClipRuntimes()
	testcheck.NoError(t, err)
	testcheck.NoError(t, jsonfile.Write(record, runtimes, 0o600))
}
