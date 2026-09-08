package testutil

import "testing"

// UnexpectedLog fails the owning test when a server or transport emits a log.
// It is an io.Writer for loggers whose diagnostic output must not be ignored.
type UnexpectedLog struct{ Test testing.TB }

// Write reports unexpected diagnostics without suppressing test failures.
func (writer UnexpectedLog) Write(data []byte) (int, error) {
	writer.Test.Helper()
	writer.Test.Errorf("unexpected transport/server log: %s", data)
	return len(data), nil
}
