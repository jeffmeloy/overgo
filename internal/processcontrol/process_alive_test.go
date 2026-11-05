package processcontrol

import (
	"os"
	"os/exec"
	"testing"

	"overgo/internal/testcheck"
)

// TestProcessAlive holds the liveness probe: the test's own process is
// alive, an exited child is not, and a non-positive pid never is.
func TestProcessAlive(t *testing.T) {
	t.Parallel()
	if !ProcessAlive(os.Getpid()) {
		t.Fatal("the running test process was reported dead")
	}
	if ProcessAlive(0) || ProcessAlive(-1) {
		t.Fatal("a non-positive pid was reported alive")
	}
	executable, err := os.Executable()
	testcheck.NoError(t, err)
	child := exec.Command(executable, "-test.run=^$")
	testcheck.NoError(t, child.Run())
	if ProcessAlive(child.Process.Pid) {
		t.Fatal("an exited child was reported alive")
	}
}
