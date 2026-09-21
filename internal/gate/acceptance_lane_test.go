package gate

import (
	"slices"
	"testing"
)

// TestModelAcceptanceRunsInItsDeferredLane holds the deferred test lane to
// what it is for. On the live graph the speech acceptance suite, which
// declares itself by importing the marker, is parted into the lane that runs
// after the commit, and a package that declares nothing stays in the part
// that blocks it. The declaration does not make a package a device package:
// that group is still only what links the CUDA runtime, which
// TestDeviceGroupLinksOnlyKernelTests holds, so the two memberships stay two
// facts.
func TestModelAcceptanceRunsInItsDeferredLane(t *testing.T) {
	t.Parallel()
	live := liveRepositoryFixture(t)
	g := live.context()
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	const suite, plain = "overgo/internal/audioparity", "overgo/internal/artifact"
	group := []string{plain, suite}
	if declared := graph.acceptancePackages(group); !slices.Equal(declared, []string{suite}) {
		t.Fatalf("declared acceptance suites = %v, want %s alone", declared, suite)
	}
	devices, err := graph.devicePackages(group)
	if err != nil || len(devices) != 0 {
		t.Fatalf("device packages = %v, %v: a declaration must not make a CPU suite a device package", devices, err)
	}
	deferred, host, err := g.splitDevice(group)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(deferred, []string{suite}) || !slices.Equal(host, []string{plain}) {
		t.Fatalf("deferred = %v, blocking = %v; want the suite after the commit and the plain package before it", deferred, host)
	}
}
