package landing

import (
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"

	"overgo/internal/cuda/driver"
)

// TestACachedCUDAPassIsKeyedOnItsDriver holds every lane's command to the environment that names the CUDA driver the
// host runs, which cudatest.Require reads, so go test keys a cached CUDA pass on the driver it ran on. A fixture module
// pins go test's keying: a test that reads an environment value replays its pass under the same value and runs again
// under another, as a CUDA test does under a driver update.
func TestACachedCUDAPassIsKeyedOnItsDriver(t *testing.T) {
	t.Parallel()
	identity := driver.IdentityEnvironment + "=" + driver.LibraryIdentity()
	for _, candidate := range lanes {
		if !slices.Contains(laneEnvironment(candidate), identity) {
			t.Errorf("the %s lane runs without %s", candidate.name, identity)
		}
	}
	repo := t.TempDir()
	writeModule(t, repo, map[string]string{
		"go.mod":      "module fixture\n\ngo 1.26\n",
		"p/p.go":      "package p\n",
		"p/p_test.go": "package p\n\nimport (\n\t\"os\"\n\t\"testing\"\n)\n\nfunc TestReadsIdentity(t *testing.T) { _ = os.Getenv(\"FIXTURE_DRIVER\") }\n",
	})
	run := func(value string) string {
		t.Helper()
		command := exec.CommandContext(t.Context(), "go", "test", "./p")
		command.Dir, command.Env = repo, append(os.Environ(), "FIXTURE_DRIVER="+value, "GOFLAGS=")
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("go test under %s: %v\n%s", value, err, output)
		}
		return string(output)
	}
	run("first")
	if output := run("first"); !strings.Contains(output, cachedMarker) {
		t.Errorf("under the same value the pass ran again: %s", output)
	}
	if output := run("second"); strings.Contains(output, cachedMarker) {
		t.Errorf("under another value the pass replayed: %s", output)
	}
}
