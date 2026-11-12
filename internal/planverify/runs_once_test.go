package planverify

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestACheckRunsOnce holds a step's check to one run (cheap-first/form): a second run of every host check, held to
// agree with the first, doubled each check's time, and no refusal recorded nor any commit names a run that disagreed.
// The fixture's test writes a line to a file outside its package each time it runs.
func TestACheckRunsOnce(t *testing.T) {
	t.Parallel()
	directory, runs := t.TempDir(), filepath.Join(t.TempDir(), "runs")
	for name, content := range map[string]string{
		"go.mod": "module example.com/probe\n\ngo 1.26\n",
		"probe_test.go": "package probe\n\nimport (\n\t\"os\"\n\t\"testing\"\n)\n\nfunc TestCounted(t *testing.T) {\n" +
			"\tfile, err := os.OpenFile(os.Getenv(\"PROBE_RUNS\"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)\n" +
			"\tif err != nil {\n\t\tt.Fatal(err)\n\t}\n\tdefer file.Close()\n" +
			"\tif _, err := file.WriteString(\"ran\\n\"); err != nil {\n\t\tt.Fatal(err)\n\t}\n}\n",
	} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600))
	}
	if _, err := Execute(t.Context(), directory, "go test ./... -count=1 -run '^TestCounted$'", []string{"PROBE_RUNS=" + runs}); err != nil {
		t.Fatal(err)
	}
	if ran := strings.Count(string(testcheck.Must(os.ReadFile(runs))(t)), "ran\n"); ran != 1 {
		t.Fatalf("the check ran its test %d times; want once", ran)
	}
}
