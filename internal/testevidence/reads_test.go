package testevidence

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestAPackageRunNamesWhatItRead holds observed-inputs/measure: a package's test run names what its test binary read,
// from the binary's own test log, each file by its content's digest, each path it looked at and each variable it
// read. The fixture's test opens one file, looks at another that is not there and reads one variable; go test keeps a
// pass by the same log, which nothing in overgo had read.
func TestAPackageRunNamesWhatItRead(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	content := "what the probe reads\n"
	for name, text := range map[string]string{
		"go.mod":              "module example.com/probe\n\ngo 1.26\n",
		"probe/data.txt":      content,
		"probe/probe_test.go": "package probe\n\nimport (\n\t\"os\"\n\t\"testing\"\n)\n\nfunc TestReads(t *testing.T) {\n\tif _, err := os.ReadFile(\"data.txt\"); err != nil {\n\t\tt.Fatal(err)\n\t}\n\t_, _ = os.Stat(\"absent.txt\")\n\t_ = os.Getenv(\"OVERGO_PROBE_SETTING\")\n}\n",
	} {
		path := filepath.Join(directory, filepath.FromSlash(name))
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(text), 0o600))
	}
	reads := testcheck.Must(ReadsOf(t.Context(), directory, "./probe"))(t)
	sum := sha256.Sum256([]byte(content))
	read := filepath.Join(directory, "probe", "data.txt")
	if reads.Opened[read] != hex.EncodeToString(sum[:]) {
		t.Errorf("the run named its opened files %v; want %s by its content's digest", reads.Opened, read)
	}
	if present, looked := reads.Looked[filepath.Join(directory, "probe", "absent.txt")]; !looked || present {
		t.Errorf("the run named the paths it looked at %v; want absent.txt looked at and not there", reads.Looked)
	}
	if _, read := reads.Environment["OVERGO_PROBE_SETTING"]; !read {
		t.Errorf("the run named the variables it read %v; want OVERGO_PROBE_SETTING", reads.Environment)
	}
}
