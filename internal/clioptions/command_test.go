package clioptions

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

func TestWriteJSON(t *testing.T) {
	if _, err := CombinedOutputIn("", os.Environ(), os.Args[0], "-test.run=^$"); err != nil {
		t.Fatal(err)
	}
	if Verdict(nil) != "ok" || Verdict(os.ErrInvalid) != "FAIL" {
		t.Fatal("command verdict differs from exit state")
	}
	if got := Tail("abcdef", 3); got != "...def" {
		t.Fatalf("tail = %q", got)
	}
	if got := Tail("abc", 3); got != "abc" {
		t.Fatalf("unbounded tail = %q", got)
	}
	var compact bytes.Buffer
	testcheck.NoError(t, WriteJSON(&compact, map[string]int{"value": 3}))
	if compact.String() != "{\"value\":3}\n" {
		t.Fatalf("compact JSON = %q", compact.String())
	}
	var pretty bytes.Buffer
	testcheck.NoError(t, WritePrettyJSON(&pretty, map[string]int{"value": 3}))
	if pretty.String() != "{\n  \"value\": 3\n}\n" {
		t.Fatalf("pretty JSON = %q", pretty.String())
	}
}

func TestOutputGenerated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "generated.txt")
	testcheck.NoError(t, OutputGenerated([]byte("new"), path, false, true, "stale", &bytes.Buffer{}))
	testcheck.NoError(t, OutputGenerated([]byte("new"), path, true, false, "stale", &bytes.Buffer{}))
	if err := OutputGenerated([]byte("old"), path, true, false, "stale", &bytes.Buffer{}); err == nil || err.Error() != "stale" {
		t.Fatalf("stale error = %v", err)
	}
	var output bytes.Buffer
	testcheck.NoError(t, OutputGenerated([]byte("print"), path, false, false, "stale", &output))
	if output.String() != "print" {
		t.Fatalf("output = %q", output.String())
	}
	if data, err := os.ReadFile(path); err != nil || string(data) != "new" {
		t.Fatalf("file = %q, error = %v", data, err)
	}
}
