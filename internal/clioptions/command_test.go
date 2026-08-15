package clioptions

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteJSON(t *testing.T) {
	if got := Tail("abcdef", 3); got != "...def" {
		t.Fatalf("tail = %q", got)
	}
	if got := Tail("abc", 3); got != "abc" {
		t.Fatalf("unbounded tail = %q", got)
	}
	var compact bytes.Buffer
	if err := WriteJSON(&compact, map[string]int{"value": 3}); err != nil {
		t.Fatal(err)
	}
	if compact.String() != "{\"value\":3}\n" {
		t.Fatalf("compact JSON = %q", compact.String())
	}
	var pretty bytes.Buffer
	if err := WritePrettyJSON(&pretty, map[string]int{"value": 3}); err != nil {
		t.Fatal(err)
	}
	if pretty.String() != "{\n  \"value\": 3\n}\n" {
		t.Fatalf("pretty JSON = %q", pretty.String())
	}
}

func TestOutputGenerated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "generated.txt")
	if err := OutputGenerated([]byte("new"), path, false, true, "stale", &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if err := OutputGenerated([]byte("new"), path, true, false, "stale", &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if err := OutputGenerated([]byte("old"), path, true, false, "stale", &bytes.Buffer{}); err == nil || err.Error() != "stale" {
		t.Fatalf("stale error = %v", err)
	}
	var output bytes.Buffer
	if err := OutputGenerated([]byte("print"), path, false, false, "stale", &output); err != nil {
		t.Fatal(err)
	}
	if output.String() != "print" {
		t.Fatalf("output = %q", output.String())
	}
	if data, err := os.ReadFile(path); err != nil || string(data) != "new" {
		t.Fatalf("file = %q, error = %v", data, err)
	}
}
