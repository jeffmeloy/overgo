package modeltest

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
)

// TestRegisteredGGUFSelection holds GGUF selection to the registered files it
// can read: the smallest file declaring the architecture is chosen, through
// the canonical reader when the architecture is not the header's first entry;
// an absent registration or an architecture nothing declares selects nothing,
// which the caller reports as inapplicable; a registered file that cannot be
// read is an error, never an architecture it lacks, and is read again once it
// is repaired. Tiny fixtures stand in for models, so no model or device is
// needed.
func TestRegisteredGGUFSelection(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	encode := func(metadata ...gguf.Metadata) []byte {
		t.Helper()
		var encoded bytes.Buffer
		if err := gguf.Write(&encoded, metadata, nil, gguf.WriteOptions{}); err != nil {
			t.Fatal(err)
		}
		return encoded.Bytes()
	}
	text := func(key, value string) gguf.Metadata {
		return gguf.Metadata{Key: key, Value: gguf.Value{Type: gguf.ValueTypeString, Data: value}}
	}
	write := func(name string, data []byte) string {
		t.Helper()
		path := filepath.Join(directory, name)
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	valid := encode(text(architectureKey, "fixture"))
	small := write("small.gguf", valid)
	larger := write("larger.gguf", encode(text(architectureKey, "fixture"), text("general.name", "a larger file declaring the same architecture")))
	late := write("late.gguf", encode(text("general.name", "an architecture after another entry"), text(architectureKey, "late")))
	absent := filepath.Join(directory, "absent.gguf")
	notGGUF := write("weights.bin", valid)
	for _, check := range []struct {
		name, architecture string
		locations          []string
		want               string
	}{
		{"smallest declaring file", "fixture", []string{larger, absent, small, late}, small},
		{"architecture after another entry", "late", []string{small, late}, late},
		{"architecture nothing declares", "other", []string{small, larger, late}, ""},
		{"absent registration", "fixture", []string{absent}, ""},
		{"not a GGUF location", "fixture", []string{notGGUF}, ""},
	} {
		if got, err := smallestGGUF(check.locations, check.architecture); err != nil || got != check.want {
			t.Errorf("%s: selected %q, %v; want %q", check.name, got, err, check.want)
		}
	}
	broken := write("broken.gguf", valid[:len(gguf.Magic)+1])
	if got, err := smallestGGUF([]string{small, broken}, "fixture"); err == nil {
		t.Fatalf("an unreadable registered GGUF selected %q instead of failing", got)
	}
	if got, err := smallestGGUF([]string{broken}, "other"); err == nil {
		t.Fatalf("an unreadable registered GGUF declared no architecture (%q) instead of failing", got)
	}
	write("broken.gguf", valid)
	if got, err := smallestGGUF([]string{broken}, "fixture"); err != nil || got != broken {
		t.Fatalf("the repaired GGUF selected %q, %v; a failed read was remembered", got, err)
	}
}
