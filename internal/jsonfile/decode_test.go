package jsonfile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStrictRoundTrip(t *testing.T) {
	type document struct {
		Name string `json:"name"`
	}
	path := filepath.Join(t.TempDir(), "nested", "document.json")
	if err := Write(path, document{Name: "overgo"}, 0o600); err != nil {
		t.Fatal(err)
	}
	var decoded document
	if err := DecodeStrict(path, &decoded); err != nil || decoded.Name != "overgo" {
		t.Fatalf("round trip = (%+v, %v)", decoded, err)
	}
	if err := os.WriteFile(path, []byte(`{"name":"overgo","unknown":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := DecodeStrict(path, &decoded); err == nil {
		t.Fatal("strict decode accepted an unknown field")
	}
}
