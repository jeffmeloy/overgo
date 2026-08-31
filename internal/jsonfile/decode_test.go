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
	if err := os.Mkdir(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := Write(path, document{Name: "overgo"}, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := Write(path, document{Name: "overgo-rewritten"}, 0o600); err != nil {
		t.Fatalf("atomic rewrite: %v", err)
	}
	var decoded document
	if err := DecodeStrict(path, &decoded); err != nil || decoded.Name != "overgo-rewritten" {
		t.Fatalf("round trip = (%+v, %v)", decoded, err)
	}
	if err := os.WriteFile(path, []byte(`{"name":"overgo","unknown":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := DecodeStrict(path, &decoded); err == nil {
		t.Fatal("strict decode accepted an unknown field")
	}
}
