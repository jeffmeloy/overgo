package jsonfile

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

func TestStrictRoundTrip(t *testing.T) {
	type document struct {
		Name string `json:"name"`
	}
	path := filepath.Join(t.TempDir(), "nested", "document.json")
	testcheck.NoError(t, os.Mkdir(filepath.Dir(path), 0o700))
	testcheck.NoError(t, Write(path, document{Name: "overgo"}, 0o600))
	if err := Write(path, document{Name: "overgo-rewritten"}, 0o600); err != nil {
		t.Fatalf("atomic rewrite: %v", err)
	}
	var decoded document
	if err := DecodeStrict(path, &decoded); err != nil || decoded.Name != "overgo-rewritten" {
		t.Fatalf("round trip = (%+v, %v)", decoded, err)
	}
	testcheck.NoError(t, os.WriteFile(path, []byte(`{"name":"overgo","unknown":true}`), 0o600))
	if err := DecodeStrict(path, &decoded); err == nil {
		t.Fatal("strict decode accepted an unknown field")
	}
}
