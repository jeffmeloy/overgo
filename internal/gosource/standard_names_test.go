package gosource

import "testing"

// TestStandardNamesResolveOnce holds package-name resolution to asking go
// list for a standard-library package once per process: the toolchain fixes
// its name, and every code-manifest generation asked again for the same few
// hundred packages. With go off the path the second resolution still names
// them, so it asked nothing.
func TestStandardNamesResolveOnce(t *testing.T) {
	root := t.TempDir()
	imports := []string{"encoding/json", "net/http/httptest"}
	first, err := ImportNames(root, imports)
	if err != nil || first["encoding/json"] != "json" || first["net/http/httptest"] != "httptest" {
		t.Fatalf("first resolution = %v, %v", first, err)
	}
	t.Setenv("PATH", "")
	second, err := ImportNames(root, imports)
	if err != nil || second["encoding/json"] != "json" || second["net/http/httptest"] != "httptest" {
		t.Fatalf("standard names were asked for again: %v, %v", second, err)
	}
}
