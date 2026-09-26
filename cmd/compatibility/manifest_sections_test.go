package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// TestManifestHoldsOnlyDecodedSections holds compatibility.json to the
// sections the manifest decodes: a top-level section no field reads is prose
// nothing checks, so it cannot be added.
func TestManifestHoldsOnlyDecodedSections(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile(filepath.Join("..", "..", manifestPath))
	if err != nil {
		t.Fatal(err)
	}
	var sections map[string]json.RawMessage
	if err := json.Unmarshal(data, &sections); err != nil {
		t.Fatal(err)
	}
	decoded := map[string]bool{}
	fields := reflect.TypeFor[manifest]()
	for index := range fields.NumField() {
		name, _, _ := strings.Cut(fields.Field(index).Tag.Get("json"), ",")
		decoded[name] = true
	}
	for section := range sections {
		if !decoded[section] {
			t.Errorf("%s section %q is read by no manifest field; decode it or delete it", manifestPath, section)
		}
	}
	if len(sections) == 0 {
		t.Fatal("no section was measured")
	}
}
