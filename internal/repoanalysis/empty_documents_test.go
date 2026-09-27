package repoanalysis

import (
	"encoding/json"
	"os"
	"path"
	"path/filepath"
	"testing"
)

// TestNoDocumentRecordsNothing refuses a tracked JSON document that records
// nothing beyond its version and description: a ledger whose last entry left
// has become its own rule, which the check enforces without it.
func TestNoDocumentRecordsNothing(t *testing.T) {
	t.Parallel()
	root := "../.."
	documents, err := TrackedDocuments(t.Context(), root)
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, document := range documents {
		if path.Ext(document) != ".json" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(document)))
		if err != nil {
			t.Fatal(err)
		}
		var value any
		if json.Unmarshal(data, &value) != nil {
			continue
		}
		checked++
		if object, isObject := value.(map[string]any); isObject {
			delete(object, "version")
			delete(object, "doc")
		}
		if !recordsSomething(value) {
			t.Errorf("%s records nothing; enforce its rule in the check and delete it", document)
		}
	}
	if checked == 0 {
		t.Fatal("no JSON document was checked")
	}
}

// recordsSomething reports whether a decoded JSON value holds any entry: a
// number, a boolean, a non-empty string, or a container holding one.
func recordsSomething(value any) bool {
	switch typed := value.(type) {
	case nil:
		return false
	case string:
		return typed != ""
	case []any:
		for _, element := range typed {
			if recordsSomething(element) {
				return true
			}
		}
		return false
	case map[string]any:
		for _, element := range typed {
			if recordsSomething(element) {
				return true
			}
		}
		return false
	default:
		return true
	}
}
