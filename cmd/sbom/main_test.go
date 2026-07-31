package main

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestGenerateIsDeterministicAndValidJSON(t *testing.T) {
	first, err := generate("../..")
	if err != nil {
		t.Fatal(err)
	}
	second, err := generate("../..")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("SBOM output is not deterministic")
	}
	var document map[string]any
	if err := json.Unmarshal(first, &document); err != nil {
		t.Fatal(err)
	}
	if document["bomFormat"] != "CycloneDX" || document["specVersion"] != "1.6" {
		t.Fatalf("document header = %#v", document)
	}
	components, ok := document["components"].([]any)
	if !ok || len(components) < 8 {
		t.Fatalf("components = %#v", document["components"])
	}
}
