package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
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

func TestSBOMOmitUndeclaredLicenses(t *testing.T) {
	generated, err := generate("../..")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(generated, []byte("NOASSERTION")) {
		t.Fatal("SBOM retains undeclared-license policy")
	}
	var document struct {
		Metadata struct {
			Component map[string]any `json:"component"`
		} `json:"metadata"`
	}
	if err := json.Unmarshal(generated, &document); err != nil {
		t.Fatal(err)
	}
	if _, ok := document.Metadata.Component["licenses"]; ok {
		t.Fatal("project component declares a license")
	}
}

func TestReleaseIntegrityContract(t *testing.T) {
	generated, err := generate("../..")
	if err != nil {
		t.Fatal(err)
	}
	committed, err := os.ReadFile(filepath.Join("..", "..", sbomPath))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(generated, committed) {
		t.Fatal("committed SBOM does not match the dependency graph")
	}
}
