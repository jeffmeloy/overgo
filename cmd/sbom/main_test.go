package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

func TestGenerateIsDeterministicAndValidJSON(t *testing.T) {
	first, err := generate("../..")
	testcheck.NoError(t, err)
	second, err := generate("../..")
	testcheck.NoError(t, err)
	if !bytes.Equal(first, second) {
		t.Fatal("SBOM output is not deterministic")
	}
	var document map[string]any
	testcheck.NoError(t, json.Unmarshal(first, &document))
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
	testcheck.NoError(t, err)
	if bytes.Contains(generated, []byte("NOASSERTION")) {
		t.Fatal("SBOM retains undeclared-license policy")
	}
	var document struct {
		Metadata struct {
			Component map[string]any `json:"component"`
		} `json:"metadata"`
	}
	testcheck.NoError(t, json.Unmarshal(generated, &document))
	if _, ok := document.Metadata.Component["licenses"]; ok {
		t.Fatal("project component declares a license")
	}
}

func TestSBOMOmitReferenceOnlyComponents(t *testing.T) {
	generated, err := generate("../..")
	testcheck.NoError(t, err)
	for _, reference := range [][]byte{
		[]byte("llama.cpp compatibility baseline"),
		[]byte("pkg:github/ggml-org/llama.cpp"),
		[]byte("overgo:upstream-commit"),
	} {
		if bytes.Contains(generated, reference) {
			t.Fatalf("SBOM contains reference-only component %q", reference)
		}
	}
	for _, used := range [][]byte{
		[]byte("internal/cuda/kernel/torch_cuda_randn.ptx"),
		[]byte("kernels/cuda/torch_cuda_randn.cu"),
	} {
		if !bytes.Contains(generated, used) {
			t.Fatalf("SBOM omits used kernel component %q", used)
		}
	}
}

func TestReleaseIntegrityContract(t *testing.T) {
	generated, err := generate("../..")
	testcheck.NoError(t, err)
	committed, err := os.ReadFile(filepath.Join("..", "..", sbomPath))
	testcheck.NoError(t, err)
	if !bytes.Equal(generated, committed) {
		t.Fatal("committed SBOM does not match the dependency graph")
	}
}
