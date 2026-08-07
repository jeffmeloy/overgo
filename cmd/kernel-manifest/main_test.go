package main

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	cudaKernel "overgo/internal/cuda/kernel"
)

func TestPTXEntriesSorted(t *testing.T) {
	got := ptxEntries([]byte(".visible .entry z(\n.visible .entry a(\n"))
	if !slices.Equal(got, []string{"a", "z"}) {
		t.Fatalf("entries = %v", got)
	}
}

func TestPTXEntryABIsPreserveOrderSizeAndAlignment(t *testing.T) {
	data := []byte(`.visible .entry x(
	.param .u64 x_param_0,
	.param .align 8 .b8 x_param_1[16],
	.param .f32 x_param_2
)
`)
	got := ptxEntryABIs(data)["x"]
	want := []string{".u64", ".align 8 .b8[16]", ".f32"}
	if !slices.Equal(got, want) {
		t.Fatalf("ABI = %v, want %v", got, want)
	}
}

func TestRunUpdatesAndRejectsStaleManifest(t *testing.T) {
	root := t.TempDir()
	for _, directory := range []string{"kernels", "src", "asset"} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "src", "x.cu"), []byte("kernel"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "src", "x.cuh"), []byte("table"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(root, "asset", "x.ptx"),
		[]byte(".visible .entry x(\n\t.param .u64 x_param_0\n)\n"),
		0o644,
	); err != nil {
		t.Fatal(err)
	}
	document := fmt.Sprintf(`{
  "schema": 2,
  "abiVersion": %d,
  "cudaToolkit": "12.9",
  "nvcc": "12.9.86",
  "target": "compute_89",
  "compilerFlags": ["-ptx"],
  "defaultThreads": 256,
  "sharedMemoryABI": "explicit-per-launch-v1",
  "modules": [{
    "source": "src/x.cu",
    "sourceSha256": "",
    "sourceDependencies": [{
      "path": "src/x.cuh",
      "sha256": ""
    }],
    "asset": "asset/x.ptx",
    "assetSha256": "",
    "functions": ["x"],
    "argumentLayouts": [{
      "parameters": [".u64"],
      "functions": ["x"]
    }]
  }]
}`, cudaKernel.BundleABIVersion)
	if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(manifestPath)), []byte(document), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run(root, true); err != nil {
		t.Fatal(err)
	}
	if err := run(root, false); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "src", "x.cuh"), []byte("changed"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run(root, false); err == nil {
		t.Fatal("stale source dependency was accepted")
	}
	if err := os.WriteFile(filepath.Join(root, "src", "x.cuh"), []byte("table"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "asset", "x.ptx"), []byte("changed"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := run(root, false); err == nil {
		t.Fatal("stale manifest was accepted")
	}
	if err := os.WriteFile(
		filepath.Join(root, "asset", "x.ptx"),
		[]byte(".visible .entry x(\n\t.param .u32 x_param_0\n)\n"),
		0o644,
	); err != nil {
		t.Fatal(err)
	}
	if err := run(root, true); err != nil {
		t.Fatal(err)
	}
	if err := run(root, false); err != nil {
		t.Fatal(err)
	}
}
