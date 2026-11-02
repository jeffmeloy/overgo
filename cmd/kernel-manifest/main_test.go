package main

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	cudaKernel "overgo/internal/cuda/kernel"
)

// TestEveryIncludedKernelSourceIsPinned: every header a module compiles in is pinned, whichever file includes it, so a
// planted include refuses the manifest, and a pin of a header no longer included does too.
func TestEveryIncludedKernelSourceIsPinned(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		file := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("kernels/cuda/main.cu", "#include <stdint.h>\n  #include \"first.cuh\"\n// #include is prose here\n")
	write("kernels/cuda/first.cuh", "#  include \"nested/second.cuh\"\n")
	write("kernels/cuda/nested/second.cuh", "#include \"../first.cuh\"\n")
	included, err := includedSources(root, "kernels/cuda/main.cu")
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"kernels/cuda/first.cuh", "kernels/cuda/nested/second.cuh"}; !slices.Equal(included, want) {
		t.Fatalf("included = %v, want %v", included, want)
	}
	item := &module{Source: "kernels/cuda/main.cu", SourceDependencies: pinnedAs(included, nil)}
	if err := requirePinned(item, included); err != nil {
		t.Fatal(err)
	}
	write("kernels/cuda/first.cuh", "#include \"planted.cuh\"\n#include \"nested/second.cuh\"\n")
	write("kernels/cuda/planted.cuh", "\n")
	planted, err := includedSources(root, "kernels/cuda/main.cu")
	if err != nil {
		t.Fatal(err)
	}
	if err := requirePinned(item, planted); err == nil || !strings.Contains(err.Error(), "planted.cuh") {
		t.Fatalf("a planted include: %v", err)
	}
	if err := requirePinned(item, included[:1]); err == nil || !strings.Contains(err.Error(), "does not include") {
		t.Fatalf("a pin of a header no longer included: %v", err)
	}
}

// fixtureCompiler is the line nvcc writes into a PTX file, naming the toolchain the fixture manifest records.
const fixtureCompiler = "// Cuda compilation tools, release 12.9, V12.9.86\n"

// TestRecordedToolchainMatchesCompiledCode: a PTX file naming another release is refused, as one that names none is, and
// the committed assets name what the manifest records.
func TestRecordedToolchainMatchesCompiledCode(t *testing.T) {
	t.Parallel()
	recorded := manifest{CUDAToolkit: "12.9", NVCC: "12.9.86"}
	if err := requireToolchain(recorded, "x.ptx", []byte(fixtureCompiler)); err != nil {
		t.Fatal(err)
	}
	for name, ptx := range map[string]string{
		"another release": "// Cuda compilation tools, release 13.0, V13.0.1\n",
		"another build":   "// Cuda compilation tools, release 12.9, V12.9.99\n",
		"no compiler":     ".version 8.8\n",
	} {
		if err := requireToolchain(recorded, "x.ptx", []byte(ptx)); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
}

func TestPTXEntriesSorted(t *testing.T) {
	got := ptxEntries([]byte(fixtureCompiler + ".visible .entry z(\n.visible .entry a(\n"))
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

func TestFileHashCanonicalizesCheckoutNewlines(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "source.cu")
	if err := os.WriteFile(path, []byte("first\nsecond\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	want, err := fileHash(root, "source.cu")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("first\r\nsecond\r\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := fileHash(root, "source.cu")
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("checkout newline changed source hash: %s != %s", got, want)
	}
}

func TestRunUpdatesAndRejectsStaleManifest(t *testing.T) {
	root := t.TempDir()
	for _, directory := range []string{"kernels", "src", "asset"} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "src", "x.cu"), []byte("#include \"x.cuh\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "src", "x.cuh"), []byte("table"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(root, "asset", "x.ptx"),
		[]byte(fixtureCompiler+".visible .entry x(\n\t.param .u64 x_param_0\n)\n"),
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
  "vaeConvTile": %d,
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
}`, cudaKernel.BundleABIVersion, cudaKernel.BundleVAEConvTile)
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
		[]byte(fixtureCompiler+".visible .entry x(\n\t.param .u32 x_param_0\n)\n"),
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
