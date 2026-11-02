package main

import (
	"os"
	"strings"
	"testing"
)

func TestGeneratedBindingsMatchManifest(t *testing.T) {
	document, err := os.ReadFile("../../kernels/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	want, err := generateBindings(document)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile("../../internal/cuda/executor/kernel_bindings_generated.go")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatal("CUDA kernel bindings are stale; run go generate ./internal/cuda/executor")
	}
	kernelWant, err := generateKernelManifest(document)
	if err != nil {
		t.Fatal(err)
	}
	kernelGot, err := os.ReadFile("../../internal/cuda/kernel/manifest_generated.go")
	if err != nil {
		t.Fatal(err)
	}
	if string(kernelGot) != string(kernelWant) {
		t.Fatal("CUDA kernel manifest constants are stale; run go generate ./internal/cuda/executor")
	}
}

// TestBindingsRecordArgumentWidths: each parameter's width reaches the bindings, scalars and by-value arrays alike, and a
// parameter of no known shape refuses generation rather than passing unchecked.
func TestBindingsRecordArgumentWidths(t *testing.T) {
	for parameter, want := range map[string]int{".u64": 8, ".u32": 4, ".f32": 4, ".f64": 8, ".align 8 .b8[72]": 72, ".align 8 .b8[1232]": 1232} {
		if got, err := parameterWidth(parameter); err != nil || got != want {
			t.Errorf("width of %q = %d, %v; want %d", parameter, got, err, want)
		}
	}
	if _, err := parameterWidth(".opaque"); err == nil {
		t.Fatal("a parameter of no known width was accepted")
	}
	document, err := os.ReadFile("../../kernels/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	generated, err := generateBindings(document)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(generated), "var kernelFunctionArgumentWidths = [...][]uint16{") {
		t.Fatal("the bindings record no widths")
	}
}

func TestKernelIdentifierPreservesABIWords(t *testing.T) {
	const (
		name = "mul_mat_q8_K_f32"
		want = "kernelMulMatQ8KF32"
	)
	if got := kernelIdentifier(name); got != want {
		t.Fatalf("identifier = %q, want %q", got, want)
	}
}

func TestAssetIdentifierPreservesCUDA(t *testing.T) {
	if got, want := assetIdentifier("torch_cuda_randn.ptx"), "TorchCUDARandn"; got != want {
		t.Fatalf("identifier = %q, want %q", got, want)
	}
}
