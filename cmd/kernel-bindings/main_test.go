package main

import (
	"os"
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
