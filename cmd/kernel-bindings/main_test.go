package main

import (
	"encoding/json"
	"os"
	"testing"
)

func TestBindingModuleRolesAndCollisions(t *testing.T) {
	module := func(asset, name string, executor bool) kernelModule {
		return kernelModule{Asset: asset, Executor: executor, Functions: []string{name}, ArgumentLayouts: []kernelArgumentLayout{{Functions: []string{name}, Parameters: []string{".u64"}}}}
	}
	m := kernelManifest{Modules: []kernelModule{module("ops_f32.ptx", "fast", true), module("ops_native.ptx", "precise", true), module("vector_add.ptx", "fast", false)}}
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = generateBindings(data); err != nil {
		t.Fatal("distinct active modules or excluded duplicate refused", err)
	}
	m.Modules[2].Executor = true
	data, err = json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = generateBindings(data); err == nil {
		t.Fatal("ambiguous cross-module entry admitted")
	}
	for i := range m.Modules {
		m.Modules[i].Executor = false
	}
	data, err = json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = generateBindings(data); err == nil {
		t.Fatal("empty tensor execution role admitted")
	}
}

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
