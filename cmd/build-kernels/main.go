// build-kernels: compile PTX, update the manifest, regenerate the bindings, verify them together.
package main

import (
	"bytes"
	"cmp"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"overgo/internal/clioptions"
)

// deviceArch: pinned compatibility target.
const deviceArch = "compute_89"

type kernel struct {
	source, output string
	// noFastMath omits -use_fast_math for this module. torch_cuda_randn draws
	// normals through logf/sqrtf/sincospif; the fast-math intrinsics change
	// those last bits and break byte-parity with PyTorch's standard-precision
	// curand. Bit-exact RNG is an ABI requirement, not a knob.
	noFastMath bool
}

var kernels = []kernel{
	{source: "kernels/cuda/vector_add.cu", output: "internal/cuda/kernel/vector_add.ptx"},
	{source: "kernels/cuda/ops_f32.cu", output: "internal/cuda/kernel/ops_f32.ptx"},
	{source: "kernels/cuda/torch_cuda_randn.cu", output: "internal/cuda/kernel/torch_cuda_randn.ptx", noFastMath: true},
}

func main() {
	clioptions.MainNamed("build-kernels", run)
}

func run() error {
	nvcc, compiler, err := toolchain()
	if err != nil {
		return err
	}
	for _, item := range kernels {
		args := []string{"-ptx", "-arch=" + deviceArch}
		if !item.noFastMath {
			args = append(args, "-use_fast_math")
		} else {
			// curand device headers are C++17 and resolve from the toolkit
			// include dir; keep standard precision for bit-exact normals.
			args = append(args, "--std=c++17")
		}
		args = append(args, "-ccbin", compiler, "-o", filepath.FromSlash(item.output), filepath.FromSlash(item.source))
		cmd := exec.Command(nvcc, args...)
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("nvcc %s: %v: %s", item.source, err, strings.TrimSpace(string(out)))
		}
		if err := normalizePTX(filepath.FromSlash(item.output)); err != nil {
			return err
		}
		fmt.Printf("generated %s\n", item.output)
	}
	// The manifest's hashes and argument layouts first, then the Go bindings and pins generated from it, its ABI
	// version included, then the check that holds the two together: a kernel's new parameters need no other step.
	for _, args := range [][]string{{"run", "./cmd/kernel-manifest", "-update"}, {"generate", "./internal/cuda/executor"}, {"run", "./cmd/kernel-manifest"}} {
		if out, err := exec.Command("go", args...).CombinedOutput(); err != nil {
			return fmt.Errorf("go %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
		}
	}
	fmt.Println("kernel manifest updated, bindings regenerated and both verified")
	return nil
}

func toolchain() (nvcc, compiler string, err error) {
	cudaRoot := strings.TrimSpace(os.Getenv("CUDA_PATH"))
	cudaRoot = cmp.Or(cudaRoot, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`)
	nvcc = filepath.Join(cudaRoot, "bin", "nvcc.exe")
	if _, statErr := os.Stat(nvcc); statErr != nil {
		return "", "", fmt.Errorf("nvcc not found at %s (set CUDA_PATH)", nvcc)
	}
	for _, candidate := range []string{
		`C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC`,
		`C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC`,
	} {
		versions, globErr := filepath.Glob(filepath.Join(candidate, "*", "bin", "Hostx64", "x64", "cl.exe"))
		if globErr == nil && len(versions) > 0 {
			return nvcc, filepath.Dir(versions[len(versions)-1]), nil
		}
		if _, statErr := os.Stat(filepath.Join(candidate, "cl.exe")); statErr == nil {
			return nvcc, candidate, nil
		}
	}
	return "", "", fmt.Errorf("no CUDA-compatible MSVC x64 compiler found")
}

// normalizePTX rewrites a generated PTX file with Unix line ends, which the pins hash whatever nvcc wrote.
func normalizePTX(path string) error {
	payload, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return clioptions.WriteOutputFile(path, bytes.ReplaceAll(payload, []byte("\r\n"), []byte("\n")))
}
