// build-kernels: nvcc PTX build + runtime pin refresh (floor component 8;
// replaces build-kernels.ps1). The ABI manifest owns kernel provenance; this
// command owns only the invocation: compile each .cu to PTX for the pinned
// device class, refresh the SHA-256 runtime pins in the validation source,
// then regenerate and verify the manifest so provenance and bytes agree.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// deviceArch is the pinned primary device class (compute capability 8.9, per
// the compatibility baseline). Changing device class is a baseline decision,
// not a flag.
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

var runtimePins = map[string]string{
	"VectorAddSHA256":      "internal/cuda/kernel/vector_add.ptx",
	"OpsF32SHA256":         "internal/cuda/kernel/ops_f32.ptx",
	"TorchCudaRandnSHA256": "internal/cuda/kernel/torch_cuda_randn.ptx",
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "build-kernels: %v\n", err)
		os.Exit(1)
	}
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
		fmt.Printf("generated %s\n", item.output)
	}
	if err := refreshPins("internal/cuda/kernel/manifest_generated.go"); err != nil {
		return err
	}
	for _, args := range [][]string{{"-update"}, nil} {
		cmd := exec.Command("go", append([]string{"run", "./cmd/kernel-manifest"}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("kernel-manifest %v: %v: %s", args, err, strings.TrimSpace(string(out)))
		}
	}
	fmt.Println("kernel manifest updated and verified")
	return nil
}

func toolchain() (nvcc, compiler string, err error) {
	cudaRoot := strings.TrimSpace(os.Getenv("CUDA_PATH"))
	if cudaRoot == "" {
		cudaRoot = `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`
	}
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

func refreshPins(validationPath string) error {
	raw, err := os.ReadFile(filepath.FromSlash(validationPath))
	if err != nil {
		return err
	}
	text := string(raw)
	for name, asset := range runtimePins {
		payload, err := os.ReadFile(filepath.FromSlash(asset))
		if err != nil {
			return err
		}
		digest := sha256.Sum256(payload)
		pattern := regexp.MustCompile(`(?m)^(\s*` + regexp.QuoteMeta(name) + `\s*=\s*")[0-9a-f]{64}("\s*)$`)
		if len(pattern.FindAllString(text, -1)) != 1 {
			return fmt.Errorf("expected exactly one %s runtime pin in %s", name, validationPath)
		}
		text = pattern.ReplaceAllString(text, "${1}"+hex.EncodeToString(digest[:])+"${2}")
	}
	return os.WriteFile(filepath.FromSlash(validationPath), []byte(text), 0o644)
}
