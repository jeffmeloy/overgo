// build-kernels: compile PTX, refresh pins, verify manifest.
package main

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"overgo/internal/clioptions"
)

// deviceArch: pinned compatibility target.
const deviceArch = "compute_89"

type kernel struct {
	source, output string
	// noFastMath preserves native arithmetic and subnormals in source-pinned modules.
	noFastMath bool
}

var kernels = []kernel{
	{source: "kernels/cuda/vector_add.cu", output: "internal/cuda/kernel/vector_add.ptx"},
	{source: "kernels/cuda/ops_f32.cu", output: "internal/cuda/kernel/ops_f32.ptx"},
	{source: "kernels/cuda/ops_native.cu", output: "internal/cuda/kernel/ops_native.ptx", noFastMath: true},
	{source: "kernels/cuda/torch_cuda_randn.cu", output: "internal/cuda/kernel/torch_cuda_randn.ptx", noFastMath: true},
}

var runtimePins = map[string]string{
	"VectorAddSHA256":      "internal/cuda/kernel/vector_add.ptx",
	"OpsF32SHA256":         "internal/cuda/kernel/ops_f32.ptx",
	"OpsNativeSHA256":      "internal/cuda/kernel/ops_native.ptx",
	"TorchCUDARandnSHA256": "internal/cuda/kernel/torch_cuda_randn.ptx",
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
			// Source-pinned scalar math and curand require standard precision.
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
		payload = bytes.ReplaceAll(payload, []byte("\r\n"), []byte("\n"))
		if err := clioptions.WriteOutputFile(filepath.FromSlash(asset), payload); err != nil {
			return err
		}
		digest := sha256.Sum256(payload)
		pattern := regexp.MustCompile(`(?m)^(\s*` + regexp.QuoteMeta(name) + `\s*=\s*")[0-9a-f]{64}("\s*)$`)
		if len(pattern.FindAllString(text, -1)) != 1 {
			return fmt.Errorf("expected exactly one %s runtime pin in %s", name, validationPath)
		}
		text = pattern.ReplaceAllString(text, "${1}"+hex.EncodeToString(digest[:])+"${2}")
	}
	return clioptions.WriteOutputFile(filepath.FromSlash(validationPath), []byte(text))
}
