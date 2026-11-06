package main

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestKernelNearCopiesHaveOneBody holds the kernel sources to one body for what consolidate/near-copies folded: no two
// __device__ functions of one kernel module, a source as its compiler reads it, whose bodies are one once their
// parameters' names are set aside, as the two little-endian word readers were, a struct's own members aside, which read
// its own constants, and a header kept in a namespace of its own keeping its own helpers; the relative position
// bucket's arithmetic once, in relative_position_bucket; and no hand-written dequant_rows_<tag>_f16 kernel that is,
// token for token, DEFINE_DEQUANT_ROWS_F16's expansion.
func TestKernelNearCopiesHaveOneBody(t *testing.T) {
	t.Parallel()
	for _, path := range testcheck.Must(filepath.Glob(filepath.Join("..", "..", "kernels", "cuda", "*.cu*")))(t) {
		// The ops_f32 module's family headers are read as the one text they compile in.
		if strings.HasPrefix(filepath.Base(path), "ops_f32_") {
			continue
		}
		source, seen := string(testcheck.Must(os.ReadFile(path))(t)), map[string]string{}
		if filepath.Base(path) == "ops_f32.cu" {
			source = moduleText(t)
		}
		for _, function := range deviceFunctions(source) {
			if first, repeated := seen[function.shape]; repeated {
				t.Errorf("%s: %s repeats %s's body", filepath.Base(path), function.name, first)
				continue
			}
			seen[function.shape] = function.name
		}
	}
	source := moduleText(t)
	bucket := cudaTokens("logf(128.0f / (float) max_exact)")
	if found := tokenRuns(cudaTokens(source), bucket); found != 1 {
		t.Errorf("ops_f32.cu takes the relative position bucket %d times, want once, in relative_position_bucket", found)
	}
	template := cudaTokens(macroBody(t, source, "DEFINE_DEQUANT_ROWS_F16"))
	for _, name := range kernelNames(source, "dequant_rows_") {
		tag := strings.TrimSuffix(strings.TrimPrefix(name, "dequant_rows_"), "_f16")
		got := cudaTokens(kernelText(source, name))
		width, bytes, dequant := after(got, "inner", "/"), after(got, ")", "*"), after(got, "__float2half", "(")
		want := make([]string, 0, len(template))
		for _, token := range template {
			switch token {
			case "BLOCK_WIDTH":
				token = width
			case "BLOCK_BYTES":
				token = bytes
			case "DEQUANT":
				token = dequant
			}
			want = append(want, strings.ReplaceAll(token, "##TAG##", tag))
		}
		if slices.Equal(got, want) {
			t.Errorf("%s is written out by hand; say it as DEFINE_DEQUANT_ROWS_F16(%s, %s, %s, %s)", name, tag, width, bytes, dequant)
		}
	}
}

// deviceFunction is one __device__ function of a kernel source: its name, and its shape, its tokens with each
// parameter named by its place.
type deviceFunction struct{ name, shape string }

// deviceFunctions are the __device__ functions source defines with a body.
func deviceFunctions(source string) []deviceFunction {
	var functions []deviceFunction
	for at := 0; ; {
		found := strings.Index(source[at:], "__device__ ")
		if found < 0 {
			return functions
		}
		start := at + found
		if strings.HasPrefix(source[start:], "__device__ static ") {
			at = start + len("__device__ ")
			continue
		}
		open, semicolon := strings.Index(source[start:], "{"), strings.Index(source[start:], ";")
		if open < 0 || semicolon >= 0 && semicolon < open {
			at = start + len("__device__ ")
			continue
		}
		text := kernelBody(source, start)
		at = start + len(text)
		tokens := cudaTokens(text)
		paren := slices.Index(tokens, "(")
		if paren < 1 {
			continue
		}
		parameters := map[string]string{}
		for index := paren + 1; index < len(tokens) && tokens[index] != ")"; index++ {
			if tokens[index+1] == "," || tokens[index+1] == ")" {
				parameters[tokens[index]] = "p" + strconv.Itoa(len(parameters))
			}
		}
		shape := make([]string, 0, len(tokens))
		for _, token := range tokens[paren:] {
			if renamed, isParameter := parameters[token]; isParameter {
				token = renamed
			}
			shape = append(shape, token)
		}
		functions = append(functions, deviceFunction{tokens[paren-1], strings.Join(tokens[:paren-1], " ") + " " + strings.Join(shape, " ")})
	}
}

// tokenRuns counts where run stands in tokens.
func tokenRuns(tokens, run []string) int {
	count := 0
	for index := 0; index+len(run) <= len(tokens); index++ {
		if slices.Equal(tokens[index:index+len(run)], run) {
			count++
		}
	}
	return count
}
