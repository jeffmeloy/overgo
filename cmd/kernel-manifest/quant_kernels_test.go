package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode"

	"overgo/internal/testcheck"
)

// smallQuantMacros are the macros whose expansions are the small-quant get_rows and mul_mat kernels, in the order a
// tag's pair stands.
var smallQuantMacros = []string{"DEFINE_SMALL_QUANT_GET_ROWS", "DEFINE_SMALL_QUANT_MUL_MAT"}

// TestSmallQuantKernelsHaveOneBody holds kernels/cuda/ops_f32.cu to one body for the small-quant get_rows and mul_mat
// kernels, DEFINE_SMALL_QUANT_KERNELS: no hand-written get_rows_<tag>_f32 and mul_mat_<tag>_f32 pair is, token for
// token, the macro's expansion for its tag, block width, block bytes and dequantizer, which the macro then compiles to
// the same code.
func TestSmallQuantKernelsHaveOneBody(t *testing.T) {
	t.Parallel()
	source := string(testcheck.Must(os.ReadFile(filepath.Join("..", "..", "kernels", "cuda", "ops_f32.cu")))(t))
	var template []string
	for _, name := range smallQuantMacros {
		template = append(template, cudaTokens(macroBody(t, source, name))...)
	}
	for _, rows := range kernelNames(source, "get_rows_") {
		tag := strings.TrimSuffix(strings.TrimPrefix(rows, "get_rows_"), "_f32")
		getRows, mulMat := kernelText(source, rows), kernelText(source, "mul_mat_"+tag+"_f32")
		if getRows == "" || mulMat == "" {
			continue
		}
		got := cudaTokens(getRows + "\n" + mulMat)
		width, bytes, dequant := after(got, "width", "/"), after(got, "block_index", "*"), after(got, "]", "=")
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
			t.Errorf("get_rows_%s_f32 and mul_mat_%s_f32 are written out by hand; say them as DEFINE_SMALL_QUANT_KERNELS(%s, %s, %s, %s)", tag, tag, tag, width, bytes, dequant)
		}
	}
}

// macroBody is the replacement text of the #define of name, its continuation lines joined.
func macroBody(t *testing.T, source, name string) string {
	t.Helper()
	start := strings.Index(source, "#define "+name+"(")
	if start < 0 {
		t.Fatalf("ops_f32.cu defines no %s", name)
	}
	var body []string
	for _, line := range strings.Split(source[start:], "\n")[1:] {
		trimmed, continued := strings.CutSuffix(strings.TrimRight(line, "\r"), "\\")
		body = append(body, trimmed)
		if !continued {
			break
		}
	}
	return strings.Join(body, "\n")
}

// kernelNames are the names of the extern "C" kernels written out in source whose names begin with prefix.
func kernelNames(source, prefix string) []string {
	const opening = `extern "C" __global__ void `
	var names []string
	for line := range strings.SplitSeq(source, "\n") {
		if rest, found := strings.CutPrefix(line, opening+prefix); found {
			if name, _, open := strings.Cut(rest, "("); open && !strings.Contains(name, "#") {
				names = append(names, prefix+name)
			}
		}
	}
	return names
}

// kernelText is the extern "C" kernel name written out in source, from its declaration to its closing brace, or
// nothing when source holds none.
func kernelText(source, name string) string {
	start := strings.Index(source, `extern "C" __global__ void `+name+"(")
	if start < 0 {
		return ""
	}
	depth := 0
	for index := strings.Index(source[start:], "{") + start; index < len(source); index++ {
		switch source[index] {
		case '{':
			depth++
		case '}':
			if depth--; depth == 0 {
				return source[start : index+1]
			}
		}
	}
	return ""
}

// cudaTokens splits CUDA source into identifiers, which keep a token paste's ##, numbers and single punctuation,
// dropping line comments and space.
func cudaTokens(text string) []string {
	var tokens []string
	for line := range strings.SplitSeq(text, "\n") {
		code, _, _ := strings.Cut(line, "//")
		runes := []rune(code)
		for index := 0; index < len(runes); {
			switch character := runes[index]; {
			case unicode.IsSpace(character):
				index++
			case unicode.IsLetter(character) || unicode.IsDigit(character) || character == '_' || character == '#' && index+1 < len(runes) && runes[index+1] == '#':
				end := index
				for end < len(runes) && (unicode.IsLetter(runes[end]) || unicode.IsDigit(runes[end]) || runes[end] == '_' || runes[end] == '#') {
					end++
				}
				tokens = append(tokens, string(runes[index:end]))
				index = end
			default:
				tokens = append(tokens, string(character))
				index++
			}
		}
	}
	return tokens
}

// after is the token that follows the first first, then second when it is given, in tokens.
func after(tokens []string, first, second string) string {
	for index := 0; index+2 < len(tokens); index++ {
		if tokens[index] != first {
			continue
		}
		if second == "" {
			return tokens[index+1]
		}
		if tokens[index+1] == second {
			return tokens[index+2]
		}
	}
	return ""
}
