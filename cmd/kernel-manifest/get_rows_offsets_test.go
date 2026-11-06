package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestGetRowsAddressAWholeTable holds every get_rows kernel of the kernel sources, hand-written or a macro's, to an
// offset into its table taken in size_t: a 32-bit offset wraps past 4 GiB and reads another row's bytes, as mul_mat's
// size_t offsets do not. Each read of the table, table[...] or table + ..., widens before it multiplies.
func TestGetRowsAddressAWholeTable(t *testing.T) {
	t.Parallel()
	sources := testcheck.Must(filepath.Glob(filepath.Join("..", "..", "kernels", "cuda", "*.cu")))(t)
	kernels := 0
	for _, path := range sources {
		source := string(testcheck.Must(os.ReadFile(path))(t))
		for at := 0; ; {
			found := strings.Index(source[at:], "__global__ void get_rows_")
			if found < 0 {
				break
			}
			start := at + found
			text := kernelBody(source, start)
			at = start + len(text) + 1
			kernels++
			tokens := cudaTokens(text)
			for index := 0; index+2 < len(tokens); index++ {
				if tokens[index] != "table" || tokens[index+1] != "[" && tokens[index+1] != "+" {
					continue
				}
				if !slices.Equal(tokens[index+2:min(index+5, len(tokens))], []string{"(", "size_t", ")"}) {
					t.Errorf("%s: %s reads its table at a 32-bit offset, %s", filepath.Base(path), strings.Fields(text)[2], strings.Join(tokens[index:min(index+8, len(tokens))], " "))
				}
			}
		}
	}
	if kernels == 0 {
		t.Fatal("the kernel sources hold no get_rows kernel")
	}
}

// kernelBody is the kernel declared at start in source, through its closing brace.
func kernelBody(source string, start int) string {
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
	return source[start:]
}
