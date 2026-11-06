package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode"

	"overgo/internal/testcheck"
)

// unnoticedSources are the projects whose code the kernels do not carry: they match these projects' results, written
// independently (decision kernel-provenance), and licenses/ holds no notice for them.
var unnoticedSources = []string{"torch", "pytorch", "cutlass", "adaptive_new"}

// commitDigits is the fewest hexadecimal characters a word holds to read as a commit, as git abbreviates one.
const commitDigits = 7

// TestNoKernelCommentClaimsUnnoticedCode holds the kernel sources' comments to what decision kernel-provenance settled:
// no comment that names Torch, CUTLASS or adaptive_new says the code is ported, verbatim, followed, defined or copied
// from it, or cites it at a commit. A comment may say whose results a kernel matches; a claim of derived code would
// need a notice the bill of materials does not carry.
func TestNoKernelCommentClaimsUnnoticedCode(t *testing.T) {
	t.Parallel()
	sources := testcheck.Must(filepath.Glob(filepath.Join("..", "..", "kernels", "cuda", "*.cu*")))(t)
	if len(sources) == 0 {
		t.Fatal("no kernel source")
	}
	for _, path := range sources {
		for _, block := range commentBlocks(string(testcheck.Must(os.ReadFile(path))(t))) {
			words := strings.FieldsFunc(strings.ToLower(block.text), func(r rune) bool {
				return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
			})
			names, claims := false, false
			for _, word := range words {
				for _, source := range unnoticedSources {
					names = names || word == source || strings.HasPrefix(word, source+"_")
				}
				claims = claims || derivationWord(word)
			}
			if names && claims {
				t.Errorf("%s:%d claims code of a project with no notice: %s", filepath.Base(path), block.line, block.text)
			}
		}
	}
}

// commentBlock is a run of comment lines, one comment as a reader takes it, and the line it starts on.
type commentBlock struct {
	text string
	line int
}

// commentBlocks are source's comments: each run of whole-line comments, and each comment after code.
func commentBlocks(source string) []commentBlock {
	var blocks []commentBlock
	// whole is whether the last line was a comment alone, which a comment alone on the next line continues.
	whole := false
	for number, line := range strings.Split(source, "\n") {
		code, comment, found := strings.Cut(line, "//")
		own := found && strings.TrimSpace(code) == ""
		switch {
		case !found:
		case own && whole:
			blocks[len(blocks)-1].text += " " + strings.TrimSpace(comment)
		default:
			blocks = append(blocks, commentBlock{strings.TrimSpace(comment), number + 1})
		}
		whole = own
	}
	return blocks
}

// derivationWord reports whether a comment's word claims code was taken from a source: ported, verbatim, follows,
// defines, copied, a source's owners or a commit.
func derivationWord(word string) bool {
	for _, stem := range []string{"port", "verbatim", "follow", "defin", "copied", "copy", "owner"} {
		if strings.HasPrefix(word, stem) {
			return true
		}
	}
	if len(word) < commitDigits || strings.Trim(word, "0123456789abcdef") != "" {
		return false
	}
	return strings.ContainsAny(word, "0123456789") && strings.ContainsAny(word, "abcdef")
}
