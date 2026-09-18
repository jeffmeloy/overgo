package tokenizer

import "errors"

// These complete-coverage expressions identify the already implemented native
// splitters. CompileBPESplit recognizes artifact syntax, not model names.
const (
	// BPEPatternGPT2 is the default byte-level split expression declared by HF.
	BPEPatternGPT2   = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
	bpePatternQwen2  = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	bpePatternLlama3 = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

// CompileBPESplit returns an existing native scanner for an exact declared
// expression. Unknown expressions must not silently acquire another split.
func CompileBPESplit(pattern string) (func(string) []string, error) {
	switch pattern {
	case BPEPatternGPT2:
		return preTokenizeGPT2, nil
	case bpePatternQwen2:
		return preTokenizeQwen2, nil
	case bpePatternLlama3:
		return preTokenizeLlama3, nil
	default:
		return nil, errors.New("unsupported BPE split expression")
	}
}
