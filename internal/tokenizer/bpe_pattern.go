package tokenizer

import "errors"

// Exact artifact expressions select shared native scanners. The RxBrain
// stages are partial matches and preserve unmatched spans as separate pieces.
// CompileBPESplit recognizes artifact syntax, not model names.
const (
	// BPEPatternGPT2 is the default byte-level split expression declared by HF.
	BPEPatternGPT2      = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
	bpePatternQwen2     = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	bpePatternLlama3    = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	bpePatternRxNumber  = "\\p{N}{1,3}"
	rxCJKClass          = "一-龥぀-ゟ゠-ヿ"
	bpePatternRxCJK     = "[" + rxCJKClass + "]+"
	bpePatternRxUnicode = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
)

// CompileBPESplit returns an exact native scanner and whether its matches
// cover every input rune. Unknown expressions never acquire another scanner.
func CompileBPESplit(pattern string) (func(string) []string, bool, error) {
	switch pattern {
	case BPEPatternGPT2:
		return preTokenizeGPT2, true, nil
	case bpePatternQwen2:
		return preTokenizeQwen2, true, nil
	case bpePatternLlama3:
		return preTokenizeLlama3, true, nil
	case bpePatternRxNumber:
		return preTokenizeRxNumber, false, nil
	case bpePatternRxCJK:
		return preTokenizeRxCJK, false, nil
	case bpePatternRxUnicode:
		return preTokenizeRxUnicode, false, nil
	default:
		return nil, false, errors.New("unsupported BPE split expression")
	}
}
