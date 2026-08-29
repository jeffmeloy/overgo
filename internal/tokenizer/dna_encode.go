// DNA-mode encoding: the hybrid-tokenizer input half for any model that
// declares the k-mer extension in its metadata. Nothing here names a model:
// the begin, end and out-of-vocabulary tags are the declared special tokens
// in declaration order, and k, the ID range and auto-tagging all come from
// the same metadata the decoder already trusts. Ported for parity with the
// upstream reference tokenizer: segmenting, k-mer chunking, out-of-alphabet
// substitution and right-pad rules are byte-for-byte.
package tokenizer

import (
	"fmt"
	"strings"
)

// Positions of the declared special tokens: begin tag, end tag, then the
// out-of-vocabulary marker. The declaration order is the format contract.
const (
	dnaSpecialBegin = 0
	dnaSpecialEnd   = 1
	dnaSpecialOOV   = 2
	dnaSpecialCount = 3
)

type dnaRegion struct {
	text string
	dna  bool
}

// splitDNARegions ports the upstream tag scanner: a region spans from a
// begin tag through its closing tag inclusive; an unopened end tag closes
// the region that began at the cursor; an unclosed begin tag runs to the
// end of the text.
func splitDNARegions(text, beginTag, endTag string) []dnaRegion {
	var regions []dnaRegion
	i := 0
	for i < len(text) {
		start := strings.Index(text[i:], beginTag)
		end := strings.Index(text[i:], endTag)
		if start >= 0 {
			start += i
		}
		if end >= 0 {
			end += i
		}
		switch {
		case start < 0 && end < 0:
			if remaining := text[i:]; remaining != "" {
				regions = append(regions, dnaRegion{text: remaining, dna: false})
			}
			return regions
		case start < 0:
			regions = append(regions, dnaRegion{text: text[i : end+len(endTag)], dna: true})
			i = end + len(endTag)
		case end < 0:
			if i < start {
				regions = append(regions, dnaRegion{text: text[i:start], dna: false})
			}
			regions = append(regions, dnaRegion{text: text[start:], dna: true})
			return regions
		case start < end:
			if i < start {
				regions = append(regions, dnaRegion{text: text[i:start], dna: false})
			}
			regions = append(regions, dnaRegion{text: text[start : end+len(endTag)], dna: true})
			i = end + len(endTag)
		default:
			regions = append(regions, dnaRegion{text: text[i : end+len(endTag)], dna: true})
			i = end + len(endTag)
		}
	}
	return regions
}

// parseDNARegion strips the region's tags and whitespace, reporting which
// tags were present so encode emits exactly the declared boundaries.
func parseDNARegion(region, beginTag, endTag string) (content string, hasStart, hasEnd bool) {
	if region == beginTag {
		return "", true, false
	}
	if region == endTag {
		return "", false, true
	}
	hasStart = strings.HasPrefix(region, beginTag)
	hasEnd = strings.HasSuffix(region, endTag)
	content = region
	if hasStart {
		content = content[len(beginTag):]
	}
	if hasEnd && strings.HasSuffix(content, endTag) {
		content = content[:len(content)-len(endTag)]
	}
	return strings.TrimSpace(content), hasStart, hasEnd
}

// processDNASequence chunks an uppercased sequence into k-mer tokens:
// windows step by k, any window off the extension alphabet becomes the
// out-of-vocabulary marker (returned as an empty string here), and a
// trailing remainder is right-padded with the first alphabet symbol --
// padding sits at the END so the per-position supervision mask stays a
// prefix, exactly as upstream.
func processDNASequence(sequence string, k int) []string {
	sequence = strings.ToUpper(sequence)
	kmers := make([]string, 0, len(sequence)/k+1)
	valid := func(window string) bool {
		for i := range len(window) {
			if strings.IndexByte(dnaAlphabet, window[i]) < 0 {
				return false
			}
		}
		return true
	}
	full := 0
	for ; full+k <= len(sequence); full += k {
		window := sequence[full : full+k]
		if valid(window) {
			kmers = append(kmers, window)
		} else {
			kmers = append(kmers, "")
		}
	}
	if remaining := sequence[full:]; remaining != "" {
		padded := remaining + strings.Repeat(dnaAlphabet[:1], k-len(remaining))
		if valid(padded) {
			kmers = append(kmers, padded)
		} else {
			kmers = append(kmers, "")
		}
	}
	return kmers
}

// specialID resolves one declared special token by declaration position.
func (d *dnaExtension) specialID(position int) (TokenID, bool) {
	if position < 0 || position >= len(d.specialTokens) {
		return NullToken, false
	}
	return TokenID(d.start + uint32(position)), true
}

// kmerID maps a clean k-mer to its extension ID: base-4 digits over ATCG,
// most significant base first -- the exact inverse of piece().
func (d *dnaExtension) kmerID(kmer string) (TokenID, bool) {
	if uint32(len(kmer)) != d.k {
		return NullToken, false
	}
	index := uint64(0)
	for i := range len(kmer) {
		digit := strings.IndexByte(dnaAlphabet, kmer[i])
		if digit < 0 {
			return NullToken, false
		}
		index = index*4 + uint64(digit)
	}
	return TokenID(uint64(d.start) + uint64(len(d.specialTokens)) + index), true
}

// encodeWithDNA is the hybrid encode: tagged regions become tag and k-mer
// IDs from the extension range, everything else takes the base tokenizer
// path. With auto-tags declared, untagged input is wrapped whole, matching
// the upstream default for sequence-only deployments. All tags, IDs and the
// chunk width come from the declared extension -- no model is named here.
func (v *Vocab) encodeWithDNA(text string, options EncodeOptions) ([]TokenID, error) {
	if len(v.dna.specialTokens) < dnaSpecialCount {
		return nil, fmt.Errorf("tokenizer: sequence extension declares %d special tokens, need %d (begin, end, out-of-vocabulary)",
			len(v.dna.specialTokens), dnaSpecialCount)
	}
	beginTag, endTag := v.dna.specialTokens[dnaSpecialBegin], v.dna.specialTokens[dnaSpecialEnd]
	begin, _ := v.dna.specialID(dnaSpecialBegin)
	end, _ := v.dna.specialID(dnaSpecialEnd)
	oov, _ := v.dna.specialID(dnaSpecialOOV)
	if v.dna.autoTags && !strings.Contains(text, beginTag) {
		text = beginTag + text + endTag
	}
	output := make([]TokenID, 0, len(text)/int(v.dna.k)+2)
	if options.AddSpecial && (v.Model == "bert" || v.AddBOS) {
		output = append(output, v.BOS)
	}
	for _, region := range splitDNARegions(text, beginTag, endTag) {
		if !region.dna {
			ids, err := v.Encode(region.text, EncodeOptions{ParseSpecial: options.ParseSpecial})
			if err != nil {
				return nil, err
			}
			output = append(output, ids...)
			continue
		}
		content, hasStart, hasEnd := parseDNARegion(region.text, beginTag, endTag)
		if hasStart {
			output = append(output, begin)
		}
		for _, kmer := range processDNASequence(content, int(v.dna.k)) {
			if kmer == "" {
				output = append(output, oov)
				continue
			}
			id, ok := v.dna.kmerID(kmer)
			if !ok {
				return nil, fmt.Errorf("tokenizer: k-mer %q escaped validation", kmer)
			}
			output = append(output, id)
		}
		if hasEnd {
			output = append(output, end)
		}
	}
	if options.AddSpecial {
		if v.Model == "bert" {
			output = append(output, v.SEP)
		} else if v.AddEOS {
			output = append(output, v.EOS)
		}
	}
	return output, nil
}
