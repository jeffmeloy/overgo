// Package hfbpe is a Hugging Face tokenizer.json BPE encoder/decoder:
// specials -> GPT-2 regex split -> byte alphabet -> BPE -> ids (byte-level
// scheme), or specials -> U+2581 marker -> BPE -> byte fallback (the
// sentencepiece-marker scheme, selected by the DECLARED normalizer).
//
// Ported from adaptive_new go/extmodel tokenizer.go (BPE paths).
package hfbpe

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"overgo/internal/binaryschema"
	"overgo/internal/jsonfile"
)

type Tokenizer struct {
	vocab        map[string]int
	mergeRank    map[string]int
	special      map[string]int
	specials     []string // longest-first
	b2u          [binaryschema.ByteValueCount]rune
	u2bDense     [2 * binaryschema.ByteValueCount]int16
	id2tok       map[int]string
	spaceMarker  string
	byteFallback bool
}

type tokenizerJSON struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	Normalizer json.RawMessage `json:"normalizer"`
	Model      struct {
		Type         string            `json:"type"`
		Vocab        map[string]int    `json:"vocab"`
		Merges       []json.RawMessage `json:"merges"`
		ByteFallback bool              `json:"byte_fallback"`
	} `json:"model"`
}

// Load: reads tokenizer.json from a model dir and builds the encoder.
func Load(dir string) (*Tokenizer, error) {
	var tj tokenizerJSON
	if err := jsonfile.Decode(filepath.Join(dir, "tokenizer.json"), &tj); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}
	if !strings.EqualFold(tj.Model.Type, "BPE") {
		return nil, fmt.Errorf("tokenizer model type %q unsupported (only BPE)", tj.Model.Type)
	}
	t := &Tokenizer{
		vocab:        tj.Model.Vocab,
		mergeRank:    make(map[string]int, len(tj.Model.Merges)),
		special:      map[string]int{},
		byteFallback: tj.Model.ByteFallback,
	}
	if len(tj.Normalizer) > 0 {
		var norm struct {
			Type    string `json:"type"`
			Pattern struct {
				String string `json:"String"`
			} `json:"pattern"`
			Content string `json:"content"`
		}
		if json.Unmarshal(tj.Normalizer, &norm) == nil &&
			norm.Type == "Replace" && norm.Pattern.String == " " && norm.Content != "" {
			t.spaceMarker = norm.Content
		}
	}
	for i, raw := range tj.Model.Merges {
		var pair [2]string
		if err := json.Unmarshal(raw, &pair); err == nil {
			t.mergeRank[pair[0]+" "+pair[1]] = i
			continue
		}
		var pairText string
		if err := json.Unmarshal(raw, &pairText); err == nil {
			t.mergeRank[pairText] = i
			continue
		}
		return nil, fmt.Errorf("merges[%d]: unrecognized wire format", i)
	}
	for _, a := range tj.AddedTokens {
		t.special[a.Content] = a.ID
		t.specials = append(t.specials, a.Content)
	}
	sort.Slice(t.specials, func(i, j int) bool { return len(t.specials[i]) > len(t.specials[j]) })
	t.buildByteAlphabet()
	return t, nil
}

// buildByteAlphabet: the GPT-2 reversible byte<->unicode map.
func (t *Tokenizer) buildByteAlphabet() {
	var bs []int
	for i := '!'; i <= '~'; i++ {
		bs = append(bs, int(i))
	}
	for i := 0xA1; i <= 0xAC; i++ {
		bs = append(bs, i)
	}
	for i := 0xAE; i <= 0xFF; i++ {
		bs = append(bs, i)
	}
	inSet := map[int]bool{}
	for _, b := range bs {
		inSet[b] = true
	}
	n := 0
	cs := make([]rune, binaryschema.ByteValueCount)
	for b := 0; b < binaryschema.ByteValueCount; b++ {
		if inSet[b] {
			cs[b] = rune(b)
		} else {
			cs[b] = rune(binaryschema.ByteValueCount + n)
			n++
		}
	}
	for i := range t.u2bDense {
		t.u2bDense[i] = -1
	}
	for b := 0; b < binaryschema.ByteValueCount; b++ {
		t.b2u[b] = cs[b]
		t.u2bDense[cs[b]] = int16(b)
	}
}

// Encode: text -> token ids under the declared scheme.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	var ids []int
	for _, seg := range t.splitOnSpecials(text) {
		if id, ok := t.special[seg]; ok {
			ids = append(ids, id)
			continue
		}
		if t.spaceMarker != "" {
			segIDs, err := t.encodeSentencepiece(seg)
			if err != nil {
				return nil, err
			}
			ids = append(ids, segIDs...)
			continue
		}
		for _, piece := range gpt2Pretokenize(seg) {
			var sb strings.Builder
			for i := 0; i < len(piece); i++ {
				sb.WriteRune(t.b2u[piece[i]])
			}
			for _, tok := range t.bpe(sb.String()) {
				id, ok := t.vocab[tok]
				if !ok {
					return nil, fmt.Errorf("token %q not in vocab", tok)
				}
				ids = append(ids, id)
			}
		}
	}
	return ids, nil
}

func (t *Tokenizer) encodeSentencepiece(seg string) ([]int, error) {
	var ids []int
	for _, tok := range t.bpe(strings.ReplaceAll(seg, " ", t.spaceMarker)) {
		if id, ok := t.vocab[tok]; ok {
			ids = append(ids, id)
			continue
		}
		if !t.byteFallback {
			return nil, fmt.Errorf("token %q not in vocab (no byte fallback)", tok)
		}
		for i := 0; i < len(tok); i++ {
			bt := fmt.Sprintf("<0x%02X>", tok[i])
			id, ok := t.vocab[bt]
			if !ok {
				return nil, fmt.Errorf("byte-fallback token %s not in vocab", bt)
			}
			ids = append(ids, id)
		}
	}
	return ids, nil
}

func (t *Tokenizer) ensureID2Vocab() {
	if t.id2tok != nil {
		return
	}
	t.id2tok = make(map[int]string, len(t.vocab))
	for tok, id := range t.vocab {
		t.id2tok[id] = tok
	}
	for sp, id := range t.special {
		t.id2tok[id] = sp
	}
}

// Decode: token ids -> UTF-8 string (specials decode to their literals).
func (t *Tokenizer) Decode(ids []int) string {
	t.ensureID2Vocab()
	var bytes []byte
	for _, id := range ids {
		tok, ok := t.id2tok[id]
		if !ok {
			continue
		}
		if _, isSpecial := t.special[tok]; isSpecial {
			bytes = append(bytes, []byte(tok)...)
			continue
		}
		if t.spaceMarker != "" {
			if b, ok := parseByteFallbackToken(tok); ok {
				bytes = append(bytes, b)
				continue
			}
			bytes = append(bytes, []byte(strings.ReplaceAll(tok, t.spaceMarker, " "))...)
			continue
		}
		for _, r := range tok {
			if r >= 0 && int(r) < len(t.u2bDense) {
				if b := t.u2bDense[r]; b >= 0 {
					bytes = append(bytes, byte(b))
				}
			}
		}
	}
	return string(bytes)
}

func parseByteFallbackToken(tok string) (byte, bool) {
	if len(tok) != 6 || tok[0] != '<' || tok[1] != '0' || tok[2] != 'x' || tok[5] != '>' {
		return 0, false
	}
	hex := func(c byte) (byte, bool) {
		switch {
		case c >= '0' && c <= '9':
			return c - '0', true
		case c >= 'A' && c <= 'F':
			return c - 'A' + 10, true
		case c >= 'a' && c <= 'f':
			return c - 'a' + 10, true
		}
		return 0, false
	}
	hi, ok1 := hex(tok[3])
	lo, ok2 := hex(tok[4])
	if !ok1 || !ok2 {
		return 0, false
	}
	return hi<<4 | lo, true
}

// SpecialID: resolves a special-token literal to its id.
func (t *Tokenizer) SpecialID(lit string) (int, bool) {
	id, ok := t.special[lit]
	return id, ok
}

func (t *Tokenizer) splitOnSpecials(text string) []string {
	segs := []string{text}
	for _, sp := range t.specials {
		var next []string
		for _, s := range segs {
			if _, isSpecial := t.special[s]; isSpecial {
				next = append(next, s)
				continue
			}
			for {
				idx := strings.Index(s, sp)
				if idx < 0 {
					if s != "" {
						next = append(next, s)
					}
					break
				}
				if idx > 0 {
					next = append(next, s[:idx])
				}
				next = append(next, sp)
				s = s[idx+len(sp):]
			}
		}
		segs = next
	}
	return segs
}

func (t *Tokenizer) bpe(word string) []string {
	parts := strings.Split(word, "")
	if len(parts) < 2 {
		if word == "" {
			return nil
		}
		return parts
	}
	for {
		bestRank := -1
		bestPos := -1
		for i := 0; i+1 < len(parts); i++ {
			if r, ok := t.mergeRank[parts[i]+" "+parts[i+1]]; ok {
				if bestRank == -1 || r < bestRank {
					bestRank = r
					bestPos = i
				}
			}
		}
		if bestPos < 0 {
			break
		}
		merged := parts[bestPos] + parts[bestPos+1]
		parts = append(parts[:bestPos], append([]string{merged}, parts[bestPos+2:]...)...)
	}
	return parts
}

// gpt2Pretokenize: hand-written scanner reproducing the GPT-2/Qwen2 regex
// alternation (Go RE2 cannot express the trailing-space lookahead).
func gpt2Pretokenize(s string) []string {
	runes := []rune(s)
	var out []string
	i := 0
	n := len(runes)
	isL := func(r rune) bool { return unicode.IsLetter(r) }
	isN := func(r rune) bool { return unicode.IsNumber(r) }
	isSpace := func(r rune) bool { return unicode.IsSpace(r) }
	for i < n {
		if runes[i] == '\'' && i+1 < n {
			if m := matchContraction(runes, i); m > 0 {
				out = append(out, string(runes[i:i+m]))
				i += m
				continue
			}
		}
		if j := matchOptPrefixThenClass(runes, i, isL); j > i {
			out = append(out, string(runes[i:j]))
			i = j
			continue
		}
		if isN(runes[i]) {
			out = append(out, string(runes[i:i+1]))
			i++
			continue
		}
		if j := matchPunct(runes, i, isL, isN, isSpace); j > i {
			out = append(out, string(runes[i:j]))
			i = j
			continue
		}
		if isSpace(runes[i]) {
			j := i
			for j < n && isSpace(runes[j]) {
				j++
			}
			if j < n && j-1 > i {
				nxt := runes[j]
				if isL(nxt) || (!isSpace(nxt) && !isN(nxt)) {
					out = append(out, string(runes[i:j-1]))
					i = j - 1
					continue
				}
			}
			out = append(out, string(runes[i:j]))
			i = j
			continue
		}
		out = append(out, string(runes[i:i+1]))
		i++
	}
	return out
}

func matchContraction(r []rune, i int) int {
	rest := strings.ToLower(string(r[i:min(i+3, len(r))]))
	for _, c := range []string{"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"} {
		if strings.HasPrefix(rest, c) {
			return len(c)
		}
	}
	return 0
}

func matchOptPrefixThenClass(r []rune, i int, cls func(rune) bool) int {
	n := len(r)
	j := i
	if j < n && r[j] != '\r' && r[j] != '\n' && !unicode.IsLetter(r[j]) && !unicode.IsNumber(r[j]) {
		if j+1 < n && cls(r[j+1]) {
			j++
		} else {
			return i
		}
	}
	if j >= n || !cls(r[j]) {
		return i
	}
	for j < n && cls(r[j]) {
		j++
	}
	return j
}

func matchPunct(r []rune, i int, isL, isN, isSpace func(rune) bool) int {
	n := len(r)
	j := i
	start := i
	if j < n && r[j] == ' ' {
		if j+1 < n && !isSpace(r[j+1]) && !isL(r[j+1]) && !isN(r[j+1]) {
			j++
		} else {
			return i
		}
	}
	k := j
	for k < n && !isSpace(r[k]) && !isL(r[k]) && !isN(r[k]) {
		k++
	}
	if k == j {
		return start
	}
	for k < n && (r[k] == '\r' || r[k] == '\n') {
		k++
	}
	return k
}
