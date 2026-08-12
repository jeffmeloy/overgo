package hfbpe

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/jsonfile"
)

// LoadLegacy consumes the split-file Hugging Face byte-level BPE representation
// (vocab.json + merges.txt + optional added_tokens.json) when a model ships no
// tokenizer.json. SenseNova-U1 is Qwen2-family (byte-level, no normalizer, no
// byte fallback), so the resulting Tokenizer runs the same byte-level Encode
// path as Load. Ported from adaptive_new go/extmodel loadLegacyByteLevelTokenizer.
func LoadLegacy(dir string) (*Tokenizer, error) {
	var vocab map[string]int
	if err := jsonfile.Decode(filepath.Join(dir, "vocab.json"), &vocab); err != nil {
		return nil, fmt.Errorf("parse vocab.json: %w", err)
	}
	file, err := os.Open(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, err
	}
	defer file.Close()
	t := &Tokenizer{vocab: vocab, mergeRank: map[string]int{}, special: map[string]int{}}
	scanner, rank := bufio.NewScanner(file), 0
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if len(strings.Fields(line)) != 2 {
			return nil, fmt.Errorf("invalid BPE merge %q", line)
		}
		t.mergeRank[line] = rank
		rank++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	var added map[string]int
	err = jsonfile.Decode(filepath.Join(dir, "added_tokens.json"), &added)
	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("parse added_tokens.json: %w", err)
	}
	for token, id := range added {
		t.special[token] = id
		t.specials = append(t.specials, token)
	}
	sort.Slice(t.specials, func(i, j int) bool { return len(t.specials[i]) > len(t.specials[j]) })
	t.buildByteAlphabet()
	return t, nil
}
