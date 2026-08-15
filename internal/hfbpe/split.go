package hfbpe

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/jsonfile"
)

// LoadSplit: vocab.json + merges.txt + optional added_tokens.json.
func LoadSplit(dir string) (*Tokenizer, error) {
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
	reader, rank := bufio.NewReader(file), 0
	for {
		text, readErr := reader.ReadString('\n')
		line := strings.TrimSpace(text)
		if line == "" || strings.HasPrefix(line, "#") {
		} else if len(strings.Fields(line)) != 2 {
			return nil, fmt.Errorf("invalid BPE merge %q", line)
		} else {
			t.mergeRank[line] = rank
			rank++
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, readErr
		}
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
	sort.Slice(t.specials, func(i, j int) bool {
		left, right := t.specials[i], t.specials[j]
		return len(left) > len(right) || len(left) == len(right) && left < right
	})
	t.buildByteAlphabet()
	return t, nil
}
