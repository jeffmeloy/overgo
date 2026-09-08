package hfbpe

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestDecodeTextChunkUTF8Restart(t *testing.T) {
	tokenizer := &Tokenizer{id2tok: make(map[int]string)}
	tokenizer.buildByteAlphabet()
	for b, r := range tokenizer.b2u {
		tokenizer.id2tok[b] = string(r)
	}
	want := "Aé漢🙂 Z"
	ids := make([]int, len(want))
	for i, b := range []byte(want) {
		ids[i] = int(b)
	}
	for size := 1; size <= len(ids); size++ {
		var state DecodeState
		var output strings.Builder
		for start := 0; start < len(ids); start += size {
			encoded, err := json.Marshal(state)
			if err != nil {
				t.Fatal(err)
			}
			var restored DecodeState
			if err := json.Unmarshal(encoded, &restored); err != nil {
				t.Fatal(err)
			}
			before := slices.Clone(restored.Pending)
			end := min(len(ids), start+size)
			text, next, err := tokenizer.DecodeTextChunk(ids[start:end], restored, end == len(ids))
			if err != nil || !utf8.ValidString(text) || len(next.Pending) >= utf8.UTFMax || !slices.Equal(before, restored.Pending) {
				t.Fatalf("chunk %d: invalid state, mutation or output: %v", size, err)
			}
			output.WriteString(text)
			state = next
		}
		if output.String() != want || len(state.Pending) != 0 {
			t.Fatalf("chunk %d output=%q", size, output.String())
		}
	}
	for _, bad := range [][]int{{256}, {0xff}, {0xe2}} {
		if _, _, err := tokenizer.DecodeTextChunk(bad, DecodeState{}, true); err == nil {
			t.Fatal("accepted unknown token or invalid final UTF-8")
		}
	}
	for _, bad := range []DecodeState{{Pending: []byte{0xe2}}, {Pending: []byte{0xff}, Started: true}, {Pending: []byte("text"), Started: true}} {
		if _, _, err := tokenizer.DecodeTextChunk(nil, bad, false); err == nil {
			t.Fatal("accepted invalid restart")
		}
	}
}
