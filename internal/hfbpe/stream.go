package hfbpe

import (
	"errors"
	"slices"
	"strings"
	"unicode/utf8"
)

// DecodeState retains at most one incomplete UTF-8 rune and whether the
// declared initial-space policy has been applied. Its owner binds tokenizer
// identity and finalization; it never retains preceding tokens or text.
type DecodeState struct {
	Pending []byte `json:"pending,omitempty"`
	Started bool   `json:"started"`
}

// DecodeTextChunk emits a strict UTF-8 text suffix using the same vocabulary,
// special-token and space rules as DecodeText. Finalization refuses unfinished
// UTF-8; errors leave the caller's previous state unchanged.
func (t *Tokenizer) DecodeTextChunk(ids []int, previous DecodeState, final bool) (string, DecodeState, error) {
	if t == nil || len(previous.Pending) >= utf8.UTFMax ||
		len(previous.Pending) != 0 && (!previous.Started || utf8.FullRune(previous.Pending)) {
		return "", DecodeState{}, errors.New("tokenizer: invalid incremental decode state")
	}
	data, err := t.decodeBytes(ids, true, true)
	if err != nil {
		return "", DecodeState{}, err
	}
	bytes := data
	if len(previous.Pending) != 0 {
		bytes = append(slices.Clone(previous.Pending), data...)
	}
	end := 0
	for end < len(bytes) && utf8.FullRune(bytes[end:]) {
		r, size := utf8.DecodeRune(bytes[end:])
		if r == utf8.RuneError && size == 1 {
			return "", DecodeState{}, errors.New("decoded token sequence is not UTF-8")
		}
		end += size
	}
	if final && end != len(bytes) {
		return "", DecodeState{}, errors.New("tokenizer: incomplete final UTF-8 sequence")
	}
	text := string(bytes[:end])
	if t.stripDecodePrefix && !previous.Started {
		text = strings.TrimPrefix(text, " ")
	}
	return text, DecodeState{Pending: slices.Clone(bytes[end:]), Started: previous.Started || len(bytes) != 0}, nil
}
