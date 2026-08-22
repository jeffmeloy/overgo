package tokenizer

import (
	"errors"
	"math"
	"strings"
	"unicode/utf8"
)

type ugmBest struct {
	token TokenID
	from  int
	score float64
	set   bool
}

// encodeUGM implements SentencePiece's unigram Viterbi tokenizer used by T5
// GGUF profile handled here has no precompiled normalization character map
func (v *Vocab) encodeUGM(text string) ([]TokenID, error) {
	normalized := v.normalizeUGM(text)
	if normalized == "" {
		return nil, nil
	}
	if v.UNK == NullToken {
		return nil, errors.New("tokenizer: UGM vocabulary has no unknown token")
	}
	minScore := float32(math.Inf(1))
	for _, token := range v.Tokens {
		if token.Type == TokenNormal && token.Score < minScore {
			minScore = token.Score
		}
	}
	unknownScore := float64(minScore) - 10
	best := make([]ugmBest, len(normalized)+1)
	best[0] = ugmBest{score: 0, set: true}
	for offset := 0; offset < len(normalized); {
		_, runeSize := utf8.DecodeRuneInString(normalized[offset:])
		if runeSize == 0 {
			break
		}
		current := best[offset]
		singleFound := false
		if current.set {
			limit := min(len(normalized), offset+v.ugmMaxLen)
			for end := offset + runeSize; end <= limit; {
				if id, ok := v.tokenToID[normalized[offset:end]]; ok {
					token := v.Tokens[id]
					if token.Type == TokenNormal || token.Type == TokenUserDefined ||
						token.Type == TokenUnused {
						if end-offset == runeSize {
							singleFound = true
						}
						score := float64(token.Score)
						if token.Type == TokenUserDefined {
							score = 0
						}
						challenger := current.score + score
						if !best[end].set || challenger > best[end].score {
							best[end] = ugmBest{
								token: id, from: offset, score: challenger, set: true,
							}
						}
					}
				}
				if end == limit {
					break
				}
				_, size := utf8.DecodeRuneInString(normalized[end:])
				if size == 0 || end+size > limit {
					break
				}
				end += size
			}
			if !singleFound {
				end := offset + runeSize
				challenger := current.score + unknownScore
				if !best[end].set || challenger > best[end].score {
					best[end] = ugmBest{
						token: v.UNK, from: offset, score: challenger, set: true,
					}
				}
			}
		}
		offset += runeSize
	}
	if !best[len(normalized)].set {
		return nil, errors.New("tokenizer: UGM could not tokenize normalized input")
	}
	reversed := make([]TokenID, 0, utf8.RuneCountInString(normalized))
	previousUnknown := false
	for position := len(normalized); position > 0; {
		item := best[position]
		unknown := item.token == v.UNK
		if !(previousUnknown && unknown) {
			reversed = append(reversed, item.token)
		}
		previousUnknown = unknown
		position = item.from
	}
	output := make([]TokenID, len(reversed))
	for index := range reversed {
		output[len(reversed)-1-index] = reversed[index]
	}
	return output, nil
}

func (v *Vocab) normalizeUGM(text string) string {
	var result strings.Builder
	result.Grow(len(text) + len(sentencePieceSpaceMarker))
	spacePrepended := false
	processingNonWhitespace := false
	for _, value := range strings.ToValidUTF8(text, "\ufffd") {
		if value != ' ' {
			if !processingNonWhitespace {
				processingNonWhitespace = true
				if v.AddPrefix && !spacePrepended {
					result.WriteString(sentencePieceSpaceMarker)
					spacePrepended = true
				}
			}
			result.WriteRune(value)
		} else {
			processingNonWhitespace = false
			result.WriteString(sentencePieceSpaceMarker)
		}
	}
	return result.String()
}
