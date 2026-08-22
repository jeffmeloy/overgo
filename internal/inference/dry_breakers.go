package inference

import (
	"fmt"
	"strings"

	"overgo/internal/tensor"
	"overgo/internal/textcheck"
	"overgo/internal/tokenizer"
)

// TokenizeDryBreakers ports llama.cpp's overlapping-token expansion; finds
// both tokens containing complete breaker and token sequences where
// breaker: starts in one token and continues in later tokens
func (r *Runner) TokenizeDryBreakers(breakers []string) ([][]int, error) {
	if r == nil || r.vocab == nil {
		return nil, errRunnerNil
	}
	var result [][]int
	seen := make(map[string]struct{})
	add := func(sequence []tokenizer.TokenID) {
		encoded := make([]int, len(sequence))
		var key strings.Builder
		for index, id := range sequence {
			encoded[index] = int(id)
			fmt.Fprintf(&key, "%d,", id)
		}
		if _, exists := seen[key.String()]; exists {
			return
		}
		seen[key.String()] = struct{}{}
		result = append(result, encoded)
	}

	for breakerIndex, breaker := range breakers {
		if breaker == "" {
			return nil, fmt.Errorf("inference: DRY breaker %d is empty", breakerIndex)
		}
		for tokenIndex := range r.vocab.Tokens {
			tokenID := tokenizer.TokenID(tokenIndex)
			piece, err := r.vocab.DecodePiece(tokenID, true)
			if err != nil {
				return nil, err
			}
			if strings.Contains(piece, breaker) {
				add([]tokenizer.TokenID{tokenID})
				continue
			}
			for position := strings.IndexByte(piece, breaker[tensor.FirstOffset]); textcheck.FoundIndex(position); {
				matched := tensor.SingletonExtent
				for matched < len(breaker) &&
					position+matched < len(piece) &&
					piece[position+matched] == breaker[matched] {
					matched++
				}
				if position+matched == len(piece) && matched < len(breaker) {
					tail, encodeErr := r.vocab.Encode(
						breaker[matched:],
						tokenizer.EncodeOptions{},
					)
					if encodeErr != nil {
						return nil, encodeErr
					}
					sequence := append([]tokenizer.TokenID{tokenID}, tail...)
					add(sequence)
				}
				next := strings.IndexByte(piece[position+tensor.SingletonExtent:], breaker[tensor.FirstOffset])
				if !textcheck.FoundIndex(next) {
					break
				}
				position += next + tensor.SingletonExtent
			}
		}
	}
	return result, nil
}
