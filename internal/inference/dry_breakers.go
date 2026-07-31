package inference

import (
	"errors"
	"fmt"
	"strings"

	"llamacpp2go/internal/tokenizer"
)

const (
	maxDryBreakerBytes = 40
	maxDryBreakerTail  = 20
	maxDryBreakers     = 1024
)

// TokenizeDryBreakers ports llama.cpp's overlapping-token expansion. It finds
// both tokens containing a complete breaker and token sequences where a
// breaker starts in one token and continues in later tokens.
func (r *Runner) TokenizeDryBreakers(breakers []string) ([][]int, error) {
	if r == nil || r.vocab == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if len(breakers) > maxDryBreakers {
		return nil, fmt.Errorf("inference: DRY breaker count exceeds %d", maxDryBreakers)
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
		if len(breaker) > maxDryBreakerBytes {
			breaker = breaker[:maxDryBreakerBytes]
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
			for position := strings.IndexByte(piece, breaker[0]); position >= 0; {
				matched := 1
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
					if len(tail) > maxDryBreakerTail {
						tail = tail[:maxDryBreakerTail]
					}
					sequence := make([]tokenizer.TokenID, 1, len(tail)+1)
					sequence[0] = tokenID
					sequence = append(sequence, tail...)
					add(sequence)
				}
				next := strings.IndexByte(piece[position+1:], breaker[0])
				if next < 0 {
					break
				}
				position += next + 1
			}
		}
	}
	return result, nil
}
