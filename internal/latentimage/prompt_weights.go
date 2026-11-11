package latentimage

import (
	"fmt"
	"strconv"
	"strings"

	"overgo/internal/checked"
)

// A prompt may weight a stretch of itself, written (text:weight): the attention value of each of its tokens is scaled
// by the weight in the denoiser's co-attention, and no other value changes (krea-pack/prompt-weighting). The idea, one
// sentence, is written here from that sentence alone, with a clean-room provenance: KJNodes' Krea and MOSS weighting
// patches are GPL-3.0 and are not read.

// weightedSpan is a stretch of a prompt and the weight its tokens' attention values are scaled by.
type weightedSpan struct {
	text   string
	weight float32
}

// unweighted is the weight of a token no span weights.
const unweighted = 1

// weightBits is the precision a weight is read at: the float32 the program scales by.
const weightBits = 32

// weightedSpans splits prompt into the stretches its parentheses weight and the rest, which weighs one; a prompt that
// weights nothing gives none, and is tokenized whole as it always was. A parenthesis that opens no (text:weight) is
// text.
func weightedSpans(prompt string) ([]weightedSpan, error) {
	var spans []weightedSpan
	var plain strings.Builder
	weighted := false
	for rest := prompt; rest != ""; {
		open := strings.IndexByte(rest, '(')
		if open < 0 {
			plain.WriteString(rest)
			break
		}
		closing := strings.IndexByte(rest[open:], ')')
		body := ""
		if closing > 0 {
			body = rest[open+1 : open+closing]
		}
		text, number, colon := strings.Cut(body, ":")
		weight, err := strconv.ParseFloat(strings.TrimSpace(number), weightBits)
		if closing <= 0 || !colon || err != nil || text == "" {
			plain.WriteString(rest[:open+1])
			rest = rest[open+1:]
			continue
		}
		if !checked.PositiveFinite32(float32(weight)) {
			return nil, fmt.Errorf("latent image: the prompt weights %q by %g, which is no positive finite number", text, weight)
		}
		plain.WriteString(rest[:open])
		if plain.Len() > 0 {
			spans = append(spans, weightedSpan{text: plain.String(), weight: unweighted})
			plain.Reset()
		}
		spans = append(spans, weightedSpan{text: text, weight: float32(weight)})
		weighted = true
		rest = rest[open+closing+1:]
	}
	if !weighted {
		return nil, nil
	}
	if plain.Len() > 0 {
		spans = append(spans, weightedSpan{text: plain.String(), weight: unweighted})
	}
	return spans, nil
}

// weightedTokens are a prompt's tokens in order and the weight of each.
type weightedTokens struct {
	ids     []int
	weights []float32
}

// spanTokens tokenizes each span through encode.
func spanTokens(spans []weightedSpan, encode func(string) ([]int, error)) (weightedTokens, error) {
	var tokens weightedTokens
	for _, span := range spans {
		encoded, err := encode(span.text)
		if err != nil {
			return weightedTokens{}, err
		}
		tokens.ids = append(tokens.ids, encoded...)
		for range encoded {
			tokens.weights = append(tokens.weights, span.weight)
		}
	}
	return tokens, nil
}
