package latentimage

import (
	"fmt"
	"slices"
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

// tokenizer is what a prompt's weights are aligned to its tokens with: its encoding, and the bytes each token spells.
type tokenizer interface {
	Encode(string) ([]int, error)
	TokenBytes([]int) ([][]byte, error)
}

// cleanText is the prompt spans spell with their markup gone: the text the model reads.
func cleanText(spans []weightedSpan) string {
	var text strings.Builder
	for _, span := range spans {
		text.WriteString(span.text)
	}
	return text.String()
}

// alignedTokens are the tokens of text, encoded whole, and the weight of each: the clean prompt the spans spell stands
// at byte start of text, every other byte weighing one, and a token takes the weight of the stretch that holds most of
// its bytes, the later on a tie. Tokenized a stretch at a time, markup at weight 1 changed the prompt's tokens where a
// stretch's edge split what the whole text merges (SQA-382). A text the tokens do not spell, a normalizing tokenizer's,
// is refused, as no byte of it can be weighed.
func alignedTokens(text string, start int, spans []weightedSpan, tok tokenizer) (weightedTokens, error) {
	ids, err := tok.Encode(text)
	if err != nil {
		return weightedTokens{}, err
	}
	pieces, err := tok.TokenBytes(ids)
	if err != nil {
		return weightedTokens{}, err
	}
	spelled := slices.Concat(pieces...)
	// A tokenizer that opens its text with a space marker spells one more byte before it.
	shift := len(spelled) - len(text)
	if shift < 0 || string(spelled[shift:]) != text {
		return weightedTokens{}, fmt.Errorf("latent image: the tokenizer's tokens do not spell the weighted prompt, so its weights cannot be placed")
	}
	byteWeight := make([]float32, len(spelled))
	for index := range byteWeight {
		byteWeight[index] = unweighted
	}
	at := shift + start
	for _, span := range spans {
		for index := range len(span.text) {
			byteWeight[at+index] = span.weight
		}
		at += len(span.text)
	}
	weights := make([]float32, len(ids))
	offset := 0
	for index, piece := range pieces {
		counts := map[float32]int{}
		chosen, most := float32(unweighted), 0
		for _, weight := range byteWeight[offset : offset+len(piece)] {
			if counts[weight]++; counts[weight] >= most {
				chosen, most = weight, counts[weight]
			}
		}
		weights[index] = chosen
		offset += len(piece)
	}
	return weightedTokens{ids: ids, weights: weights}, nil
}

// promptMark stands for the prompt in a chat template's rendering, so the text the template puts before and after it is
// found: no prompt or template holds it.
const promptMark = "\x00overgo prompt\x00"

// chatTokens are the tokens of a chat template's rendering of prompt (render) and the weight of each: a prompt that
// weights nothing is rendered and tokenized whole, as it always was; one that weights its stretches renders its clean
// text whole and its tokens are weighed where the template placed it (alignedTokens), found by rendering promptMark.
func chatTokens(prompt string, render func(string) (string, error), tok tokenizer) (weightedTokens, error) {
	spans, err := weightedSpans(prompt)
	if err != nil {
		return weightedTokens{}, err
	}
	if spans == nil {
		formatted, err := render(prompt)
		if err != nil {
			return weightedTokens{}, err
		}
		ids, err := tok.Encode(formatted)
		return weightedTokens{ids: ids}, err
	}
	marked, err := render(promptMark)
	if err != nil {
		return weightedTokens{}, err
	}
	before, after, found := strings.Cut(marked, promptMark)
	if !found {
		return weightedTokens{}, fmt.Errorf("latent image: the chat template's rendering holds no prompt to weigh")
	}
	clean := cleanText(spans)
	formatted, err := render(clean)
	if err != nil {
		return weightedTokens{}, err
	}
	if formatted != before+clean+after {
		return weightedTokens{}, fmt.Errorf("latent image: the chat template changes the prompt it renders, so its weights cannot be placed")
	}
	return alignedTokens(formatted, len(before), spans, tok)
}
