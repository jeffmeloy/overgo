package tokenizer

import (
	"errors"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

const escapedWordPieceSpace = "▁"

func (v *Vocab) encodeWPM(text string) ([]TokenID, error) {
	if v.UNK == NullToken {
		return nil, errors.New("tokenizer: WPM vocabulary has no unknown token")
	}
	words := v.preprocessWPM(text)
	output := make([]TokenID, 0, len(words)*2)
	for _, word := range words {
		word = escapedWordPieceSpace + word
		start := len(output)
		for offset := 0; offset < len(word); {
			maximum := min(len(word), offset+v.maxTokenLen)
			matched := false
			for end := maximum; end > offset; end-- {
				id, ok := v.tokenToID[word[offset:end]]
				if !ok {
					continue
				}
				output = append(output, id)
				offset = end
				matched = true
				break
			}
			if matched {
				continue
			}
			output = output[:start]
			output = append(output, v.UNK)
			break
		}
	}
	return output, nil
}

func (v *Vocab) preprocessWPM(text string) []string {
	if v.StripAccents {
		text = norm.NFD.String(text)
	}
	words := make([]string, 0, len(text)/4+1)
	var current strings.Builder
	flush := func() {
		if current.Len() == 0 {
			return
		}
		words = append(words, current.String())
		current.Reset()
	}
	for _, character := range text {
		if unicode.IsSpace(character) {
			flush()
			continue
		}
		if character == 0 ||
			character == unicode.ReplacementChar ||
			unicode.Is(unicode.C, character) {
			continue
		}
		if v.StripAccents && unicode.Is(unicode.M, character) {
			continue
		}
		if v.Lowercase {
			character = unicode.ToLower(character)
		}
		if unicode.IsPunct(character) ||
			(character < 0x7f && unicode.IsSymbol(character)) ||
			isWPMChinese(character) {
			flush()
			words = append(words, string(character))
			continue
		}
		current.WriteRune(character)
	}
	flush()
	return words
}

func isWPMChinese(character rune) bool {
	return (character >= 0x04e00 && character <= 0x09fff) ||
		(character >= 0x03400 && character <= 0x04dbf) ||
		(character >= 0x20000 && character <= 0x2a6df) ||
		(character >= 0x2a700 && character <= 0x2b73f) ||
		(character >= 0x2b740 && character <= 0x2b81f) ||
		(character >= 0x2b920 && character <= 0x2ceaf) ||
		(character >= 0x0f900 && character <= 0x0faff) ||
		(character >= 0x2f800 && character <= 0x2fa1f)
}

func cleanWPMSpaces(text string) string {
	first := make([]byte, 0, len(text))
	for index, character := range []byte(text) {
		if index > 0 &&
			text[index-1] == ' ' &&
			(character == '?' || character == '!' ||
				character == '.' || character == ',') &&
			len(first) > 0 {
			first = first[:len(first)-1]
		}
		first = append(first, character)
	}
	second := make([]byte, 0, len(first))
	for index := 0; index < len(first); index++ {
		character := first[index]
		if character == '\'' &&
			index > 0 &&
			index+1 < len(first) &&
			first[index-1] == ' ' &&
			first[index+1] == ' ' {
			if len(second) > 0 {
				second = second[:len(second)-1]
			}
			second = append(second, character)
			index++
			continue
		}
		second = append(second, character)
	}
	third := make([]byte, 0, len(second))
	for index, character := range second {
		if character == '\'' && len(third) > 0 && third[len(third)-1] == ' ' &&
			index+1 < len(second) {
			remove := second[index+1] == 's' || second[index+1] == 'm'
			if index+2 < len(second) {
				remove = remove ||
					(second[index+1] == 'r' && second[index+2] == 'e') ||
					(second[index+1] == 'v' && second[index+2] == 'e')
			}
			if remove {
				third = third[:len(third)-1]
			}
		}
		third = append(third, character)
	}
	return string(third)
}
