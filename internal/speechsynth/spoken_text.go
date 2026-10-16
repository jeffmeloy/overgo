package speechsynth

import (
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"
)

// The model's own code does two things to a text before and after it speaks
// it, and a served request did neither: the round-trip reading was of speech
// cut at the frame its end was first seen in. This file does them as that
// code does.

const (
	// shortTextWords is the most words a text holds for the model's own code
	// to call it short: a short text's end is seen early, and it is spoken on
	// for longer.
	shortTextWords = 4
	// shortTextGuess and longTextGuess are that code's guesses of the frames
	// to generate after the end of speech is first seen, for a short text and
	// for a longer one, and framesBeyondGuess what its generation adds to the
	// guess: five frames after a short text and three after a longer one,
	// each 80 milliseconds.
	shortTextGuess, longTextGuess, framesBeyondGuess = 3, 1, 2
)

// spokenText prepares a text as the model's own code prepares one it is about
// to speak, and gives the frames to generate after the end of speech is first
// seen. Line breaks become spaces and a doubled space one; the text opens
// with a capital; a text that ends in a letter or a digit is given a full
// stop, since the model ends a sentence it can see the end of. That code also
// pads a short text with spaces and then strips them before the model reads
// it, so no padding is written here.
func spokenText(text string, removeSemicolons bool) (string, int, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return "", 0, errors.New("speechsynth: a text to speak is not empty")
	}
	text = strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(text, "\n", " "), "\r", " "), "  ", " ")
	if removeSemicolons {
		text = strings.ReplaceAll(text, ";", ",")
	}
	frames := longTextGuess + framesBeyondGuess
	if len(strings.Fields(text)) <= shortTextWords {
		frames = shortTextGuess + framesBeyondGuess
	}
	if first, size := utf8.DecodeRuneInString(text); !unicode.IsUpper(first) {
		text = string(unicode.ToUpper(first)) + text[size:]
	}
	if last, _ := utf8.DecodeLastRuneInString(text); unicode.IsLetter(last) || unicode.IsNumber(last) {
		text += "."
	}
	return strings.TrimSpace(text), frames, nil
}
