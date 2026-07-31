package tokenizer

import (
	"reflect"
	"testing"
)

func TestWPMNormalizationGreedySegmentationAndSpecials(t *testing.T) {
	texts := []string{"[UNK]", "[CLS]", "[SEP]", "▁ap", "fe", "l", "▁中", "▁!"}
	vocab := &Vocab{
		Model:        "bert",
		Tokens:       make([]Token, len(texts)),
		BOS:          1,
		SEP:          2,
		UNK:          0,
		Lowercase:    true,
		StripAccents: true,
		tokenToID:    make(map[string]TokenID, len(texts)),
	}
	for index, text := range texts {
		vocab.Tokens[index] = Token{Text: text, Type: TokenNormal}
		vocab.tokenToID[text] = TokenID(index)
		vocab.maxTokenLen = max(vocab.maxTokenLen, len(text))
	}
	vocab.Tokens[0].Type = TokenUnknown
	vocab.Tokens[1].Type = TokenControl
	vocab.Tokens[2].Type = TokenControl

	got, err := vocab.Encode("Äpfel 中!", EncodeOptions{AddSpecial: true})
	if err != nil {
		t.Fatal(err)
	}
	want := []TokenID{1, 3, 4, 5, 6, 7, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("WPM IDs = %v, want %v", got, want)
	}
	unknown, err := vocab.Encode("unrepresented", EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(unknown, []TokenID{0}) {
		t.Fatalf("unknown IDs = %v", unknown)
	}
}

func TestWPMDecodeCleanupMatchesPinnedPasses(t *testing.T) {
	for input, want := range map[string]string{
		" hello , world !": " hello, world!",
		" rock ' roll":     " rock'roll",
		" it 's done":      " it's done",
		" we 're done":     " we're done",
		" don 't":          " don 't",
		" we 'll":          " we 'll",
	} {
		if got := cleanWPMSpaces(input); got != want {
			t.Fatalf("cleanWPMSpaces(%q) = %q, want %q", input, got, want)
		}
	}
}
