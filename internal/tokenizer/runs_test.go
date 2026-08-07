package tokenizer

import (
	"slices"
	"strings"
	"testing"
)

const runFixturePlaceholder = "<media>"

func runFixtureEncoder(text string, options EncodeOptions) ([]TokenID, error) {
	var ids []TokenID
	if options.AddSpecial {
		ids = append(ids, 1)
	}
	for len(text) > 0 {
		if options.ParseSpecial && strings.HasPrefix(text, runFixturePlaceholder) {
			ids = append(ids, 9)
			text = text[len(runFixturePlaceholder):]
			continue
		}
		ids = append(ids, TokenID(text[0])+16)
		text = text[1:]
	}
	return ids, nil
}

func TestEncodeRunsAssemblesPlaceholderTokens(t *testing.T) {
	text := "a" + strings.Repeat(runFixturePlaceholder, 2) + "bc" + runFixturePlaceholder + "d"
	ids, starts, err := EncodeRuns(
		runFixtureEncoder,
		text,
		runFixturePlaceholder,
		[]int{2, 1},
		EncodeOptions{AddSpecial: true, ParseSpecial: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []TokenID{1, TokenID('a') + 16, 9, 9, TokenID('b') + 16, TokenID('c') + 16, 9, TokenID('d') + 16}
	if !slices.Equal(ids, want) || !slices.Equal(starts, []int{2, 6}) {
		t.Fatalf("runs = %v at %v, want %v at [2 6]", ids, starts, want)
	}
}

func TestEncodeRunsRejectsUnexpectedPlaceholder(t *testing.T) {
	text := strings.Repeat(runFixturePlaceholder, 2)
	if _, _, err := EncodeRuns(
		runFixtureEncoder,
		text,
		runFixturePlaceholder,
		[]int{1},
		EncodeOptions{ParseSpecial: true},
	); err == nil {
		t.Fatal("expected extra placeholder rejection")
	}
}

func TestEncodeMarkersExpandsCompactRuns(t *testing.T) {
	ids, starts, err := EncodeMarkers(
		runFixtureEncoder,
		"a"+runFixturePlaceholder+"bc"+runFixturePlaceholder+"d",
		runFixturePlaceholder,
		[]int{2, 1},
		EncodeOptions{AddSpecial: true, ParseSpecial: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []TokenID{1, TokenID('a') + 16, 9, 9, TokenID('b') + 16, TokenID('c') + 16, 9, TokenID('d') + 16}
	if !slices.Equal(ids, want) || !slices.Equal(starts, []int{2, 6}) {
		t.Fatalf("markers = %v at %v, want %v at [2 6]", ids, starts, want)
	}
}
