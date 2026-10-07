package evaluation

import (
	"errors"
	"fmt"
	"slices"
)

const (
	wikitextPassagesSource = "wikitext-2-raw-v1/test"
	wikitextPassagesSchema = "overgo/wikitext-2-raw-v1-test-passages/v1"
	// wikitextPassages passages of about wikitextPassageCharacters characters
	// are what a held-out likelihood is scored on: enough text for a stable
	// perplexity, short enough that every model's context holds one.
	wikitextPassages          = 64
	wikitextPassageCharacters = 2000
)

// BuildWikiTextPassages builds the held-out likelihood suite of the
// WikiText-2 test split from the lines of its text column, in row order:
// passages at evenly spaced offsets through the whole split, each starting on
// a line and ending on a word, so the suite spans every article of the split
// and no two passages overlap. Offsets and lengths count characters, not
// bytes, as the split's text is not all ASCII. It returns the suite and the
// number of its cases.
func BuildWikiTextPassages(lines []string) (any, int, error) {
	var text []rune
	for _, line := range lines {
		text = append(text, []rune(line)...)
	}
	suite := sequenceScoringSuite{Kind: sequenceScoringKind, Schema: wikitextPassagesSchema, Source: wikitextPassagesSource}
	end := 0
	for index := range wikitextPassages {
		start := index * len(text) / wikitextPassages
		if start > 0 {
			line := slices.Index(text[start:], '\n')
			if line < 0 {
				return nil, 0, errors.New("evaluation: the split ends before a passage's line does")
			}
			start += line + 1
		}
		if start < end {
			return nil, 0, fmt.Errorf("evaluation: the split is too short for %d passages of %d characters that do not overlap", wikitextPassages, wikitextPassageCharacters)
		}
		passage := text[start:min(len(text), start+wikitextPassageCharacters)]
		word := len(passage) - 1
		for word >= 0 && passage[word] != ' ' {
			word--
		}
		if word <= 0 {
			return nil, 0, errors.New("evaluation: a passage holds no word boundary to end on")
		}
		end = start + word
		suite.Cases = append(suite.Cases, sequenceScoringCase{
			Name: fmt.Sprintf("%s/%02d", wikitextPassagesSource, index), Group: wikitextPassagesSource, Text: string(passage[:word]),
		})
	}
	if _, err := compileSequenceScoring(suite); err != nil {
		return nil, 0, err
	}
	return suite, len(suite.Cases), nil
}
