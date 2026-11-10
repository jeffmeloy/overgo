package evaluation

import (
	"encoding/json"
	"fmt"
	"strings"
)

// UnreadCauses sorts the answers that named no option by why each named none (answer-format/unanswered), read from
// the text it generated: what overgo's request ended, and what the model chose.
type UnreadCauses struct {
	// Budget: the answer began, a value, a working or a sentence, and the token budget ended before it named a letter
	// (chatAnswerTokens); overgo's request.
	Budget uint64 `json:"budget,omitzero"`
	// Thinking: the answer opened a thought and never closed it; overgo's request, which asked for no thought.
	Thinking uint64 `json:"thinking,omitzero"`
	// Declined: the model said no option fits or none is given; the model's own.
	Declined uint64 `json:"declined,omitzero"`
	// Repeated: the answer repeats itself, a phrase three times over; the model's own decode.
	Repeated uint64 `json:"repeated,omitzero"`
}

// String says each cause's count in words, the request's first, empty where there are none.
func (causes UnreadCauses) String() string {
	var said []string
	for _, cause := range []struct {
		count uint64
		words string
	}{
		{causes.Budget, "ran out of the answer's tokens"}, {causes.Thinking, "left a thought open"},
		{causes.Declined, "said no option fits"}, {causes.Repeated, "repeated themselves"},
	} {
		if cause.count > 0 {
			said = append(said, fmt.Sprintf("%d %s", cause.count, cause.words))
		}
	}
	return strings.Join(said, ", ")
}

// declinedPhrases are what an answer says when the model holds that no option fits, each lowercase.
var declinedPhrases = []string{"not explicitly", "not listed", "not among", "none of the", "not provided", "not given",
	"cannot be determined", "can't be determined", "not possible to determine", "no correct"}

// repeatedPhrase is how many words make a phrase, and repeatedTimes how often one appears, for an answer to repeat
// itself.
const (
	repeatedPhrase = 3
	repeatedTimes  = 3
)

// add counts the text of one answer that named no option under its cause.
func (causes *UnreadCauses) add(raw string) {
	if strings.Contains(raw, "<think>") && !strings.Contains(raw, "</think>") {
		causes.Thinking++
		return
	}
	lower := strings.ToLower(raw)
	for _, phrase := range declinedPhrases {
		if strings.Contains(lower, phrase) {
			causes.Declined++
			return
		}
	}
	words := strings.Fields(lower)
	seen := map[string]int{}
	for at := 0; at+repeatedPhrase <= len(words); at++ {
		phrase := strings.Join(words[at:at+repeatedPhrase], " ")
		if seen[phrase]++; seen[phrase] == repeatedTimes {
			causes.Repeated++
			return
		}
	}
	causes.Budget++
}

// unreadCauses sorts, in a report's observations, the answers that named no option by their cause; zero where the
// report cannot be read or every answer named one.
func unreadCauses(report []byte) UnreadCauses {
	var answers struct {
		Observations []struct {
			Selected int    `json:"selected"`
			Raw      string `json:"raw"`
		} `json:"observations"`
	}
	var causes UnreadCauses
	if json.Unmarshal(report, &answers) != nil {
		return causes
	}
	for _, answer := range answers.Observations {
		if answer.Selected == unmatchedChoice {
			causes.add(answer.Raw)
		}
	}
	return causes
}
