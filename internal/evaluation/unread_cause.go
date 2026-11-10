package evaluation

import (
	"encoding/json"
	"fmt"
	"strings"
)

// UnreadCauses sorts the answers that named no option by why each named none (answer-format/unanswered), from what
// the record shows (SQA-362): the text an answer holds, and how its generation ended where the report keeps it. A cause
// the record cannot show is unknown, never inferred.
type UnreadCauses struct {
	// Budget: the generation wrote as many tokens as its protocol bounds an answer to (generationBudget) before it named
	// a letter; overgo's request.
	Budget uint64 `json:"budget,omitzero"`
	// Thinking: the answer opened a thought and never closed it.
	Thinking uint64 `json:"thinking,omitzero"`
	// Declined: the answer says no option fits or none is given.
	Declined uint64 `json:"declined,omitzero"`
	// Repeated: the answer repeats a phrase three times over.
	Repeated uint64 `json:"repeated,omitzero"`
	// Unknown: the record shows none of these: the model ended its turn first, or a report written before generations
	// kept how they ended does not say.
	Unknown uint64 `json:"unknown,omitzero"`
}

// Total is how many answers named no option.
func (causes UnreadCauses) Total() uint64 {
	return causes.Budget + causes.Thinking + causes.Declined + causes.Repeated + causes.Unknown
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
		{causes.Unknown, "of a cause the record does not show"},
	} {
		if cause.count > 0 {
			said = append(said, fmt.Sprintf("%d %s", cause.count, cause.words))
		}
	}
	return strings.Join(said, ", ")
}

// How a generated answer's generation ended (choiceObservation.Ended): it wrote as many tokens as its protocol bounds an
// answer to, or the model ended it sooner.
const (
	generationBudget = "budget"
	generationTurn   = "turn"
)

// generationEnded says how a generation of generated tokens under a bound of bound ended.
func generationEnded(generated, bound int) string {
	if generated >= bound {
		return generationBudget
	}
	return generationTurn
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

// add counts one answer that named no option, of text raw whose generation ended as ended says, under its cause.
func (causes *UnreadCauses) add(raw, ended string) {
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
	if ended == generationBudget {
		causes.Budget++
		return
	}
	causes.Unknown++
}

// reportAnswer is what a report keeps of one case's answer that its readers read: its case, the option it selected,
// and, of a generated answer, its text and how its generation ended.
type reportAnswer struct {
	Name     string `json:"name"`
	Selected int    `json:"selected"`
	Raw      string `json:"raw"`
	Ended    string `json:"ended"`
}

// reportAnswers reads a report's answers once, for each reader of them; none where the report cannot be read.
func reportAnswers(report []byte) []reportAnswer {
	var answers struct {
		Observations []reportAnswer `json:"observations"`
	}
	if json.Unmarshal(report, &answers) != nil {
		return nil
	}
	return answers.Observations
}

// unreadCauses sorts, in a report's answers, those that named no option by their cause; zero where the report cannot
// be read or every answer named one. Its Total is how many named none.
func unreadCauses(report []byte) UnreadCauses {
	return sortUnread(reportAnswers(report))
}

// sortUnread sorts the answers that named no option by their cause.
func sortUnread(answers []reportAnswer) UnreadCauses {
	var causes UnreadCauses
	for _, answer := range answers {
		if answer.Selected == unmatchedChoice {
			causes.add(answer.Raw, answer.Ended)
		}
	}
	return causes
}
