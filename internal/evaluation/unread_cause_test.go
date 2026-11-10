package evaluation

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAnAnswerNamingNoOptionIsSortedByItsCause sorts the MMLU-Pro answers that named no option by what their record
// shows (answer-format/unanswered, SQA-362), on texts the models wrote under the chat-template protocol on 2026-11-07:
// Qwen3.5-9B and both gemma-4-12B-it builds began working the problem ("calculated as follows") and MiniCPM5-1B gave
// the value ("11.9, which corresponds to"), each recorded as having written the budget's tokens; gemma also said no
// option fits. A thought left open and a phrase repeated three times are their own indicators. The same working with
// no record of how its generation ended, an answer that ended its turn with no text, and one a report written before
// generations kept how they ended holds are of no cause the record shows. An answer that named a letter, or was scored
// by likelihood, counts toward none, and what is no report sorts nothing.
func TestAnAnswerNamingNoOptionIsSortedByItsCause(t *testing.T) {
	t.Parallel()
	type observation struct {
		Selected int    `json:"selected"`
		Raw      string `json:"raw,omitempty"`
		Ended    string `json:"ended,omitempty"`
	}
	report := testcheck.Must(json.Marshal(map[string][]observation{"observations": {
		{Selected: unmatchedChoice, Raw: " calculated as follows:\n\n1. ", Ended: "budget"},
		{Selected: unmatchedChoice, Raw: " 11.9, which corresponds to", Ended: "budget"},
		{Selected: unmatchedChoice, Raw: " derived from the compound interest formula:\n", Ended: "budget"},
		{Selected: unmatchedChoice, Raw: " not explicitly listed among the options provided because", Ended: "turn"},
		{Selected: unmatchedChoice, Raw: "<think>\nLet me work through the interest", Ended: "budget"},
		{Selected: unmatchedChoice, Raw: "To find the find the find the degree of the degree of the degree of the", Ended: "budget"},
		{Selected: unmatchedChoice, Raw: " 11.9, which corresponds to", Ended: "turn"},
		{Selected: unmatchedChoice, Ended: "turn"},
		{Selected: unmatchedChoice, Raw: " calculated as follows:\n\n1. "},
		{Selected: 2, Raw: " C", Ended: "turn"},
		{Selected: 0},
	}}))(t)
	got := unreadCauses(report)
	if got.Budget != 3 || got.Thinking != 1 || got.Declined != 1 || got.Repeated != 1 || !strings.Contains(got.String(), "3 of a cause the record does not show") {
		t.Fatalf("the answers that named no option sort as %+v (%s), want three budget-ended by their record, one each of the indicators and three of no cause the record shows", got, got)
	}
	for _, other := range []string{`{"observations":[]}`, `{"accuracy":0.5}`, `not a report`} {
		if got := unreadCauses([]byte(other)); got != (UnreadCauses{}) {
			t.Errorf("%s sorts as %+v, want nothing", other, got)
		}
	}
}
