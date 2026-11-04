package landing

import (
	"strings"
	"testing"
)

// formed holds requireSections to refusing each message whose refusal names the text given, and to passing each with
// none.
func formed(t *testing.T, messages map[string]string) {
	t.Helper()
	for message, refusal := range messages {
		err := requireSections([]byte(message))
		if refusal == "" && err != nil || refusal != "" && (err == nil || !strings.Contains(err.Error(), refusal)) {
			t.Errorf("message %q ended as %v, want a refusal naming %q (none when empty)", message, err, refusal)
		}
	}
}

// TestAMessageWithoutItsPredictionIsRefused refuses a message that says why the change was needed and not what it
// should do.
func TestAMessageWithoutItsPredictionIsRefused(t *testing.T) {
	t.Parallel()
	formed(t, map[string]string{
		"row/do: change\n\nCause: a fixture.\n":                                        predictedSection,
		"row/do: change\n\nCause: a fixture.\nPredicted effect: failure-rate holds.\n": "",
	})
}

// TestAnEmptySectionIsRefused refuses a section whose own line says nothing, whatever follows it.
func TestAnEmptySectionIsRefused(t *testing.T) {
	t.Parallel()
	formed(t, map[string]string{
		"row/do: change\n\nCause:\nPredicted effect: failure-rate holds.\n":                  "Cause:",
		"row/do: change\n\nCause: a fixture.\nPredicted effect:\nfailure-rate holds.\n":      predictedSection,
		"row/do: change\n\nCause: a fixture.\nPredicted effect: \n\nfailure-rate holds.\n":   predictedSection,
		"row/do: change\n\nCause: a fixture, read.\nPredicted effect: failure-rate holds.\n": "",
	})
}

// TestAPredictionNamesAMetric refuses a prediction that names no scorecard metric, which no scorecard reading could
// then be compared with.
func TestAPredictionNamesAMetric(t *testing.T) {
	t.Parallel()
	formed(t, map[string]string{
		"row/do: change\n\nCause: a fixture.\nPredicted effect: none.\n":                                           "scorecard metric",
		"row/do: change\n\nCause: a fixture.\nPredicted effect: it serves the loop.\n":                             "scorecard metric",
		"row/do: change\n\nCause: a fixture.\nPredicted effect: throughput rises.\n":                               "",
		"row/do: change\n\nCause: a fixture.\nPredicted effect: peak-device-memory falls, and setup-steps hold.\n": "",
	})
}

// TestAMessageTypesNoMeasuredFigure refuses a prediction that types a measured figure, by its unit or as a duration,
// which the gate writes itself; a count or a word with digits passes.
func TestAMessageTypesNoMeasuredFigure(t *testing.T) {
	t.Parallel()
	prediction := func(text string) string {
		return "row/do: change\n\nCause: a fixture.\nPredicted effect: " + text + "\n"
	}
	formed(t, map[string]string{
		prediction("first-output falls by 12ms."):              "12ms",
		prediction("throughput rises 40 %."):                   "40",
		prediction("peak-device-memory falls 1.5 GiB."):        "1.5",
		prediction("load-time falls to 2m45s."):                "2m45s",
		prediction("failure-rate holds, and 1,500 runs pass."): "",
		prediction("throughput holds on 64-bit hosts."):        "",
	})
}

// TestAMessageCarriesNoCoAuthorTrailer refuses a message with a line that credits a co-author, in any case.
func TestAMessageCarriesNoCoAuthorTrailer(t *testing.T) {
	t.Parallel()
	formed(t, map[string]string{
		"row/do: change\n\nCause: a fixture.\nPredicted effect: failure-rate holds.\n\nCo-Authored-By: A Model <model@example.invalid>\n": "co-author",
		"row/do: change\n\nCause: a fixture.\nPredicted effect: failure-rate holds.\n\nco-authored-by: someone\n":                         "co-author",
		"row/do: change\n\nCause: a fixture authored by the review.\nPredicted effect: failure-rate holds.\n":                             "",
	})
}
