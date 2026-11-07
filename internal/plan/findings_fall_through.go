package plan

import (
	"cmp"
	"encoding/json"
	"slices"
	"strings"
	"unicode"
)

// FindingsRegister is the register a review session writes its findings into, beside the plan.
const FindingsRegister = "docs/sqa_findings.json"

// The severities an empty plan falls through to, most urgent first; a finding of any other is filed only when the owner
// names it, so "never idle" never turns the register into endless low-value commits.
var fallThroughSeverities = []string{"P1", "P2"}

// openFinding is the part of a finding fall-through reads.
type openFinding struct {
	ID       string `json:"id"`
	Status   string `json:"status"`
	Severity string `json:"severity"`
	Title    string `json:"title"`
}

// findingOpen is the status of a finding the reviewer has not closed.
const findingOpen = "open"

// FindingsToFile are the findings an empty plan falls through to before the full measurement run: those open at P1
// and then P2, each that no row of the plan names yet, written "SQA-n (Pn): title". A register that holds none, or a
// plan whose rows name each, leaves the measurement run next.
func FindingsToFile(document Plan, register []byte) ([]string, error) {
	// The register holds more than fall-through reads, its reviewer's fields among them, so it is read for these alone.
	var read struct {
		Findings []openFinding `json:"findings"`
	}
	if err := json.Unmarshal(register, &read); err != nil {
		return nil, err
	}
	findings := slices.DeleteFunc(read.Findings, func(finding openFinding) bool {
		return finding.Status != findingOpen || !slices.Contains(fallThroughSeverities, finding.Severity) || namedByARow(document, finding.ID)
	})
	slices.SortStableFunc(findings, func(a, b openFinding) int {
		return cmp.Compare(slices.Index(fallThroughSeverities, a.Severity), slices.Index(fallThroughSeverities, b.Severity))
	})
	filed := make([]string, len(findings))
	for index, finding := range findings {
		filed[index] = finding.ID + " (" + finding.Severity + "): " + finding.Title
	}
	return filed, nil
}

// namedByARow says a row of the plan names the finding: its title, a step's title or a step's reasoning holds the id,
// not followed by another digit, so SQA-33 is not named by SQA-334.
func namedByARow(document Plan, id string) bool {
	names := func(text string) bool {
		for rest := text; ; {
			at := strings.Index(rest, id)
			if at < 0 {
				return false
			}
			rest = rest[at+len(id):]
			if rest == "" || !unicode.IsDigit([]rune(rest)[0]) {
				return true
			}
		}
	}
	return slices.ContainsFunc(document.Items, func(item Item) bool {
		return names(item.Title) || slices.ContainsFunc(item.Steps, func(step Step) bool { return names(step.Title) || names(step.Rationale) })
	})
}
