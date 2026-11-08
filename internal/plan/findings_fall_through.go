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

// openFinding is the part of a finding the plan and the landing read: its id, state and title, and the files it cites.
type openFinding struct {
	ID       string `json:"id"`
	Status   string `json:"status"`
	Severity string `json:"severity"`
	Title    string `json:"title"`
	Sources  []struct {
		Path string `json:"path"`
	} `json:"sources"`
}

// findingOpen is the status of a finding the reviewer has not closed, and triageDeferred the lead's triage of one whose
// deciding evidence the next measurement run records.
const (
	findingOpen    = "open"
	triageDeferred = "deferred"
)

// readFindings reads the register's findings. It holds more, its reviewer's fields among them, so it is read for the
// fields openFinding holds alone.
func readFindings(register []byte) ([]openFinding, error) {
	var read struct {
		Findings []openFinding `json:"findings"`
	}
	err := json.Unmarshal(register, &read)
	return read.Findings, err
}

// FindingsTouching are the open findings that cite a path of paths, written "SQA-n (Pn): title; cites path, ...": the
// ones a change to those paths may move, which its landing's message names.
func FindingsTouching(register []byte, paths []string) ([]string, error) {
	findings, err := readFindings(register)
	var touching []string
	for _, finding := range findings {
		var cited []string
		for _, source := range finding.Sources {
			if finding.Status == findingOpen && slices.Contains(paths, source.Path) && !slices.Contains(cited, source.Path) {
				cited = append(cited, source.Path)
			}
		}
		if len(cited) > 0 {
			touching = append(touching, finding.ID+" ("+finding.Severity+"): "+finding.Title+"; cites "+strings.Join(cited, ", "))
		}
	}
	return touching, err
}

// FindingsToFile are the findings an empty plan falls through to before the full measurement run: those open at P1
// and then P2, each that no row of the plan names yet, written "SQA-n (Pn): title". A register that holds none, or a
// plan whose rows name each, leaves the measurement run next.
func FindingsToFile(document Plan, register []byte) ([]string, error) {
	findings, err := readFindings(register)
	if err != nil {
		return nil, err
	}
	// A finding the lead deferred to the run's record waits for that run, and so never holds it back (SQA-266).
	var triage struct {
		Triage struct {
			Findings map[string]struct {
				Action string `json:"action"`
			} `json:"findings"`
		} `json:"master_lead_triage"`
	}
	if err := json.Unmarshal(register, &triage); err != nil {
		return nil, err
	}
	findings = slices.DeleteFunc(findings, func(finding openFinding) bool {
		return finding.Status != findingOpen || !slices.Contains(fallThroughSeverities, finding.Severity) || namedByARow(document, finding.ID) ||
			triage.Triage.Findings[finding.ID].Action == triageDeferred
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
