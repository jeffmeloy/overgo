package plan

import (
	"cmp"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/strictjson"
)

// FindingsRegister is the register a review session writes its findings into, beside the plan.
const FindingsRegister = "docs/sqa_findings.json"

// The severities an empty plan falls through to, most urgent first; a finding of any other is filed only when the owner
// names it, so "never idle" never turns the register into endless low-value commits.
var fallThroughSeverities = []string{"P1", "P2"}

// openFinding is a finding as the review writes it (owner, 2026-11-14): its id and severity, the files it cites, the
// input that shows it, what is expected of that input and what actually happens. A finding is open while the register
// holds it; one closed leaves it.
type openFinding struct {
	ID       string   `json:"id"`
	Severity string   `json:"severity"`
	Files    []string `json:"files"`
	Input    string   `json:"input"`
	Expected string   `json:"expected"`
	Actual   string   `json:"actual"`
}

// named is the finding as a line names it, "SQA-n (Pn): what actually happens".
func (f openFinding) named() string { return f.ID + " (" + f.Severity + "): " + f.Actual }

// triageDeferred is the lead's triage of a finding whose deciding evidence the next measurement run records,
// triageImplemented of one the lead answered with rows, which waits for a recheck of its input against what it expects
// to close or reopen it, and triageDeclined of one the lead will not work, its note saying why.
const (
	triageDeferred    = "deferred"
	triageImplemented = "implemented"
	triageDeclined    = "declined"
)

// readFindings reads the register's findings, each held to openFinding's fields: a finding naming another, its status
// or its title, is refused, so a review writes no field no program reads. The register holds more beside its
// findings, the lead's triage among them.
func readFindings(register []byte) ([]openFinding, error) {
	var read struct {
		Findings []json.RawMessage `json:"findings"`
	}
	if err := json.Unmarshal(register, &read); err != nil {
		return nil, err
	}
	findings := make([]openFinding, len(read.Findings))
	for index, finding := range read.Findings {
		if err := strictjson.DecodeBytes(finding, &findings[index]); err != nil {
			return nil, fmt.Errorf("%s: finding %d of %d holds id, severity, files, input, expected and actual alone: %w",
				FindingsRegister, index+1, len(read.Findings), err)
		}
	}
	return findings, nil
}

// FindingsTouching are the findings that cite a path of paths, written "SQA-n (Pn): what actually happens; cites path,
// ...": the ones a change to those paths may move, which its landing's message names.
func FindingsTouching(register []byte, paths []string) ([]string, error) {
	findings, err := readFindings(register)
	var touching []string
	for _, finding := range findings {
		var cited []string
		for _, file := range finding.Files {
			if slices.Contains(paths, file) && !slices.Contains(cited, file) {
				cited = append(cited, file)
			}
		}
		if len(cited) > 0 {
			touching = append(touching, finding.named()+"; cites "+strings.Join(cited, ", "))
		}
	}
	return touching, err
}

// FindingsToFile are the findings an empty plan falls through to before the full measurement run: those at P1 and then
// P2, each that no row of the plan names yet, written "SQA-n (Pn): what actually happens". A register that holds none,
// or a plan whose rows name each, leaves the measurement run next.
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
		return !slices.Contains(fallThroughSeverities, finding.Severity) || namedByARow(document, finding.ID) ||
			slices.Contains([]string{triageDeferred, triageImplemented, triageDeclined}, triage.Triage.Findings[finding.ID].Action)
	})
	slices.SortStableFunc(findings, func(a, b openFinding) int {
		return cmp.Compare(slices.Index(fallThroughSeverities, a.Severity), slices.Index(fallThroughSeverities, b.Severity))
	})
	filed := make([]string, len(findings))
	for index, finding := range findings {
		filed[index] = finding.named()
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
