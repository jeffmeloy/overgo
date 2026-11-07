package landing

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/plan"
)

// The message's sections, in the order the landing writes them, each opening its own line; servesLine opens the line
// of a step's reasoning that says what the step serves.
const (
	causeSection     = "Cause:"
	changeSection    = "Change:"
	planSection      = "Plan, since the last landing:"
	findingsSection  = "Findings it touches:"
	predictedSection = "Predicted effect:"
	servesLine       = "Serves:"
)

// compose writes the message the landing commits, so no hand writes one: the landed steps' titles, why they exist and
// what they serve from their reasoning in the plan, the files the change touches and the tests it adds, what it did to
// the plan since the last landing, and the open findings that cite a file it touches. The lead's own words, when
// given, follow the title, and a section they already hold is not written again; words that open with the landed
// step stand as the title.
func (l *landing) compose(ctx context.Context, before, document plan.Plan, named []ref) (string, error) {
	var titles, causes, serves []string
	for _, row := range named {
		step, _ := plan.ExactStep(document, row.item, row.step)
		titles = append(titles, step.Title)
		lines := strings.Split(step.Rationale, "\n")
		if cause := strings.TrimSpace(lines[0]); cause != "" {
			causes = append(causes, cause)
		}
		for _, line := range lines {
			if served, found := strings.CutPrefix(line, servesLine); found {
				serves = append(serves, strings.TrimSpace(served))
			}
		}
	}
	change := []string{"files: " + strings.Join(l.paths, ", ")}
	if added := l.testsAdded(ctx); len(added) > 0 {
		change = append(change, "tests added: "+strings.Join(added, ", "))
	}
	// A tree without the register has no findings to name; one the landing cannot read is said, and refuses nothing.
	var touching []string
	if register, absent := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.candidate+":"+findingsRegister); absent == nil {
		var unread error
		if touching, unread = plan.FindingsTouching(register, l.paths); unread != nil {
			touching = []string{"the register could not be read: " + unread.Error()}
		}
	}
	authored := strings.TrimSpace(string(l.Message))
	message := authored
	if !strings.HasPrefix(authored, named[0].String()) {
		message = strings.TrimSpace(l.Ref + ": " + strings.Join(titles, "; ") + "\n\n" + authored)
	}
	for _, part := range []struct {
		label string
		lines []string
	}{
		{causeSection, causes}, {changeSection, change}, {planSection, planChanges(before, document)},
		{findingsSection, touching}, {predictedSection, serves},
	} {
		message += l.section(part.label, part.lines)
	}
	return message, nil
}

// section writes one section of the message: its label on a line of its own and each line as a point under it, or
// "none." when it has none; nothing when the lead's words already hold it.
func (l *landing) section(label string, lines []string) string {
	if strings.Contains("\n"+strings.TrimSpace(string(l.Message)), "\n"+label) {
		return ""
	}
	if len(lines) == 0 {
		lines = []string{"none."}
	}
	return "\n\n" + label + "\n- " + strings.Join(lines, "\n- ")
}

// testsAdded are the tests the change's test files declare that the commit it builds on does not: a file either side
// lacks reads as declaring none.
func (l *landing) testsAdded(ctx context.Context) []string {
	var added []string
	for _, path := range l.paths {
		if !strings.HasSuffix(path, "_test.go") {
			continue
		}
		now, _ := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.candidate+":"+path)
		was, _ := gitauthority.Query(ctx, l.Repo, "cat-file", "blob", l.builtOn()+":"+path)
		held := testNames(was)
		for _, name := range testNames(now) {
			if !slices.Contains(held, name) {
				added = append(added, name)
			}
		}
	}
	return added
}

// testNames are the tests a Go source file declares, in its order.
func testNames(source []byte) []string {
	var names []string
	for line := range strings.Lines(string(source)) {
		if rest, found := strings.CutPrefix(line, "func Test"); found {
			name, _, _ := strings.Cut(rest, "(")
			names = append(names, "Test"+name)
		}
	}
	return names
}

// planChanges are what a change does to the plan beside landing its steps: the steps it retires and why, the counts it
// sets and why, and the steps it files.
func planChanges(before, document plan.Plan) []string {
	var changes []string
	for _, retired := range document.Retired {
		changes = append(changes, retired.Row+" retired: "+retired.Reason)
	}
	for _, item := range document.Items {
		at := slices.IndexFunc(before.Items, func(held plan.Item) bool { return held.ID == item.ID })
		if at >= 0 && before.Items[at].Budget.Nodes != item.Budget.Nodes {
			changes = append(changes, fmt.Sprintf("%s's count is set to %d nodes from %d: %s", item.ID, item.Budget.Nodes, before.Items[at].Budget.Nodes, item.Budget.Reason))
		}
		for _, step := range item.Steps {
			if at < 0 || !slices.ContainsFunc(before.Items[at].Steps, func(held plan.Step) bool { return held.ID == step.ID }) {
				changes = append(changes, item.ID+"/"+step.ID+" filed: "+step.Title)
			}
		}
	}
	return changes
}
