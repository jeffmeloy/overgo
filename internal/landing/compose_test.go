package landing

import (
	"io"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// composeRegister is a findings register holding a finding that cites the change's file and one that cites another.
const composeRegister = `{"findings": [
 {"id": "SQA-1", "severity": "P2", "actual": "a finding on p", "files": ["p/p.go"]},
 {"id": "SQA-2", "severity": "P2", "actual": "a finding elsewhere", "files": ["q/q.go"]}]}
`

// TestTheLandingWritesItsOwnMessage lands a change with no message: from Oct 31 to Nov 7, 168 of 263 refused commit
// attempts were a hand-written message's sections and figures. The commit's message holds the step's title, its cause
// and what it serves from the plan, the files changed and the test added, the step retired with why and the count set,
// and the campaign the plan now names, and the finding that cites a changed file, not one that cites none; the
// published plan keeps no retirement. The lead's words, when given, follow the title and keep the sections they hold.
func TestTheLandingWritesItsOwnMessage(t *testing.T) {
	t.Parallel()
	fixture := newCheckedFixture(t, map[string]string{"p/p.go": "// Package p is a fixture.\npackage p\n", findingsRegister: composeRegister}, "grep -q changed p/p.go")
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items[0].Steps[0].Title = "the fixture's step"
		document.Items[0].Steps[0].Rationale = "A fixture needs a landing.\nServes: failure-rate.\nBatch: host lane."
		document.Items = append(document.Items, plan.Item{ID: "later", Title: "later",
			Steps: []plan.Step{{ID: "do", Title: "later", Verify: "false"}}})
	})
	fixture.git("add", "-A")
	fixture.git("commit", "-q", "-m", "plan of two rows")
	fixture.write("p/p.go", "// Package p is a fixture, changed.\npackage p\n")
	fixture.write("p/p_test.go", "package p\n\nimport \"testing\"\n\nfunc TestNew(t *testing.T) { t.Parallel() }\n")
	editPlan(t, fixture, func(document *plan.Plan) {
		document.Items = document.Items[:1]
		document.Campaign = "the fixture's next campaign"
		document.Items[0].Class, document.Items[0].Budget = plan.ClassConsolidation, plan.Budget{Nodes: 40, Reason: "the fixture grows"}
		document.Retired = []plan.Retirement{{Row: "later/do", Reason: "the fixture needs it no more"}}
	})

	lead := &landing{Options: Options{Repo: fixture.repo, Ref: "row/do", Message: []byte("Words of the lead.\n\nCause: the lead's own cause.\n"), Log: io.Discard},
		scratch: t.TempDir(), details: map[string]string{}}
	testcheck.NoError(t, lead.parents(t.Context()))
	testcheck.NoError(t, lead.shipSet(t.Context()))
	if !strings.HasPrefix(lead.composed, "row/do: the fixture's step\n\nWords of the lead.") || strings.Count(lead.composed, causeSection) != 1 ||
		strings.Contains(lead.composed, "A fixture needs a landing.") || !strings.Contains(lead.composed, findingsSection) {
		t.Errorf("with the lead's words the message reads:\n%s\nwant the step's title, the words, their own cause and the other sections", lead.composed)
	}

	testcheck.Must(Run(t.Context(), Options{Repo: fixture.repo, Ref: "row/do", Log: io.Discard}))(t)
	body := fixture.git("log", "-1", "--format=%B")
	for _, want := range []string{"row/do: the fixture's step\n", causeSection + "\n- A fixture needs a landing.", "p/p.go", "tests added: TestNew",
		"later/do retired: the fixture needs it no more", "row's count is set to 40 nodes from 0: the fixture grows",
		"the campaign is now: the fixture's next campaign", "SQA-1 (P2): a finding on p; cites p/p.go", predictedSection + "\n- failure-rate."} {
		if !strings.Contains(body, want) {
			t.Errorf("the landing's own message lacks %q:\n%s", want, body)
		}
	}
	if strings.Contains(body, "SQA-2") {
		t.Errorf("the message names a finding that cites no changed file:\n%s", body)
	}
	if retired := testcheck.Must(plan.Load(fixture.planPath()))(t).Retired; len(retired) != 0 {
		t.Errorf("the published plan keeps its retirements %v", retired)
	}
}
