package evaluation

import (
	"context"
	"encoding/json"

	"overgo/internal/artifact"
)

const (
	// mmluProReportMedia and mmluProReportSchema name an MMLU-Pro report the store holds.
	//   - an MMLU-Pro suite = a grouped-choice suite => a reading of it writes a grouped-choice report
	//   - a report of this form stays readable because the store keeps it
	mmluProReportMedia  = "application/vnd.overgo.mmlu-pro-report+json"
	mmluProReportSchema = "overgo/mmlu-pro-report/v1"
	// mmluProSchema names the task the store's MMLU-Pro suite is a port of.
	mmluProSchema = "lm-eval/leaderboard-mmlu-pro/v0.1"
	// gluedLetters is the part of a protocol label that marks a likelihood
	// reading of MMLU-Pro taken before its candidates were given the task's
	// target delimiter: each option letter was scored glued to the answer
	// cue, "Answer:A".
	gluedLetters = "letters glued to the cue"
)

// scoredGlued reports whether a plan is a likelihood scoring of MMLU-Pro
// that glued each option letter to the answer cue: the suite of that task,
// under a scorer that names no method, which is every likelihood plan of it
// bound before the delimiter was. Such a record is not a reading of the
// benchmark as its harness asks it, and its label says so.
func scoredGlued(ctx context.Context, reader artifact.Reader, plan Plan) bool {
	profile, found, err := artifact.ReadContent(ctx, reader, plan.body.CaseProfile)
	if err != nil || !found {
		return false
	}
	var suite struct {
		Kind   string `json:"kind"`
		Schema string `json:"schema"`
	}
	if json.Unmarshal(profile.Data, &suite) != nil || suite.Kind != MultipleChoiceKind || suite.Schema != mmluProSchema {
		return false
	}
	scorer, found, err := artifact.ReadContent(ctx, reader, plan.body.Scorer)
	if err != nil || !found {
		return false
	}
	var method struct {
		Method string `json:"method"`
	}
	return json.Unmarshal(scorer.Data, &method) == nil && method.Method == ""
}
