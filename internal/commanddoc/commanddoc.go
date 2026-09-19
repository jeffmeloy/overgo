// Package commanddoc is the single source for a shipped command's help identity
// and its API-manifest discovery classification, so a command's help output and
// its manifest projection share one owner-declared declaration rather than
// drifting between a help string and a parallel manifest registry.
package commanddoc

import "overgo/internal/clioptions"

// Descriptor pairs the command's help identity with the caller classification the
// manifest records. Purpose, audience, and constraints render in help; the
// classification is manifest-only, decided by the command's actual callers.
type Descriptor struct {
	Command        clioptions.Command
	Classification string
}

// Plan describes the plan command: the master-lead session and its unattended
// driver dispatch through it, and operators inspect and steer it, so it is
// called both ways.
var Plan = Descriptor{
	Command: clioptions.Command{
		Name:     "plan",
		Purpose:  "edit and dispatch the validated campaign in docs/plan.json; dispatch claims work while the gate verifies, publishes and advances it",
		Audience: "the master-lead session and its unattended driver working one dispatched row at a time",
		Constraints: []string{
			"-add takes the step's verify command through -vcmd; boolean -verify runs the top open step's verify",
			"one operation per invocation; committing and advancing a row belong to cmd/gate, not plan",
			"help opens no store and mutates no plan",
			"scratch probes live in tmp (module overgo/tmp, replace overgo => ..): run one with go run ./tmp/<file>.go from the repository root or go run ./<file>.go from tmp; they stay outside ./...",
		},
	},
	Classification: "both",
}

// OvergodbQuery describes the overgodb-query command: an operator or an agent
// reads the committed catalog through it without mutating anything.
var OvergodbQuery = Descriptor{
	Command: clioptions.Command{
		Name:     "overgodb-query",
		Purpose:  "read the catalog: filtered artifact listings, lineage traversals, and the derived operational ledgers",
		Audience: "an operator or agent inspecting committed OvergoDB state without mutating it",
		Constraints: []string{
			"-repo names the store (or empty for the data-root contract) and -limit is a positive result bound",
			"-content prints the raw committed bytes of the artifact named by -id",
			"help opens no store and reads nothing",
		},
	},
	Classification: "both",
}
