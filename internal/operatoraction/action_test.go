// Package operatoraction_test holds the recovery and approval envelopes to the exported API that test support builds them through.
package operatoraction_test

import (
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/testutil"
)

func TestActionEnvelopeContract(t *testing.T) {
	t.Parallel()
	subject := testutil.ArtifactID(t, artifact.KindRecipe, "blocked recipe")
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "missing evaluation")
	action := operatoraction.Action{Code: "run-evaluation", Summary: "Run the required evaluation", Argv: []string{"overgo", "evaluate", "--recipe", subject.String()}}
	block := operatoraction.Block{Subject: subject, Reason: "promotion requires held-out evidence", Evidence: []artifact.ID{evidence}, Actions: []operatoraction.Action{action}}
	if err := block.Validate(); err != nil {
		t.Fatal(err)
	}
	failure := testutil.Recoverable(errors.New("promotion blocked"), block)
	recovered, ok := operatoraction.Recovery(failure)
	if !ok || recovered.Subject != subject || len(recovered.Actions) != 1 {
		t.Fatalf("recovery = (%+v, %t)", recovered, ok)
	}
	recovered.Actions[0].Argv[0] = "mutated"
	again, _ := operatoraction.Recovery(failure)
	if again.Actions[0].Argv[0] != "overgo" {
		t.Fatal("recovery transport exposed mutable command storage")
	}

	invalidBlock := block.Clone()
	invalidBlock.Actions[0].Argv = []string{"overgo", "evaluate\nmalformed"}
	if _, ok := operatoraction.Recovery(testutil.Recoverable(errors.New("blocked"), invalidBlock)); ok {
		t.Fatal("malformed argv was advertised as recovery")
	}
}
