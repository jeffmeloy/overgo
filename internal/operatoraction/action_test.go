package operatoraction

import (
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestActionEnvelopeContract(t *testing.T) {
	subject := testutil.ArtifactID(t, artifact.KindRecipe, "blocked recipe")
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "missing evaluation")
	action := Action{Code: "run-evaluation", Summary: "Run the required evaluation", Argv: []string{"overgo", "evaluate", "--recipe", subject.String()}}
	block := Block{Subject: subject, Reason: "promotion requires held-out evidence", Evidence: []artifact.ID{evidence}, Actions: []Action{action}}
	if err := block.Validate(); err != nil {
		t.Fatal(err)
	}
	failure := Recoverable(errors.New("promotion blocked"), block)
	recovered, ok := Recovery(failure)
	if !ok || recovered.Subject != subject || len(recovered.Actions) != 1 {
		t.Fatalf("recovery = (%+v, %t)", recovered, ok)
	}
	recovered.Actions[0].Argv[0] = "mutated"
	again, _ := Recovery(failure)
	if again.Actions[0].Argv[0] != "overgo" {
		t.Fatal("recovery transport exposed mutable command storage")
	}

	consent := Consent{
		Subject: subject, Headline: "Publish candidate?", Reason: "publication changes the active alias",
		Value: subject.String(), Evidence: []artifact.ID{evidence},
		Choices: []Choice{
			{Answer: AnswerGrant, Label: "Publish", Effect: "promote this exact subject", Action: Action{Code: "grant", Summary: "Grant publication", Argv: []string{"overgo", "promote", "--subject", subject.String()}}},
			{Answer: AnswerDecline, Label: "Keep blocked", Effect: "leave the active alias unchanged", Action: Action{Code: "decline", Summary: "Decline publication", Argv: []string{"overgo", "promote", "--decline", subject.String()}}},
		},
		OffPath: Action{Code: "inspect-policy", Summary: "Inspect the governing policy", Argv: []string{"overgo", "inspect", "--artifact", evidence.String()}},
	}
	if err := consent.Validate(); err != nil {
		t.Fatal(err)
	}
	invalid := consent.Clone()
	invalid.Choices[0], invalid.Choices[1] = invalid.Choices[1], invalid.Choices[0]
	if err := invalid.Validate(); err == nil {
		t.Fatal("reordered consent answers accepted")
	}
	invalid = consent.Clone()
	invalid.Choices[0].Action.Argv = nil
	if err := invalid.Validate(); err == nil {
		t.Fatal("consent with no runnable grant accepted")
	}
	invalidBlock := block.Clone()
	invalidBlock.Actions[0].Argv = []string{"overgo", "evaluate\nmalformed"}
	if _, ok := Recovery(Recoverable(errors.New("blocked"), invalidBlock)); ok {
		t.Fatal("malformed argv was advertised as recovery")
	}
}
