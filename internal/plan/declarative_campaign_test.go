package plan

import (
	"slices"
	"testing"
)

// A bounded plan addition needs no matching source-code inventory edit.
// Its acceptance and terminal obligation remain mandatory.
func TestValidationPlanExtensibleChecks(t *testing.T) {
	const reference = "validation-probe/bounded-check"
	for _, test := range []struct {
		name, verify string
		covered      bool
		valid        bool
	}{
		{"bounded row", "go test ./internal/plan -run '^TestSingleCanonicalCampaignPlan$' -count=1", true, true},
		{"missing verification", "", true, false},
		{"missing closeout coverage", "go test ./internal/plan -run '^TestSingleCanonicalCampaignPlan$' -count=1", false, false},
		{"nonacceptance verification", "go run ./cmd/plan -status", true, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			document := Plan{
				Campaign: "Bounded validation fixture",
				Doctrine: "capability freeze enforced by structure; report UNAVAILABLE; bind acceptance with plan -setverify",
				Items: []Item{{
					ID: "campaign-closeout", Title: "Close validation campaign", Status: StatusOpen,
					Steps: []Step{{ID: "closeout", Title: "Verify campaign structure", Status: StatusOpen,
						Verify: "go test ./internal/plan -run '^TestSingleCanonicalCampaignPlan$' -count=1"}},
				}},
			}
			document.Items = append(document.Items, Item{
				ID: "validation-probe", Title: "Bounded validation check", Status: StatusOpen,
				Steps: []Step{{ID: "bounded-check", Title: "Validate existing behavior", Status: StatusOpen, Verify: test.verify}},
			})
			if test.covered {
				for i := range document.Items {
					if document.Items[i].ID == "campaign-closeout" {
						document.Items[i].Steps[0].DependsOn = append(slices.Clone(document.Items[i].Steps[0].DependsOn), reference)
					}
				}
			}
			problems := optimizedValidationProblems(document)
			if (len(problems) == 0) != test.valid {
				t.Fatalf("valid=%t problems=%v", test.valid, problems)
			}
			if test.valid {
				assertValidationCampaignSnapshot(t, document)
			}
		})
	}
}
