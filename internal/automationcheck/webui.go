package automationcheck

import (
	"context"
	"slices"

	"overgo/internal/runrecord"
)

// The web UI lane is selected by the code manifest for every commit that
// touches the web UI (professional GUI campaign, gui-quality/acceptance-lane):
// its ownership names the server package that embeds the shell and the
// lane's own packages, and because the shell's assets are not Go symbols,
// a changed path under the web UI also triggers it directly.
const (
	// WebUICheckName is the lane's check name in the gate pipeline.
	WebUICheckName = "webui-lane"
	// WebUIImpact is the fact a web UI change contributes to the manifest plan.
	WebUIImpact Fact = "capability:webui"
)

// WebUICheck returns the real-browser lane as a gate check: it runs when
// the web UI fact is present. The full journey shares device admission.
func WebUICheck(root string, command LaneCommand) Check {
	return Check{
		Descriptor: Descriptor{
			Name: WebUICheckName, Phase: runrecord.PhaseTest, Triggers: []Fact{WebUIImpact},
			Inapplicable: "no web UI asset, shell or lane implementation changed",
			Resources:    []Resource{{Name: "device"}},
			Requirements: Requirements{Process: ProcessBrowser},
			Ownership: Ownership{
				Fact: WebUIImpact, Packages: []string{"internal/server", "internal/webuilane", "cmd/webui-lane"},
			},
		},
		Run: func(ctx context.Context, _ Invocation) (bool, string, error) {
			receipt, err := command(ctx, root, "go", "run", "./cmd/webui-lane")
			return false, receipt, err
		},
	}
}

// The model journeys drive the page against served, downloaded and hashed
// models. They belong to the serving owners: the check runs when serving,
// intake or discovery code changes, or a journey itself, and the page lane
// stays free of model work.
const (
	// ModelJourneyCheckName is the journey check's name in the gate pipeline.
	ModelJourneyCheckName = "model-journey"
	// ModelJourneyImpact is the fact a serving change contributes to the manifest plan.
	ModelJourneyImpact Fact = "capability:model-journey"
)

// ModelJourneyCheck returns the model journeys as a gate check: the lane's
// -journeys mode, sharing device admission with the other lanes.
func ModelJourneyCheck(root string, command LaneCommand) Check {
	return Check{
		Descriptor: Descriptor{
			Name: ModelJourneyCheckName, Phase: runrecord.PhaseTest, Triggers: []Fact{ModelJourneyImpact},
			Inapplicable: "no model serving, intake or discovery implementation changed, and no journey",
			Resources:    []Resource{{Name: "device"}},
			Requirements: Requirements{Process: ProcessBrowser},
			Ownership: Ownership{
				Fact: ModelJourneyImpact, Packages: []string{"internal/modelswap", "internal/libraryintake", "internal/discovery", "cmd/server"},
			},
		},
		Run: func(ctx context.Context, _ Invocation) (bool, string, error) {
			receipt, err := command(ctx, root, "go", "run", "./cmd/webui-lane", "-journeys")
			return false, receipt, err
		},
	}
}

// Trigger adds one fact to the impact and withdraws the exclusion of the
// named check, for a change the symbol closure cannot see.
func (impact Impact) Trigger(fact Fact, check string) Impact {
	if !slices.Contains(impact.Facts, fact) {
		impact.Facts = append(slices.Clone(impact.Facts), fact)
		slices.Sort(impact.Facts)
	}
	impact.Exclusions = slices.DeleteFunc(slices.Clone(impact.Exclusions), func(exclusion Exclusion) bool { return exclusion.Check == check })
	return impact
}
