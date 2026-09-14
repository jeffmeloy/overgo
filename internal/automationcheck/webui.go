package automationcheck

import (
	"context"
	"slices"
	"strings"

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

// webuiPathPrefixes are the changed-path prefixes that own the web UI.
var webuiPathPrefixes = []string{"internal/server/webui/", "internal/server/webui_", "cmd/webui-lane/", "internal/webuilane/"}

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

// WebUIPaths reports whether any changed path belongs to the web UI.
func WebUIPaths(paths []string) bool {
	return slices.ContainsFunc(paths, func(changed string) bool {
		normalized := strings.ReplaceAll(changed, "\\", "/")
		return slices.ContainsFunc(webuiPathPrefixes, func(prefix string) bool { return strings.HasPrefix(normalized, prefix) })
	})
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
