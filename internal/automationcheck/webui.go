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
// the web UI fact is present and shares the device under the context-lifetime
// admission: its served model fits beside correctness tests, and only an
// explicit measurement holds the device exclusively.
func WebUICheck(root string, command Command) Check {
	return Check{
		Descriptor: Descriptor{
			Name: WebUICheckName, Phase: runrecord.PhaseTest, Triggers: []Fact{WebUIImpact},
			Inapplicable: "no web UI asset, shell or lane implementation changed",
			Resources:    []Resource{{Name: "device"}},
			Ownership: Ownership{
				Fact: WebUIImpact, Packages: []string{"internal/server", "internal/webuilane", "cmd/webui-lane"},
			},
		},
		Run: func(context.Context, Invocation) (bool, string, error) {
			_, err := command(root, "go", "run", "./cmd/webui-lane")
			return false, "", err
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
