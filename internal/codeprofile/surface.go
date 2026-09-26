package codeprofile

import (
	"slices"
	"strings"

	"overgo/internal/protection"
)

// Surface is the set of classes one repository path belongs to, decided here
// once: the gate's check ownership, its ship-set inputs, the browser lanes and
// the profile's partitions all read it rather than keep their own prefixes.
type Surface uint16

const (
	// GoSource is a Go file of a package under cmd/ or internal/.
	GoSource Surface = 1 << iota
	// GoInput is what a build reads: Go, module and non-markdown package files.
	GoInput
	// Document is markdown, docs/, the published claims and the SBOM.
	Document
	// Plan is the campaign plan.
	Plan
	// Budget is an architecture budget the ratchets read.
	Budget
	// Harness is a hook script or the protection harness configuration.
	Harness
	// TestData is what tests read beside Go: README and docs/ but the plan.
	TestData
	// WebUI is the web UI's assets, browser tests, lane and page declarations.
	WebUI
	// ModelJourney is a served-model journey or the server command.
	ModelJourney
	// Automation is a command's Go source.
	Automation
)

var surfacePrefixes = []struct {
	surface  Surface
	prefixes []string
}{
	{WebUI, []string{"internal/server/webui/", "internal/server/webui_", "cmd/webui-lane/", "internal/webuilane/",
		"internal/server/workspace_manifest.json", "internal/server/workspace_schema.json", "internal/server/library_validation.json"}},
	{ModelJourney, []string{"cmd/server/", "internal/server/webui_browser_firstrun", "internal/server/webui_browser_library_validation",
		"internal/server/webui_browser_preparation", "internal/server/webui_browser_store_test.go", "internal/server/webui_served_projector",
		"internal/audioparity/webui_microphone"}},
	{Harness, []string{"scripts/", protection.HarnessConfigDirectory}},
	{Budget, []string{"docs/staged_surface.json", "docs/structure_budgets.json"}},
}

// Classify returns every class path belongs to; path is slash-separated and
// relative to the repository root.
func Classify(path string) Surface {
	var surface Surface
	set := func(class Surface, holds bool) {
		if holds {
			surface |= class
		}
	}
	goFile, markdown, docs := strings.HasSuffix(path, ".go"), strings.HasSuffix(path, ".md"), strings.HasPrefix(path, "docs/")
	packaged := strings.HasPrefix(path, "cmd/") || strings.HasPrefix(path, "internal/")
	set(GoSource, goFile && packaged)
	set(GoInput, goFile || path == "go.mod" || path == "go.sum" || (packaged || strings.HasPrefix(path, "kernels/")) && !markdown)
	set(Document, markdown || docs || path == "compatibility.json" || path == "SBOM.cdx.json")
	set(Plan, path == "docs/plan.json")
	set(TestData, path == "README.md" || docs && path != "docs/plan.json")
	set(Automation, goFile && strings.HasPrefix(path, "cmd/"))
	for _, rule := range surfacePrefixes {
		set(rule.surface, slices.ContainsFunc(rule.prefixes, func(prefix string) bool { return strings.HasPrefix(path, prefix) }))
	}
	return surface
}

// Holds reports whether any path belongs to a class of surface.
func Holds(paths []string, surface Surface) bool {
	return slices.ContainsFunc(paths, func(path string) bool { return Classify(strings.ReplaceAll(path, "\\", "/"))&surface != 0 })
}
