package automationcheck

import (
	"slices"
	"testing"
)

// TestWebUICheckSelection pins the lane's selection: a changed web UI
// asset path triggers the fact the symbol closure cannot see, the trigger
// withdraws the lane's exclusion while keeping every other, and the check
// declares the server package that embeds the shell as its ownership and
// the device as an exclusive resource.
func TestWebUICheckSelection(t *testing.T) {
	for path, owned := range map[string]bool{
		"internal/server/webui/mod/chat.js": true, "internal\\server\\webui\\style.css": true,
		"internal/server/webui_static_test.go": true, "cmd/webui-lane/main.go": true, "internal/webuilane/browser.go": true,
		"internal/server/routes.go": false, "docs/plan.json": false,
	} {
		if WebUIPaths([]string{path}) != owned {
			t.Errorf("WebUIPaths(%q) = %v, want %v", path, !owned, owned)
		}
	}
	impact := Impact{Facts: []Fact{"capability:device"}, Exclusions: []Exclusion{
		{Check: WebUICheckName, Reason: "disjoint"}, {Check: "device", Reason: "disjoint"},
	}}
	triggered := impact.Trigger(WebUIImpact, WebUICheckName)
	if !slices.Contains(triggered.Facts, WebUIImpact) || len(triggered.Exclusions) != 1 || triggered.Exclusions[0].Check != "device" {
		t.Fatalf("triggered impact = %+v", triggered)
	}
	if len(impact.Exclusions) != 2 {
		t.Fatal("Trigger changed the original impact")
	}
	var ran []string
	check := WebUICheck("root", func(root, name string, arguments ...string) (string, error) {
		ran = append(ran, root, name)
		ran = append(ran, arguments...)
		return "", nil
	})
	if check.Descriptor.Name != WebUICheckName || !slices.Contains(check.Descriptor.Triggers, WebUIImpact) ||
		!slices.Contains(check.Descriptor.Ownership.Packages, "internal/server") || len(check.Descriptor.Resources) != 1 || !check.Descriptor.Resources[0].Exclusive {
		t.Fatalf("descriptor = %+v", check.Descriptor)
	}
	if _, _, err := check.Run(t.Context(), Invocation{}); err != nil || !slices.Equal(ran, []string{"root", "go", "run", "./cmd/webui-lane"}) {
		t.Fatalf("run = %v %v", ran, err)
	}
	own := Impact{Exclusions: []Exclusion{{Check: WebUICheckName, Reason: "disjoint"}}}
	invocations, err := Plan([]Check{check}, own.Trigger(WebUIImpact, WebUICheckName))
	if err != nil || len(invocations) != 1 {
		t.Fatalf("planned = %+v, %v", invocations, err)
	}
	if excluded, err := Plan([]Check{check}, own); err != nil || len(excluded) != 0 {
		t.Fatalf("excluded plan = %+v, %v", excluded, err)
	}
}
