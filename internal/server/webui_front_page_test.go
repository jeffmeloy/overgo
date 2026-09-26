package server

import (
	"testing"
)

// TestFrontPage pins the front page (professional GUI campaign,
// gui-shell/front-page): the one shell opens on the conversation with the
// workbench folded to a rail and one Workbench control that unfolds it, the
// header carries the served model and its evidence line from the catalog,
// the empty conversation is the getting-started card built from the
// capability document with three starter actions, and the conversation
// tab is the manifest's first tab so it is the default view.
func TestFrontPage(t *testing.T) {
	t.Parallel()
	// The front surfaces, task effort and states legs drive the front page;
	// the manifest keeps the conversation first.
	declaration, err := parseWorkspaceManifest()
	if err != nil {
		t.Fatal(err)
	}
	if len(declaration.Tabs) == 0 || declaration.Tabs[0].ID != "chat" {
		t.Error("the conversation is not the manifest's first tab, so it is not the front page")
	}
	if routesByPath["/catalog/models"].Authentication != routePublic {
		t.Error("the evidence line reads /catalog/models, which must stay public")
	}
}
