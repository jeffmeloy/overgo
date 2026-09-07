package server

import (
	"io/fs"
	"strings"
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
	shell, err := fs.ReadFile(webuiFS, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{`class="shell front"`, `id="workbench-toggle"`, `id="model-evidence"`} {
		if !strings.Contains(string(shell), needle) {
			t.Errorf("index.html lacks %s", needle)
		}
	}
	sources := webuiJavaScript(t)
	boot := sources["boot.js"]
	for _, needle := range []string{"function setFront(", `"/catalog/models"`, "servedModel", "workbench-toggle", "model-evidence"} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot.js lacks the front-page wiring (%s)", needle)
		}
	}
	chat := sources["mod/chat.js"]
	for _, needle := range []string{"front-empty", "overgo.capabilities()", "overgo.servedModel()", "starters", "openPicker()", `"model-pill"`} {
		if !strings.Contains(chat, needle) {
			t.Errorf("chat.js lacks the getting-started card (%s)", needle)
		}
	}
	style, err := fs.ReadFile(webuiFS, "style.css")
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{".shell.front .sidebar .nav-group", ".front-empty"} {
		if !strings.Contains(string(style), needle) {
			t.Errorf("style.css lacks %s", needle)
		}
	}
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
