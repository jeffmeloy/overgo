package server

import (
	"io/fs"
	"regexp"
	"strings"
	"testing"
)

// TestWebUIClientHelpersHaveOneOwner holds the workbench's shared client
// helpers to boot.js and keeps the dead code the consolidation removed from
// returning: forty-nine failure banners composed errorBanner(friendlyError)
// by hand, five modules posted /operations/cancel themselves, evaluations
// and training each wrapped artifactLink, index.html carried an unused tab
// strip, compositions showed three inputs that could never be enabled, the
// stylesheet split its phone rules across two identical media blocks, three
// modules mixed tab and space indentation, and loading notes named routes.
func TestWebUIClientHelpersHaveOneOwner(t *testing.T) {
	t.Parallel()
	sources := webuiJavaScript(t)
	boot := sources["boot.js"]
	owned := []struct{ pattern, helper string }{
		{"/operations/cancel", "overgo.cancelOperation"},
		{"errorBanner(friendlyError(", "overgo.failure"},
		{"errorBanner(overgo.friendlyError(", "overgo.failure"},
	}
	for _, owner := range owned[:2] {
		if count := strings.Count(boot, owner.pattern); count != 1 {
			t.Errorf("boot.js defines %q %d times, want once (in %s)", owner.pattern, count, owner.helper)
		}
	}
	wrapper := regexp.MustCompile(`const artifactLink = \(`)
	leadingTab := regexp.MustCompile(`(?m)^ *\t`)
	routeCopy := regexp.MustCompile(`text: "loading /`)
	for name, source := range sources {
		if name != "boot.js" {
			for _, owner := range owned {
				if strings.Contains(source, owner.pattern) {
					t.Errorf("%s spells %q; call %s", name, owner.pattern, owner.helper)
				}
			}
		}
		if wrapper.MatchString(source) {
			t.Errorf("%s wraps artifactLink; call overgo.artifactLink(id, label, gallery)", name)
		}
		if leadingTab.MatchString(source) {
			t.Errorf("%s indents with tabs; the workbench indents with spaces", name)
		}
		if routeCopy.MatchString(source) {
			t.Errorf("%s names a route in its loading copy; say what is loading", name)
		}
	}
	if strings.Contains(sources["mod/compositions.js"], "disabled: true })") {
		t.Error("compositions.js shows an input that is never enabled")
	}
	for _, dead := range []string{"invalidateModel,\n", "evidenceLine, servedModel"} {
		if strings.Contains(boot, dead) {
			t.Errorf("boot.js exports %q, which no module or page test reads", strings.TrimSpace(dead))
		}
	}
	page, err := fs.ReadFile(webuiFS, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(page), `id="tabs"`) {
		t.Error("index.html keeps the unused tab strip")
	}
	style, err := fs.ReadFile(webuiFS, "style.css")
	if err != nil {
		t.Fatal(err)
	}
	if count := strings.Count(string(style), "@media (max-width:860px)"); count != 1 {
		t.Errorf("style.css holds %d phone-width blocks, want one", count)
	}
}
