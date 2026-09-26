package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestWebUITabsLoadThroughOneResource holds every workspace tab to the one
// lifecycle the shell hands its mount: reads, runtime subscriptions, page
// listeners, forms and conversation surfaces open through the workspace and
// end with it. Each tab once unsubscribed, removed listeners, disposed forms
// and returned its own cleanup, and a tab that forgot one leaked it into the
// next mount.
func TestWebUITabsLoadThroughOneResource(t *testing.T) {
	t.Parallel()
	// The remount and one-event-stream legs drive the workspace lifecycle;
	// the modules keep the retired per-tab handling banned.
	modules, err := filepath.Glob(filepath.Join("webui", "mod", "*.js"))
	if err != nil || len(modules) == 0 {
		t.Fatalf("tab modules = %v, %v", modules, err)
	}
	// Each is the per-tab handling the workspace replaced.
	retired := []string{
		"runtimeEvents.subscribe", "removeEventListener", ".dispose()", "onActivate", "onDeactivate",
		"this.dispose", "return () =>", "return unsubscribe", "overgo.reporter(",
	}
	tabs := 0
	for _, module := range modules {
		source, err := os.ReadFile(module)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(source), "registerTab(") {
			continue
		}
		tabs++
		for _, handling := range retired {
			if strings.Contains(string(source), handling) {
				t.Errorf("%s keeps its own lifecycle handling %q; open it through the workspace", filepath.Base(module), handling)
			}
		}
		// The mount's overgo is its workspace: a page-wide hook set on it (chat's Escape stop once was)
		// reaches nothing outside the mount, so page hooks are set on window.overgo.
		for line := range strings.SplitSeq(string(source), "\n") {
			field, assigned := strings.CutPrefix(strings.TrimSpace(line), "overgo.")
			if name, _, found := strings.Cut(field, " = "); assigned && found && !strings.ContainsAny(name, "(.[ ") {
				t.Errorf("%s sets overgo.%s on its workspace; set page hooks on window.overgo", filepath.Base(module), name)
			}
		}
	}
	if tabs == 0 {
		t.Fatal("no tab module registers a tab; the census lost its sources")
	}
}
