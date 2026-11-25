package webuilane

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestEveryAttachInABrowserCheckIsTheComposersOwn holds journey-attach/do: a browser check that attaches a file picks
// the composer's own Attach input by its name (aria-label attach), never the first file input in the composer, which
// since moss-page/pictures may be a picture slot's chooser. TestModelJourneyFirstRun's attach helper picked the VQA
// image slot's chooser so, the picture went into the slot and no attachment row was drawn, and its wait held the
// model-journey lane 59 minutes on 2026-11-24 and again from 18:38.
func TestEveryAttachInABrowserCheckIsTheComposersOwn(t *testing.T) {
	t.Parallel()
	const named = "[aria-label=attach]"
	var unnamed []string
	for _, directory := range []string{filepath.Join("..", "server"), ".", filepath.Join("..", "audioparity")} {
		sources := testcheck.Must(filepath.Glob(filepath.Join(directory, "*_test.go")))(t)
		for _, source := range sources {
			// This check names the selectors it looks for, and attaches nothing.
			if filepath.Base(source) == "attach_selector_test.go" && directory == "." {
				continue
			}
			text := string(testcheck.Must(os.ReadFile(source))(t))
			for _, picked := range []string{".composer input[type=file]", `.composer input[type="file"]`} {
				for rest := text; ; {
					at := strings.Index(rest, picked)
					if at < 0 {
						break
					}
					rest = rest[at+len(picked):]
					if !strings.HasPrefix(rest, named) {
						unnamed = append(unnamed, filepath.Base(source))
					}
				}
			}
		}
	}
	if len(unnamed) > 0 {
		t.Errorf("browser checks pick the composer's first file input, not its Attach by name (%s): %v", named, unnamed)
	}
	// The name they pick by is the composer's own: its Attach input carries it.
	composer := string(testcheck.Must(os.ReadFile(filepath.Join("..", "server", "webui", "composer.js")))(t))
	for line := range strings.Lines(composer) {
		if strings.Contains(line, `el("input", { type: "file"`) && !strings.Contains(line, `"aria-label": "attach"`) {
			t.Errorf("the composer's file input is not named attach, the name the checks pick it by: %s", strings.TrimSpace(line))
		}
	}
	if !strings.Contains(composer, `type: "file", hidden: true, "aria-label": "attach"`) {
		t.Error("the composer holds no file input named attach")
	}
}
