package server

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserExportSizePreview holds the size an export and a merge show
// before they run. On the Export tab, with a model, a type and an output
// chosen, the form shows the bytes the export would write; another type shows
// another size and the first type the first size again; and the file a run
// then writes is of exactly the bytes shown. Model Builder shows a merge's
// size the same way, and the merged file is of those bytes. The form with its
// size showing has no layout finding at desktop and phone widths in either
// colour scheme.
func TestWebUIBrowserExportSizePreview(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the export size leg uses Chromium through cmd/webui-lane")
	preview := newSizePreview(t, sizedWriters(), false)
	server := httptest.NewServer(preview.handler)
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	// shown is the size a tab's form shows, or nothing while it shows none.
	shown := func(panel string) string {
		return `(document.querySelector('#panel-` + panel + ` [aria-label="output size"]') || { textContent: '' }).textContent`
	}
	// listed opens a tab and chooses the capability listed under the words given.
	listed := func(panel, words string) {
		t.Helper()
		check(`(() => { location.hash = '` + panel + `'; return true; })()`)
		settle(`[...document.querySelectorAll('#panel-` + panel + `.active select[aria-label="capability"] option')].some(option => option.textContent === '` + words + `')`)
		check(`(() => {
  const capability = document.querySelector('#panel-` + panel + ` select[aria-label="capability"]');
  capability.value = [...capability.options].find(option => option.textContent === '` + words + `').value;
  capability.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`)
	}
	// entered sets entries of a tab's form as a person does: a choice by its label, a typed entry by the start of its name.
	entered := func(panel string, entries ...[3]string) {
		t.Helper()
		var script strings.Builder
		script.WriteString(`(() => {
  const panel = document.querySelector('#panel-` + panel + `');
  const typed = (name) => [...panel.querySelectorAll('.control-grid label.control')].find(label => label.querySelector('span').textContent.startsWith(name)).querySelector('input, textarea');
`)
		for _, entry := range entries {
			target := `typed('` + entry[1] + `')`
			if entry[0] == "change" {
				target = `panel.querySelector('select[aria-label="` + entry[1] + `"]')`
			}
			script.WriteString(`  { const field = ` + target + `; field.value = '` + entry[2] + `'; field.dispatchEvent(new Event('` + entry[0] + `', { bubbles: true })); }
`)
		}
		check(script.String() + `  return true;
})()`)
	}
	// written runs a tab's form, waits for it to complete, and holds the size it showed, kept under a name, to the file it wrote.
	written := func(panel, kept, file string) {
		t.Helper()
		check(`(() => { [...document.querySelectorAll('#panel-` + panel + ` button')].find(button => button.textContent === 'Run').click(); return true; })()`)
		settle(`document.querySelector('#panel-` + panel + ` [aria-label="run status"]').textContent.startsWith('completed / ')`)
		info, err := os.Stat(file)
		if err != nil {
			t.Fatal(err)
		}
		var text string
		if err := browser.Evaluate(ctx, `window.`+kept, &text); err != nil {
			t.Fatal(err)
		}
		if want := strconv.FormatInt(info.Size(), 10) + " bytes"; !strings.Contains(strings.ReplaceAll(text, ",", ""), want) {
			t.Fatalf("the form showed %q before the run, and the run wrote %s of %s", text, filepath.Base(file), want)
		}
	}

	// An export: a size for the type chosen, another for another type, the first again for the first.
	listed("export-jobs", "export / hermetic.gguf")
	entered("export-jobs", [3]string{"change", "type", "f16"}, [3]string{"input", "output", "sized"})
	settle(shown("export-jobs") + ` !== ''`)
	check(`(() => { window.sizeHalf = ` + shown("export-jobs") + `; return true; })()`)
	entered("export-jobs", [3]string{"change", "type", "q8_0"})
	settle(shown("export-jobs") + ` !== '' && ` + shown("export-jobs") + ` !== window.sizeHalf`)
	entered("export-jobs", [3]string{"change", "type", "f16"})
	settle(shown("export-jobs") + ` === window.sizeHalf`)
	for _, scheme := range webuilane.ColourSchemes {
		if err := browser.SetColorScheme(ctx, scheme); err != nil {
			t.Fatal(err)
		}
		for _, viewport := range webuilane.ScreenViewports {
			findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "export-size-"+scheme)
			if err != nil || len(findings) != 0 {
				t.Fatalf("the Export tab with its size showing, at %s in the %s scheme: %v %v", viewport.Name, scheme, findings, err)
			}
		}
	}
	written("export-jobs", "sizeHalf", filepath.Join(preview.root, "sized", "hermetic.f16.gguf"))

	// A merge: its size once the fine-tuned model, its scale and the output are in.
	listed("model-builder", "merge / hermetic.gguf")
	entered("model-builder", [3]string{"change", "fine tuned", "tuned.gguf"}, [3]string{"input", "scale", "0.5"}, [3]string{"input", "output", "blended"})
	settle(shown("model-builder") + ` !== ''`)
	check(`(() => { window.sizeMerged = ` + shown("model-builder") + `; return true; })()`)
	written("model-builder", "sizeMerged", filepath.Join(preview.root, "blended", "hermetic.merged.gguf"))
	webuilane.Leg(t, "export size leg", "the Export tab showed the bytes an export would write, another size for another type and the first again for the first, and Model Builder the bytes of a merge; each run then wrote a file of exactly the bytes shown")
}
