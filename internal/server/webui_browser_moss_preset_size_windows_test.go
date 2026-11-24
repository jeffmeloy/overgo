//go:build windows

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"overgo/internal/latentimage"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSPresetGivesBackTheSizeSetByHand holds moss-workflow-surface/preset-size (SQA-462): on MOSS's
// generation page, a preset that sets the picture's size is chosen and cleared, the size is set by hand, and the preset
// is chosen and cleared again; each clearing gives back the size the form held when the preset was chosen, and the run
// sends the size set by hand. The page had kept the size it held at a preset's first choice alone, so the second
// clearing gave back the first size, and the run asked four times the pixels set (1024 by 1024 for 512 by 512). One
// handler (workflow.js presetSettings) fills every form's controls from its preset.
func TestWebUIBrowserMOSSPresetGivesBackTheSizeSetByHand(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the preset size leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
	testcheck.NoError(t, os.MkdirAll(filepath.Join(fixture.root, "adapters"), 0o755))
	for _, name := range []string{"8-steps-dmd2-lora", "moss_image_sft", "moss-control"} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(fixture.root, "adapters", name+".safetensors"), nil, 0o600))
	}
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	// The preset is the first, by name, that sets both the width and the height.
	settings := testcheck.Must(latentimage.PresetSettings(fixture.root))(t)
	var preset string
	var size struct {
		Width  int `json:"width"`
		Height int `json:"height"`
	}
	for _, name := range slices.Sorted(func(yield func(string) bool) {
		for name := range settings {
			if !yield(name) {
				return
			}
		}
	}) {
		testcheck.NoError(t, json.Unmarshal(settings[name], &size))
		if size.Width != 0 && size.Height != 0 {
			preset = name
			break
		}
	}
	if preset == "" {
		t.Fatal("no MOSS preset sets the picture's width and height")
	}
	if size.Width == 512 || size.Height == 512 || size.Width == 1024 || size.Height == 1024 {
		t.Fatalf("%s sets %d by %d, a size the test sets by hand", preset, size.Width, size.Height)
	}

	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-generation.active fieldset[data-control="control"]')`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  window.bench={
    panel,
    field:(name)=>[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent===name && !label.closest('fieldset[data-control]')).querySelector('input,select,textarea'),
    choose:(name)=>{const preset=bench.field('section');preset.value=name;preset.dispatchEvent(new Event('change',{bubbles:true}));},
    set:(name,value)=>{const field=bench.field(name);field.value=value;field.dispatchEvent(new Event('input',{bubbles:true}));},
    size:()=>bench.field('width').value+' by '+bench.field('height').value,
    status:()=>bench.panel.querySelector('[aria-label="run status"]'),
    button:()=>[...bench.panel.querySelectorAll('button')].find(button=>button.textContent==='Run'),
    ran:()=>!!bench.panel.querySelector('.workbench-output') && !bench.button().disabled,
    refused:()=>!bench.button().disabled && !bench.panel.querySelector('.workbench-output') && bench.status().textContent.trim()!=='',
  };
  panel.querySelector('.advanced-controls').open=true;
  return [...bench.field('section').options].some(option=>option.value==='');
})()`)
	quoted := string(testcheck.Must(json.Marshal(preset))(t))
	presetSize := strconv.Itoa(size.Width) + " by " + strconv.Itoa(size.Height)
	shows := func(when, want string) {
		t.Helper()
		check(`bench.size()===` + strconv.Quote(want) + ` || (() => { throw new Error(` + strconv.Quote(when+": the form shows ") + `+bench.size()+` + strconv.Quote(", not "+want) + `); })()`)
	}
	check(`(() => { bench.set('width','1024'); bench.set('height','1024'); bench.choose(` + quoted + `); return true; })()`)
	shows("the preset chosen", presetSize)
	check(`(() => { bench.choose(''); return true; })()`)
	shows("the preset cleared", "1024 by 1024")
	check(`(() => { bench.set('width','512'); bench.set('height','512'); bench.choose(` + quoted + `); return true; })()`)
	shows("the preset chosen again", presetSize)
	check(`(() => { bench.choose(''); return true; })()`)
	shows("the preset cleared again", "512 by 512")
	check(`(() => { bench.set('prompt','a red fox'); bench.button().click(); return true; })()`)
	settle(`bench.ran() || bench.refused()`)
	check(`bench.ran() || (() => { throw new Error('the run was refused: ' + bench.status().textContent); })()`)
	inputs := fixture.requests()
	if len(inputs) == 0 {
		t.Fatal("the run reached no session")
	}
	if sent := inputs[len(inputs)-1]; sent.Width != 512 || sent.Height != 512 || sent.Preset != "" {
		t.Errorf("the run sent %d by %d under preset %q; want the 512 by 512 set by hand, no preset", sent.Width, sent.Height, sent.Preset)
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-preset-size", "Clearing a MOSS preset gives back the size the form held when it was chosen, however often a preset is chosen and cleared, and the run sends the size set by hand.")
}
