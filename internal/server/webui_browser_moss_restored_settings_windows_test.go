//go:build windows

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSRestoredSettingsRunAsChosen holds moss-page/restored (SQA-485). Settings restored with no event
// show and send what choosing them by hand does:
//   - A saved depth preset, reapplied after text to image was chosen, shows its depth control and sends its picture.
//   - A remembered builder run, reloaded, shows its layout and sends it.
//
// Restoring had set each field's text and left what each choice shows as it was, so the depth control stayed hidden
// and its picture unsent, and the reloaded layout stayed hidden and was refused.
func TestWebUIBrowserMOSSRestoredSettingsRunAsChosen(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the restored-settings leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
	// A depth section runs through the adapters it stacks, which the stand-in model holds empty.
	testcheck.NoError(t, os.MkdirAll(filepath.Join(fixture.root, "adapters"), 0o755))
	for _, name := range []string{"8-steps-dmd2-lora", "moss_image_sft", "moss-control"} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(fixture.root, "adapters", name+".safetensors"), nil, 0o600))
	}
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	started := tinyPNG(t).Data
	picture := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, started))(t), Size: uint64(len(started)), MediaType: media.PNGMediaType}, Data: started}
	testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/restored-picture", Artifacts: []artifact.Descriptor{picture.Descriptor}, Contents: []artifact.Content{picture}}))(t)
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	bench := `(() => {
  const panel=document.querySelector('#panel-generation');
  const label=(name, within)=>[...(within||panel).querySelectorAll('label.control')].find(found=>found.querySelector('span').textContent===name && (within || !found.closest('fieldset[data-control]')));
  window.bench={panel, label,
    group:(name)=>panel.querySelector('fieldset[data-control="'+name+'"]'),
    shown:(node)=>!!node && !node.closest('[hidden]'),
    choose:(select, value)=>{select.value=value; select.dispatchEvent(new Event('change',{bubbles:true}));},
    section:()=>panel.querySelector('.workflow-task + .control-grid').firstElementChild.querySelector('select'),
    set:(input, value)=>{input.value=value; input.dispatchEvent(new Event('input',{bubbles:true}));},
    button:(text)=>[...panel.querySelectorAll('button')].find(button=>button.textContent===text),
    status:()=>panel.querySelector('[aria-label="run status"]'),
    run:()=>{panel.querySelectorAll('.workbench-output').forEach(shown=>shown.remove()); bench.status().replaceChildren(); bench.button('Run').click();},
    ran:()=>!!panel.querySelector('.workbench-output') && !bench.button('Run').disabled,
    // A run ends drawn or refused; a wait reads both, so a refusal fails the check rather than holding the wait.
    ended:()=>bench.ran() || (!bench.button('Run').disabled && bench.status().textContent.trim()!==''),
    drawn:()=>bench.ran() || (()=>{ throw new Error('the run was refused: '+bench.status().textContent); })(),
  };
  return true;
})()`
	settle(`!!document.querySelector('#panel-generation.active fieldset[data-control="control"]')`)
	check(bench)
	// A depth preset with its control's picture is saved, text to image is chosen, and the preset is chosen again.
	check(`(() => {
  bench.choose(bench.section(), 'ultra-depth-control');
  const picture=bench.label('picture', bench.group('control')).querySelector('input[type=hidden]');
  picture.value=` + strconv.Quote(picture.Descriptor.ID.String()) + `; picture.dispatchEvent(new Event('change',{bubbles:true}));
  bench.set(bench.panel.querySelector('[aria-label="Name for these settings"]'), 'depth');
  bench.button('Save settings').click();
  bench.choose(bench.section(), 'ultra-text-to-image');
  if (bench.shown(bench.group('control'))) throw new Error('text to image shows the depth control');
  bench.choose(bench.panel.querySelector('select[aria-label="saved settings"]'), 'depth');
  if (!bench.shown(bench.group('control'))) throw new Error('the depth preset restored leaves its control hidden');
  bench.set(bench.label('prompt').querySelector('textarea'), 'a lighthouse in fog');
  bench.run();
  return true;
})()`)
	settle(`bench.ended()`)
	check(`bench.drawn()`)
	if inputs := fixture.requests(); len(inputs) != 1 || inputs[0].Preset != "ultra-depth-control" || inputs[0].Control == nil || inputs[0].Control.Picture != picture.Descriptor.ID.String() {
		t.Fatalf("the restored depth preset ran %+v; want its section and its control's picture", inputs)
	}
	// A builder run is remembered, and the page reloaded restores it with its layout shown and sent.
	check(`(() => {
  bench.choose(bench.section(), '');
  bench.choose(bench.label('prompt mode').querySelector('select'), 'builder');
  bench.group('layout').value=JSON.stringify({high_level_description:'a fox at dawn',regions:[{x:0.1,y:0.2,w:0.4,h:0.5,desc:'a fox'}]});
  bench.run();
  return true;
})()`)
	settle(`bench.ended()`)
	check(`bench.drawn()`)
	testcheck.NoError(t, browser.Reload(ctx))
	settle(`!!document.querySelector('#panel-generation.active fieldset[data-control="layout"]')`)
	check(bench)
	check(`(() => {
  if (!bench.shown(bench.group('layout'))) throw new Error('the remembered builder run reloaded leaves its layout hidden');
  bench.run();
  return true;
})()`)
	settle(`bench.ended()`)
	check(`bench.drawn()`)
	inputs := fixture.requests()
	if last := inputs[len(inputs)-1]; last.PromptMode != "builder" || last.Layout == nil || last.Layout.Description != "a fox at dawn" {
		t.Fatalf("the reloaded builder run asked %+v; want its layout", last)
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-restored-settings", "A saved depth preset reapplied shows its control and sends its picture, and a remembered builder run reloaded shows and sends its layout.")
}
