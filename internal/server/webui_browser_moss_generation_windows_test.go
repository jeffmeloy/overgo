//go:build windows

package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"testing"

	"overgo/internal/latentimage"
	"overgo/internal/mediacapability"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSGeneration checks source controls and exact prompt transport in Chromium.
// The output is a stored fixture PNG and establishes no model quality claim.
func TestWebUIBrowserMOSSGeneration(t *testing.T) {
	requireWebUILane(t, "MOSS controls run through cmd/webui-lane")
	started := make(chan struct{})
	fixture := newMOSSGenerationFixture(t, func(_ context.Context, request latentimage.Request) error {
		if request.Prompt == "resource refusal" {
			return errors.New("image generation needs more device memory; release another model and retry")
		}
		if request.Prompt == "cancel" {
			close(started)
		}
		return nil
	})
	content, err := mediacapability.OutputContent(tinyPNG(t))
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	check(`(() => { location.hash='library'; return true; })()`)
	settle(`!!document.querySelector('#panel-library.active input[placeholder="model GGUF or directory on disk"]')`)
	settle(`overgo.api.inFlight()===0 && !document.querySelector('#panel-library .grid.stack tbody tr')`)
	check(`(() => {
  document.querySelector('#panel-library input[placeholder="model GGUF or directory on disk"]').value=` + strconv.Quote(fixture.root) + `;
  [...document.querySelectorAll('#panel-library button')].find(button=>button.textContent==='Add a local model').click();
  return true;
})()`)
	settle(`!!document.querySelector('.operation-chip.completed') && [...document.querySelectorAll('#panel-library button')].some(button=>button.textContent==='registered' && button.disabled)`)
	check(`(() => {
  [...document.querySelectorAll('#panel-library button')].find(button=>button.textContent==='Refresh catalog').click();
  return true;
})()`)
	settle(`document.querySelector('#panel-library').textContent.includes('1 model is ready to serve')`)
	check(`(() => { location.hash='chat'; return true; })()`)
	settle(`!!document.querySelector('.composer select[aria-label="mode"] option[value="image-gen"]')`)
	check(`(() => { const mode=document.querySelector('.composer select[aria-label="mode"]'); mode.value='image-gen'; mode.dispatchEvent(new Event('change',{bubbles:true})); return true; })()`)
	settle(`document.querySelector('select[aria-label="generation model"]')?.textContent.includes(` + strconv.Quote(filepath.Base(fixture.root)) + `) && !!document.querySelector('.composer .advanced-controls')`)
	check(`(() => {
  window.mossField=(name)=>[...document.querySelectorAll('.composer label.control')].find(label=>label.firstChild.textContent===name)?.querySelector('input,textarea,select');
  window.mossSend=(text)=>{const input=document.querySelector('.composer>textarea');input.value=text;input.dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('.send-button').click();};
  const advanced=document.querySelector('.composer .advanced-controls');
  return !advanced.open && mossField('width').value==='1024' && mossField('height').value==='1024' && mossField('steps').value==='50' && mossField('guidance scale').value==='3' && !mossField('mu') && !mossField('num train timesteps');
})()`)
	check(`(() => { const help=document.querySelector('.composer .control-help'); help.querySelector('summary').click(); return help.open && help.textContent.includes('canvas center') && help.textContent.includes('"bbox_format":"yxyx"'); })()`)
	check(`(() => {document.querySelector('.composer .advanced-controls summary').click();mossField('guidance scale').value='0';mossField('seed').value='42';return true;})()`)
	check(`(() => {mossSend('blue cube');return true;})()`)
	settle(`!!document.querySelector('.artifact.msg.media img') && document.querySelector('.artifact.msg.media img').naturalWidth===2 && !document.querySelector('.send-button').disabled`)
	check(`(async () => {const content=await overgo.api.blob(overgo.contentURL('` + content.Descriptor.ID.String() + `'));return content.type.includes('image/png') && content.size>0;})()`)
	const layout = `{"summary":{"caption":"蓝色方块"},"items":[{"label":"object","caption":"blue cube","bbox":[-128,-128,128,128]}],"bbox_format":"yxyx","bbox_coord_space":"abs_centralized"}`
	encoded, err := json.Marshal(layout)
	if err != nil {
		t.Fatal(err)
	}
	check(`(() => {mossField('prompt mode').value='layout';mossSend(` + string(encoded) + `);return true;})()`)
	settle(`document.querySelectorAll('.artifact.msg.media img').length===2 && !document.querySelector('.send-button').disabled`)
	inputs := fixture.requests()
	if len(inputs) != 2 {
		t.Fatal("request count", len(inputs))
	}
	for i, want := range []string{"blue cube", layout} {
		request := inputs[i]
		if request.Prompt != want || request.GuidanceScale == nil || *request.GuidanceScale != 0 || request.Seed != 42 {
			t.Fatal("typed request changed", request)
		}
	}
	if inputs[0].PromptMode != "" || inputs[1].PromptMode != "layout" {
		t.Fatal("prompt validation mode lost", inputs)
	}
	check(`(() => {mossField('width').value='16';mossSend(` + string(encoded) + `);return true;})()`)
	settle(`document.querySelector('.chat-log').textContent.includes('bbox exceeds the canvas') && !document.querySelector('.send-button').disabled`)
	check(`document.querySelectorAll('.artifact.msg.media img').length===2`)
	check(`(() => {mossField('prompt mode').value='text';mossField('width').value='0';mossSend('invalid size');return true;})()`)
	settle(`document.querySelector('.chat-log').textContent.includes('invalid extent') && !document.querySelector('.send-button').disabled`)
	check(`document.querySelectorAll('.artifact.msg.media img').length===2`)
	check(`(() => {mossField('width').value='1024';mossSend('resource refusal');return true;})()`)
	settle(`document.querySelector('.chat-log').textContent.includes('release another model and retry') && !document.querySelector('.send-button').disabled`)
	check(`document.querySelectorAll('.artifact.msg.media img').length===2`)
	check(`(() => {mossSend('cancel');return true;})()`)
	select {
	case <-started:
	case <-t.Context().Done():
		t.Fatal(t.Context().Err())
	}
	settle(`!document.querySelector('.stop-button').hidden`)
	check(`(() => {document.querySelector('.stop-button').click();return true;})()`)
	settle(`!!document.querySelector('.operation-chip.cancelled') && document.querySelector('.stop-button').hidden && !document.querySelector('.send-button').disabled`)
	check(`document.querySelectorAll('.artifact.msg.media img').length===2`)
	check(`(() => {mossSend('retry after cancellation');return true;})()`)
	settle(`document.querySelectorAll('.artifact.msg.media img').length===3 && !document.querySelector('.send-button').disabled`)
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-generation", "Empty catalog, local source import, registered candidate, activation, refreshed composer modes, model choice, native defaults, optional controls, layout mode with example and coordinate errors, exact Unicode prompt transport, PNG preview/download, invalid size, resource refusal, cancellation without output and retry pass. Execution and activation use hermetic fixtures; native quality is checked separately.")
}
