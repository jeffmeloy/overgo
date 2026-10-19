//go:build windows

package server

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSGeneration checks source controls and exact prompt transport in Chromium.
// The output is a stored fixture PNG and establishes no model quality claim.
func TestWebUIBrowserMOSSGeneration(t *testing.T) {
	requireWebUILane(t, "MOSS controls run through cmd/webui-lane")
	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	handler, workspace, _, _ := nativeMediaProtocolFixture(t)
	catalog := BindGenerationCatalog(mediacapability.Catalog, mediacapability.Controls, mediacapability.OutputContent)
	controls, refusal := catalog.Controls(modelrecipe.ModuleLatentImagePrepare, root)
	if refusal != "" {
		t.Fatal(refusal)
	}
	workspace.capabilities = workspace.capabilities[:1]
	workspace.capabilities[0].Controls = controls
	workspace.capabilities[0].Name = "MOSS-Image-SFT"
	content, err := mediacapability.OutputContent(tinyPNG(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(t.Context(), handler.repository, artifact.Batch{Key: "fixture/moss/browser-png", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	prior := workspace.completion[recipe.TaskImageGen]
	workspace.completion[recipe.TaskImageGen] = operation.Completion{Run: prior.Run, Outputs: []artifact.ID{content.Descriptor.ID}}
	server := httptest.NewServer(handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!document.querySelector('.composer select[aria-label="mode"] option[value="image-gen"]')`)
	check(`(() => { const mode=document.querySelector('.composer select[aria-label="mode"]'); mode.value='image-gen'; mode.dispatchEvent(new Event('change',{bubbles:true})); return true; })()`)
	settle(`document.querySelector('select[aria-label="generation model"]')?.textContent.includes('MOSS-Image-SFT') && !!document.querySelector('.composer .advanced-controls')`)
	check(`(() => {
  window.mossField=(name)=>[...document.querySelectorAll('.composer label.control')].find(label=>label.firstChild.textContent===name)?.querySelector('input,textarea');
  window.mossSend=(text)=>{const input=document.querySelector('.composer>textarea');input.value=text;input.dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('.send-button').click();};
  const advanced=document.querySelector('.composer .advanced-controls');
  return !advanced.open && mossField('width').value==='1024' && mossField('height').value==='1024' && mossField('steps').value==='50' && mossField('guidance scale').value==='3' && !mossField('mu') && !mossField('num train timesteps');
})()`)
	check(`(() => {document.querySelector('.composer .advanced-controls summary').click();mossField('guidance scale').value='0';mossField('seed').value='42';return true;})()`)
	check(`(() => {mossSend('blue cube');return true;})()`)
	settle(`!!document.querySelector('.artifact.msg.media img') && document.querySelector('.artifact.msg.media img').naturalWidth===2 && !document.querySelector('.send-button').disabled`)
	check(`(async () => {const content=await overgo.api.blob(overgo.contentURL('` + content.Descriptor.ID.String() + `'));return content.type.includes('image/png') && content.size>0;})()`)
	const layout = "图形布局\n<objects>\nblue cube (0, 0, 16, 16)\n</objects>"
	encoded, err := json.Marshal(layout)
	if err != nil {
		t.Fatal(err)
	}
	check(`(() => {mossSend(` + string(encoded) + `);return true;})()`)
	settle(`document.querySelectorAll('.artifact.msg.media img').length===2 && !document.querySelector('.send-button').disabled`)
	workspace.mu.Lock()
	inputs := append([]json.RawMessage(nil), workspace.inputs...)
	workspace.mu.Unlock()
	if len(inputs) != 2 {
		t.Fatal("request count", len(inputs))
	}
	for i, want := range []string{"blue cube", layout} {
		var request struct {
			Prompt   string
			Guidance float32 `json:"guidance_scale"`
			Seed     int64
		}
		if err := json.Unmarshal(inputs[i], &request); err != nil || request.Prompt != want || request.Guidance != 0 || request.Seed != 42 {
			t.Fatal("typed request changed", request, err)
		}
	}
	webuilane.Leg(t, "moss-generation", "Source defaults, collapsed optional controls, explicit zero guidance, Unicode layout transport, stored PNG preview and content download pass. Library activation and cancellation have separate operation tests; no full journey or quality claim.")
}
