package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
	"testing"
	"time"
)

func TestWebUIBrowserSpeechTextImport(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": graphical speech form probe")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := &speechSurfaceGenerator{generationWorkspaceGenerator: &generationWorkspaceGenerator{fakeGenerator: &fakeGenerator{}, repository: store, run: testutil.ArtifactID(t, artifact.KindRun, "speech-surface"), capability: WorkflowCapability{
		Task: recipe.TaskSpeech, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "speech-surface"), Name: "Pocket-TTS (UI fixture)",
		Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
		Controls: []WorkflowControl{
			{Name: "text", Type: WorkflowControlText, Required: true},
			{Name: "max_frames", Type: WorkflowControlInteger, Required: true},
			{Name: "seed", Type: WorkflowControlInteger, Required: true},
			{Name: "voice", Type: WorkflowControlText, Required: true, Choices: []string{"alba", "marius"}},
			{Name: "temperature", Type: WorkflowControlNumber},
		},
	}}}
	generator.secondRecipe = testutil.ArtifactID(t, artifact.KindRecipe, "speech-surface-alternate")
	handler, err := New(Config{Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 60*time.Second, errors.New("speech surface probe did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle(`!!document.querySelector('.composer select[aria-label="mode"] option[value="speech"]')`)
	check(`(()=>{const mode=document.querySelector('.composer select[aria-label="mode"]');mode.value='speech';mode.dispatchEvent(new Event('change'));return true;})()`)
	settle(`!!document.querySelector('.mode-controls select[aria-label="voice"]')`)
	check(`(()=>{
 window.textInput=document.querySelector('.composer textarea');textInput.value='Keep this draft';textInput.dispatchEvent(new Event('input'));
 window.importFile=(bytes,type='text/plain')=>{const transfer=new DataTransfer();transfer.items.add(new File([bytes],'speech.txt',{type}));document.querySelector('.composer').dispatchEvent(new DragEvent('drop',{bubbles:true,dataTransfer:transfer}));};
 window.textAction=label=>{const button=[...document.querySelectorAll('.attachment-row button')].find(button=>button.textContent===label&&!button.hidden);if(!button||button.disabled)return false;button.focus();return true;};
 importFile(new Uint8Array([239,187,191,...new TextEncoder().encode('Café.\r\nRead every word.')]));return true;
 })()`)
	settle(`!!document.querySelector('.transcript-offer button')`)
	check(`textInput.value==='Keep this draft'&&document.querySelector('.send-button').disabled`)
	for _, viewport := range append(webuilane.ScreenViewports, webuilane.Viewport{Name: "phone-keyboard", Width: 390, Height: 480}) {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "speech-text-import-offer")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}
	check(`textAction('Insert')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`textInput.value==='Keep this draft\nCafé.\nRead every word.'&&!document.querySelector('.send-button').disabled`)
	check(`(async()=>{const link=document.querySelector('.attachment-info a');window.importID=new URL(link.href).searchParams.get('id');const blob=await overgo.api.blob('/artifacts/content?id='+encodeURIComponent(importID));const bytes=new Uint8Array(await blob.arrayBuffer());return bytes[0]===239&&bytes[1]===187&&bytes[2]===191;})()`)
	check(`textAction('Remove')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`(()=>{importFile('Replace this text.','');return true;})()`)
	settle(`!!document.querySelector('.transcript-offer button')`)
	check(`textAction('Replace')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`textInput.value==='Replace this text.'`)
	check(`textAction('Remove')`)
	pressKey(t, ctx, browser, "Enter", 13)
	for _, bytes := range []string{`new Uint8Array([255])`, `'   '`, `new Uint8Array([0])`} {
		check(`(()=>{importFile(` + bytes + `);return true;})()`)
		settle(`document.querySelector('.attachment-row')?.dataset.state==='error'`)
		check(`textInput.value==='Replace this text.'&&document.querySelector('.send-button').disabled`)
		check(`textAction('Remove')`)
		pressKey(t, ctx, browser, "Enter", 13)
	}
	check(`(()=>{window.sizeUpload=overgo.api.upload;window.sizeUploadCalls=0;overgo.api.upload=(...args)=>{sizeUploadCalls++;return sizeUpload(...args);};importFile('This is not a supported document.','application/pdf');return true;})()`)
	settle(`document.querySelector('.attachment-row')?.dataset.state==='error'`)
	check(`sizeUploadCalls===0&&textInput.value==='Replace this text.'&&document.querySelector('.attachment-row').textContent.includes('Choose a text file.')`)
	check(`textAction('Remove')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`(()=>{importFile(new Uint8Array(overgo.capabilities().media.max_media_bytes+1));return true;})()`)
	settle(`document.querySelector('.attachment-row')?.dataset.state==='error'`)
	check(`(()=>{overgo.api.upload=sizeUpload;return sizeUploadCalls===0&&textInput.value==='Replace this text.'&&document.querySelector('.attachment-row').textContent.includes('attachment limit');})()`)
	check(`textAction('Remove')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`(()=>{window.originalTextRead=File.prototype.arrayBuffer;File.prototype.arrayBuffer=function(){return new Promise(resolve=>window.finishTextRead=()=>resolve(originalTextRead.call(this)));};importFile('Cancel this read when switching tasks.');return true;})()`)
	settle(`typeof finishTextRead==='function'`)
	check(`(()=>{const mode=document.querySelector('.composer select[aria-label="mode"]');mode.value='chat';mode.dispatchEvent(new Event('change'));finishTextRead();File.prototype.arrayBuffer=originalTextRead;return textInput.value==='Replace this text.';})()`)
	settle(`document.querySelector('.attachment-row')?.dataset.state==='error'`)
	check(`textAction('Remove')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`(()=>{const mode=document.querySelector('.composer select[aria-label="mode"]');mode.value='speech';mode.dispatchEvent(new Event('change'));return true;})()`)
	settle(`!!document.querySelector('.mode-controls select[aria-label="voice"]')`)
	check(`(()=>{importFile('A long document preview.\n'.repeat(100));return true;})()`)
	settle(`!!document.querySelector('.transcript-offer details')`)
	check(`(()=>{document.querySelector('.transcript-offer summary').click();return true;})()`)
	if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), webuilane.Viewport{Name: "phone-keyboard", Width: 390, Height: 480}, "speech-long-preview"); err != nil || len(findings) != 0 {
		t.Fatalf("expanded document preview: %v %v", findings, err)
	}
	check(`(()=>{const offer=document.querySelector('.transcript-offer');return ['Insert','Replace','Discard'].every(label=>{const button=[...offer.querySelectorAll('button')].find(button=>button.textContent===label);const box=button.getBoundingClientRect();return box.width>0&&box.height>0&&button.contains(document.elementFromPoint(box.x+box.width/2,box.y+box.height/2));});})()`)
	check(`textAction('Discard')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`(()=>{importFile('A long document to edit.\n'.repeat(100));return true;})()`)
	settle(`!!document.querySelector('.transcript-offer button')`)
	check(`textAction('Replace')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`textInput.value==='A long document to edit.\n'.repeat(100)&&!textInput.readOnly`)
	if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), webuilane.Viewport{Name: "phone-keyboard", Width: 390, Height: 480}, "speech-long-edit"); err != nil || len(findings) != 0 {
		t.Fatalf("long document editing: %v %v", findings, err)
	}
	check(`(()=>{const button=document.querySelector('.send-button');const box=button.getBoundingClientRect();return box.width>0&&box.height>0&&button.contains(document.elementFromPoint(box.x+box.width/2,box.y+box.height/2));})()`)
	check(`textAction('Remove')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`(()=>{textInput.value='Replace this text.';textInput.dispatchEvent(new Event('input'));return true;})()`)
	check(`(()=>{importFile('Discard me.');return true;})()`)
	settle(`!!document.querySelector('.transcript-offer button')`)
	check(`textAction('Discard')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`!document.querySelector('.attachment-row')&&textInput.value==='Replace this text.'&&overgo.errors.length===0`)
	check(`(()=>{importFile('Preserve the draft across page lifecycle.');return true;})()`)
	settle(`!!document.querySelector('.transcript-offer button')`)
	check(`(()=>{dispatchEvent(new Event('pagehide'));dispatchEvent(new Event('pageshow'));return textInput.value==='Replace this text.'&&textAction('Retry');})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`!!document.querySelector('.transcript-offer button')`)
	check(`textAction('Discard')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`(()=>{window.originalUpload=overgo.api.upload;overgo.api.upload=(...args)=>new Promise(resolve=>window.finishTextUpload=()=>resolve(originalUpload(...args)));importFile('This late file must not replace my draft.');return true;})()`)
	settle(`typeof finishTextUpload==='function'`)
	check(`textAction('Remove')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`(()=>{finishTextUpload();overgo.api.upload=originalUpload;return textInput.value==='Replace this text.'&&!document.querySelector('.attachment-row');})()`)
	check(`(()=>{importFile('Send the complete edited text.');return true;})()`)
	settle(`!!document.querySelector('.transcript-offer button')`)
	check(`textAction('Replace')`)
	pressKey(t, ctx, browser, "Enter", 13)
	check(`(()=>{
 window.sourceID=new URL(document.querySelector('.attachment-info a').href).searchParams.get('id');
 textInput.value+=' An edit.';textInput.dispatchEvent(new Event('input'));
 const voice=document.querySelector('.mode-controls select[aria-label="voice"]');voice.value='alba';voice.dispatchEvent(new Event('change'));
 for(const [name,value] of [['max frames','120'],['seed','7']]){const control=[...document.querySelectorAll('[aria-label="Speech options"] .control')].find(label=>label.querySelector('span')?.textContent===name).querySelector('input');control.value=value;control.dispatchEvent(new Event('input'));}
 window.originalPost=overgo.api.post;overgo.api.post=(path,body,options)=>{if(path==='/generation/run')window.importRequest=body;return originalPost(path,body,options);};document.querySelector('.send-button').focus();return true;
 })()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`document.body.textContent.includes('speech fixture stopped before synthesis')&&textInput.value==='Send the complete edited text. An edit.'`)
	check(`importRequest.sources.length===1&&importRequest.sources[0]===sourceID&&importRequest.input.text==='Send the complete edited text. An edit.'&&overgo.errors.length===0`)
	if err := browser.Evaluate(ctx, `location.reload()`, nil); err != nil {
		t.Fatal(err)
	}
	settle(`document.querySelector('.composer textarea')?.value==='Send the complete edited text. An edit.'&&[...document.querySelectorAll('.attachment-row button')].some(button=>button.textContent==='Reload stored file')`)
	check(`(()=>{const button=[...document.querySelectorAll('.attachment-row button')].find(button=>button.textContent==='Reload stored file');button.focus();return true;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`!!document.querySelector('.transcript-offer button')`)
	check(`document.querySelector('.composer textarea').value==='Send the complete edited text. An edit.'&&overgo.errors.length===0`)
	t.Log("speech text import leg: real intake preserves BOM bytes; UTF-8 draft-safe insert/replace/discard and invalid input refusal; no speech completion claim")
}
