package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserDraftLifecycle(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": draft lifecycle runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Stored answer"}}))
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 45*time.Second, errors.New("draft lifecycle journey did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(predicate string) {
		t.Helper()
		if err := browser.Eventually(ctx, predicate); err != nil {
			var page string
			_ = browser.Evaluate(t.Context(), `JSON.stringify({text:document.body.innerText,errors:overgo.errors})`, &page)
			t.Fatalf("%v; page: %s", err, page)
		}
	}
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle(`!!document.querySelector('#panel-chat .composer')`)
	// A draft written before model selection must not become this recipe's draft.
	check(`(() => {sessionStorage.setItem('overgo.draft:'+JSON.stringify(['unassigned','']),JSON.stringify({text:'Unassigned speech draft'}));window.previousDraftComposer=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==previousDraftComposer && document.querySelector('.composer textarea').value===''`)
	check(`JSON.parse(sessionStorage.getItem('overgo.draft:'+JSON.stringify(['unassigned','']))).text==='Unassigned speech draft'`)
	check(`(() => {
   const cap=overgo.capabilities();
   window.draftInput=()=>document.querySelector('#panel-chat .composer textarea');
   window.draftField=name=>document.querySelector('[data-draft-setting="'+name+'"]');
   window.draftType=(field,value)=>{field.value=value;field.dispatchEvent(new Event('input',{bubbles:true}));};
   draftType(draftInput(),'Keep the new conversation draft');draftType(draftField('system'),'Be concise');draftType(draftField('temperature'),'0.25');draftType(draftField('tokens'),'4');
   const files=new DataTransfer();files.items.add(new File(['private file contents'], 'notes.txt',{type:'text/plain'}));
   const picker=document.querySelector('.composer input[type=file]');picker.files=files.files;picker.dispatchEvent(new Event('change'));
   return !!cap.recipe && !!cap.model && cap.max_output_tokens===8 && Number(draftField('tokens').max)===8;
 })()`)
	settle(`document.querySelector('.attachment-strip').textContent.includes('notes.txt') && !!document.querySelector('.attachment-row[data-state=ready]')`)
	check(`(() => {
   const stored=Object.keys(sessionStorage).filter(key=>key.startsWith('overgo.draft:')).map(key=>sessionStorage.getItem(key)).join('');
   window.draftOriginalGet=overgo.api.get;
   overgo.api.get=async function(path,options) {if(path.startsWith('/interactions/messages'))return {response:'alpha',messages:[]};return draftOriginalGet.call(this,path,options);};
   overgo.openConversation({root:'alpha',latest:'alpha',model:overgo.capabilities().model});
   return !stored.includes('private file contents') && !stored.includes('data:') && stored.includes('notes.txt');
 })()`)
	settle(`!!document.querySelector('.composer textarea') && document.querySelector('.composer textarea').value==='' && !document.querySelector('.attachment-strip').textContent.includes('notes.txt')`)
	check(`(() => {draftType(draftInput(),'Alpha draft');window.previousDraftComposer=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
	settle(`document.querySelector('.composer')!==previousDraftComposer && draftInput().value==='Keep the new conversation draft'`)
	check(`(() => {
    const files=new DataTransfer();files.items.add(new File(['private file contents'],'notes.txt',{type:'text/plain'}));
    const picker=document.querySelector('.composer input[type=file]');picker.files=files.files;picker.dispatchEvent(new Event('change'));return true;
  })()`)
	settle(`!document.querySelector('.send-button').disabled && document.querySelectorAll('.attachment-strip > span').length===1`)
	check(`(() => {window.previousDraftComposer=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
	settle(`document.querySelector('.composer')!==previousDraftComposer && document.querySelector('.attachment-strip').textContent.includes('Reattach')`)
	check(`(() => {
   if(!document.querySelector('.send-button').disabled || !document.querySelector('.attachment-strip').textContent.includes('Reattach'))return false;
   document.querySelector('[aria-label="Remove notes.txt"]').click();
   return !document.querySelector('.send-button').disabled && draftField('system').value==='Be concise' && draftField('temperature').value==='0.25' && draftField('tokens').value==='4';
 })()`)
	// Reload the actual page: module memory is gone, session storage must restore it.
	if err := browser.Call(ctx, "Page.reload", map[string]any{}, nil); err != nil {
		t.Fatal(err)
	}
	settle(`!!document.querySelector('.composer textarea') && document.querySelector('.composer textarea').value==='Keep the new conversation draft'`)
	check(`(() => {
   window.draftInput=()=>document.querySelector('#panel-chat .composer textarea');window.draftField=name=>document.querySelector('[data-draft-setting="'+name+'"]');
   window.draftType=(field,value)=>{field.value=value;field.dispatchEvent(new Event('input',{bubbles:true}));};
   window.draftOriginalGet=overgo.api.get;
   window.draftManifest=null;
   return draftField('system').value==='Be concise' && draftField('temperature').value==='0.25' && draftField('tokens').value==='4';
 })()`)
	// The settings dialog keeps focus only when its replaced field owned focus.
	check(`(() => {document.querySelector('#settings-toggle').click();draftField('system').focus();window.oldDraftSystem=draftField('system');document.querySelector('#api-key').dispatchEvent(new Event('change'));return true;})()`)
	settle(`document.querySelector('[data-draft-setting="system"]')!==oldDraftSystem && document.activeElement===document.querySelector('[data-draft-setting="system"]')`)
	check(`(() => {document.querySelector('#settings-dialog').close();return true;})()`)
	// Controlled declarations exercise delayed/failed model loading without a second GPU model.
	check(`(async () => {
   draftManifest=await draftOriginalGet('/workspace/manifest');window.currentDraftManifest=draftManifest;
   window.otherDraftManifest=JSON.parse(JSON.stringify(draftManifest));otherDraftManifest.model.recipe='draft-second-recipe';otherDraftManifest.model.model='draft-second-model';otherDraftManifest.model.id='draft-second';otherDraftManifest.model.max_output_tokens=2;otherDraftManifest.model.generation.max_tokens=2;otherDraftManifest.model.modes.find(mode=>mode.id==='embeddings').enabled=true;
   window.failDraftSwap=false;window.delayDraftMount=false;
   overgo.api.get=async function(path,options) {
     if(path==='/catalog/models')return {models:[{model:otherDraftManifest.model.model,recipe:otherDraftManifest.model.recipe,location:'second.gguf',present:true},{model:draftManifest.model.model,recipe:draftManifest.model.recipe,location:'first.gguf',present:true}]};
     if(path.startsWith('/health?swap=')){if(failDraftSwap)throw new Error('controlled swap failure');await new Promise(resolve=>window.releaseDraftSwap=resolve);currentDraftManifest=path.includes('second')?otherDraftManifest:draftManifest;return {};}
     if(path==='/workspace/manifest')return currentDraftManifest;
     if(path==='/generation/capabilities')return [];
     if(path==='/agents' && delayDraftMount){await new Promise(resolve=>window.releaseDraftMount=resolve);return [];}
     return draftOriginalGet.call(this,path,options);
   };
   document.querySelector('#model-pill').click();return true;
 })()`)
	settle(`document.querySelectorAll('[data-serve]').length===2`)
	check(`(() => {delayDraftMount=true;document.querySelector('[data-serve]').click();draftType(draftInput(),'Typing during the switch');return overgo.modelSwitching() && document.querySelector('.send-button').disabled && [...document.querySelectorAll('[data-serve]')].every(button=>button.disabled);})()`)
	settle(`typeof releaseDraftSwap==='function'`)
	check(`(() => {releaseDraftSwap();return true;})()`)
	settle(`typeof releaseDraftMount==='function'`)
	check(`(() => {if(!overgo.modelSwitching() || !document.querySelector('dialog[aria-label="Choose a model"][open]'))return false;delayDraftMount=false;releaseDraftMount();return true;})()`)
	settle(`!overgo.modelSwitching() && !!document.querySelector('.composer textarea') && !document.querySelector('dialog[aria-label="Choose a model"]')`)
	check(`(() => {draftType(draftInput(),'Second model draft');draftType(draftField('tokens'),'99');const mode=document.querySelector('.composer select[aria-label="mode"]');mode.value='embeddings';mode.dispatchEvent(new Event('change'));return draftField('system').value==='' && Number(draftField('tokens').max)===2;})()`)
	// Invalid restored limits clamp to the recipe bound rather than reaching the server.
	check(`(() => {window.previousDraftComposer=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
	settle(`document.querySelector('.composer')!==previousDraftComposer && document.querySelector('[data-draft-setting="tokens"]').value==='2' && document.querySelector('.composer select[aria-label="mode"]').value==='embeddings'`)
	check(`(() => {failDraftSwap=true;document.querySelector('#model-pill').click();return true;})()`)
	settle(`document.querySelectorAll('[data-serve]').length===2`)
	check(`(() => {document.querySelectorAll('[data-serve]')[1].click();return true;})()`)
	settle(`!overgo.modelSwitching() && document.body.textContent.includes('controlled swap failure') && draftInput().value==='Second model draft'`)
	check(`(() => {document.querySelector('dialog[aria-label="Choose a model"] button:last-child').click();failDraftSwap=false;document.querySelector('#model-pill').click();return true;})()`)
	settle(`document.querySelectorAll('[data-serve]').length===2`)
	check(`(() => {window.releaseDraftSwap=null;document.querySelectorAll('[data-serve]')[1].click();return true;})()`)
	settle(`typeof releaseDraftSwap==='function'`)
	check(`(() => {releaseDraftSwap();return true;})()`)
	settle(`!overgo.modelSwitching() && draftInput().value==='Typing during the switch'`)
	// An older rejected mount cannot replace the newer conversation.
	check(`(() => {
   let first=true;overgo.api.get=async function(path,options){if(path==='/agents'){if(first){first=false;return [{state:'active',name:'delayed'}];}return [];}if(path==='/agent/tools')return new Promise((resolve,reject)=>window.rejectDraftMount=reject);return draftOriginalGet.call(this,path,options);};
   overgo.openConversation(null);return true;
 })()`)
	settle(`typeof rejectDraftMount==='function'`)
	check(`(() => {overgo.openConversation(null);return true;})()`)
	settle(`!!document.querySelector('.composer textarea') && draftInput().value==='Typing during the switch'`)
	check(`(async () => {rejectDraftMount(new Error('obsolete mount failure'));await new Promise(resolve=>requestAnimationFrame(resolve));return !!document.querySelector('.composer') && !document.querySelector('#panel-chat').textContent.includes('obsolete mount failure');})()`)
	// Storage can fail on read or write: retain navigation drafts and tell the truth about reload.
	check(`(() => {window.draftStorageSet=Storage.prototype.setItem;window.draftStorageGet=Storage.prototype.getItem;Storage.prototype.setItem=function(){throw new Error('storage unavailable');};Storage.prototype.getItem=function(){throw new Error('storage unavailable');};draftType(draftInput(),'Memory-only draft');window.previousDraftComposer=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
	settle(`document.querySelector('.composer')!==previousDraftComposer && draftInput().value==='Memory-only draft' && document.querySelector('.composer').textContent.includes('Copy it before reloading')`)
	check(`(() => {Storage.prototype.setItem=draftStorageSet;Storage.prototype.getItem=draftStorageGet;return true;})()`)
	// The first response changes the draft's root without duplicating the next prompt into New chat.
	check(`(() => {
    overgo.api.get=draftOriginalGet;window.draftOriginalStream=overgo.api.stream;
    overgo.api.stream=async function(path,body,options){const response=await draftOriginalStream.call(this,path,body,options);await new Promise(resolve=>window.releaseDraftResponse=resolve);return response;};
    draftType(draftInput(),'Send the first turn');document.querySelector('.send-button').click();return true;
  })()`)
	settle(`typeof releaseDraftResponse==='function'`)
	check(`(() => {draftType(draftInput(),'The next unsent turn');releaseDraftResponse();return true;})()`)
	settle(`!!overgo.conversation() && !document.querySelector('.send-button').disabled && document.querySelector('.chat-log').textContent.includes('Stored answer')`)
	check(`(() => {window.draftStoredConversation=overgo.conversation();overgo.openConversation(null);return true;})()`)
	settle(`!!document.querySelector('.composer textarea') && draftInput().value===''`)
	check(`(() => {overgo.openConversation(draftStoredConversation);return true;})()`)
	settle(`!!document.querySelector('.composer textarea') && draftInput().value==='The next unsent turn' && document.querySelector('.chat-log').textContent.includes('Stored answer')`)
	t.Log("draft lifecycle leg: navigation, reload, file metadata, recipe limits, settings focus, delayed/failed switch, stale mount and unavailable storage passed; physical devices not exercised")
}
