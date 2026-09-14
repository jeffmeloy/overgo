package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserAttachmentWorkflow(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": attachment workflow runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"File answer"}}))
	handler.config.APIKey = testAPIKey
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 45*time.Second, errors.New("attachment workflow did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatal(err)
		}
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => {window.attachmentOldComposer=document.querySelector('.composer');document.querySelector('#settings-toggle').click();const key=document.querySelector('#api-key');key.value=` + strconv.Quote(testAPIKey) + `;key.dispatchEvent(new Event('change'));document.querySelector('#settings-dialog').close();return true;})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==attachmentOldComposer`)
	settle(`document.querySelector('#global-operation-shell').hidden`)
	check(`(() => {
  window.fileRows=()=>[...document.querySelectorAll('.attachment-row')];
  window.fileRow=name=>fileRows().find(row=>row.querySelector('.attachment-name').textContent===name);
  window.fileButton=(name,action)=>fileRow(name).querySelector('[aria-label="'+action+' '+name+'"]');
  window.pickFiles=files=>{const transfer=new DataTransfer();files.forEach(file=>transfer.items.add(file));const picker=document.querySelector('.composer input[type=file]');picker.files=transfer.files;picker.dispatchEvent(new Event('change'));};
  window.nativeFileReader=FileReader;let fail=true;
  window.FileReader=class extends nativeFileReader {readAsDataURL(file){if(file.name==='retry.txt'&&fail){fail=false;this.dispatchEvent(new ProgressEvent('error'));return;}if(file.name==='held.txt'){window.heldFileReader=this;this.onprogress(new ProgressEvent('progress',{lengthComputable:true,loaded:2,total:file.size}));return;}super.readAsDataURL(file);}};
  pickFiles([new File(['one'],'one.txt',{type:'text/plain'}),new File(['two'],'two.txt',{type:'text/plain'}),new File(['retry'],'retry.txt',{type:'text/plain'}),new File(['unknown'],'bad.exe',{type:'application/x-msdownload'})]);return true;
})()`)
	settle(`fileRow('one.txt').dataset.state==='ready' && fileRow('two.txt').dataset.state==='ready' && fileRow('retry.txt').dataset.state==='error' && fileRow('bad.exe').dataset.state==='refused'`)
	check(`document.querySelector('.send-button').disabled && !document.querySelector('.attachment-strip .card') && !document.querySelector('.attachment-guidance').hidden`)
	check(`(() => {fileButton('retry.txt','Retry').focus();return true;})()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle(`fileRow('retry.txt').dataset.state==='ready'`)
	check(`document.activeElement===fileButton('retry.txt','Remove')`)
	check(`(() => {fileButton('bad.exe','Remove').click();pickFiles([new File(['held file'],'held.txt',{type:'text/plain'})]);return true;})()`)
	settle(`fileRow('held.txt').textContent.includes('2 B of')`)
	check(`(() => {fileButton('held.txt','Remove').click();Object.defineProperty(heldFileReader,'result',{value:'data:text/plain;base64,aGVsZA=='});heldFileReader.onload();return !fileRow('held.txt')&&!document.querySelector('.send-button').disabled;})()`)
	check(`(() => {
  window.FileReader=nativeFileReader;
  const data=new DataTransfer();data.items.add(new File(['pasted'],'pasted.txt',{type:'text/plain'}));
  document.querySelector('.composer textarea').dispatchEvent(new ClipboardEvent('paste',{clipboardData:data,bubbles:true,cancelable:true}));
  return true;
})()`)
	settle(`fileRow('pasted.txt').dataset.state==='ready' && !document.querySelector('.send-button').disabled`)
	check(`(() => {window.previousFileLimit=overgo.capabilities().media.max_media_bytes;overgo.capabilities().media.max_media_bytes=8;pickFiles([new File(['this is too large'],'large.txt',{type:'text/plain'}),new File([],'empty.txt',{type:'text/plain'})]);return true;})()`)
	settle(`fileRow('large.txt').dataset.state==='refused' && fileRow('empty.txt').dataset.state==='refused'`)
	check(`(() => {overgo.capabilities().media.max_media_bytes=previousFileLimit;return document.querySelector('.send-button').disabled && fileRow('empty.txt').textContent.includes('File is empty');})()`)
	// A controlled capability declaration enables decoding/slot UI without
	// claiming that this synthetic generator implements image inference.
	check(`(() => {
  for(const row of fileRows())row.querySelector('[aria-label^="Remove "]').click();
  const caps=overgo.capabilities();caps.media.accept.push('image/png');caps.media.max_image_bytes=1048576;caps.media.max_image_dimension=1024;caps.media.max_image_pixels=1048576;
  caps.modes.push({id:'image',label:'Image fixture',enabled:true});
  window.attachmentGet=overgo.api.get;window.attachmentUpload=overgo.api.upload;
  const attachmentCaps=structuredClone(caps);
  overgo.api.get=async function(path,options){if(path==='/generation/capabilities')return [{task:'image',recipe:'attachment-fixture-recipe',name:'Attachment fixture',controls:[{name:'source',label:'Source file',type:'artifact',required:true}]}];const result=await attachmentGet.call(this,path,options);if(path==='/workspace/manifest')result.model=structuredClone(attachmentCaps);return result;};
  window.attachmentOldComposer=document.querySelector('.composer');overgo.openConversation(null);return true;
})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==attachmentOldComposer`)
	check(`(() => {window.attachmentPNG=Uint8Array.from(atob(` + strconv.Quote(tinyPNGBase64(t)) + `),char=>char.charCodeAt(0));pickFiles([new File(['corrupt'],'corrupt.png',{type:'image/png'}),new File([attachmentPNG],'valid.png',{type:'image/png'})]);return true;})()`)
	settle(`fileRow('corrupt.png').dataset.state==='error' && fileRow('valid.png').dataset.state==='ready'`)
	check(`(() => {
  window.stableImage=fileRow('valid.png').querySelector('img');fileButton('corrupt.png','Retry').click();
  return fileRow('valid.png').querySelector('img')===stableImage;
})()`)
	settle(`fileRow('corrupt.png').dataset.state==='error'`)
	check(`(() => {
  for(const row of fileRows())row.querySelector('[aria-label^="Remove "]').click();
  const mode=document.querySelector('.composer select[aria-label=mode]');mode.value='image';mode.dispatchEvent(new Event('change'));return true;
})()`)
	settle(`!!document.querySelector('.mode-controls label.control input')`)
	check(`(() => {
  window.fileSlot=()=>document.querySelector('.mode-controls label.control input');
  let fail=true;overgo.api.upload=async function(path,file,options){if(fail){fail=false;throw new Error('Upload temporarily unavailable');}return attachmentUpload.call(this,path,file,options);};
  pickFiles([new File([attachmentPNG],'upload.png',{type:'image/png'})]);return true;
})()`)
	settle(`fileRow('upload.png').dataset.state==='error' && fileRow('upload.png').textContent.includes('Upload temporarily unavailable')`)
	check(`(() => {fileButton('upload.png','Retry').click();return true;})()`)
	settle(`fileRow('upload.png').dataset.state==='ready' && fileSlot().value.startsWith('file:')`)
	settle(`!!document.querySelector('.intake-thumb img') && document.querySelector('.intake-thumb img').src.startsWith('blob:') && document.querySelector('.intake-thumb img').naturalWidth>0`)
	check(`(async () => {
  window.uploadedFile=fileSlot().value;const bytes=await overgo.api.blob('/artifacts/content?id='+encodeURIComponent(uploadedFile));
  return bytes.size===attachmentPNG.length && !fileRow('upload.png').querySelector('img') && !document.querySelector('.send-button').disabled;
})()`)
	check(`(() => {fileButton('upload.png','Remove').click();return fileSlot().value==='';})()`)
	// The transport can finish after a local abort. Its stored result must not
	// reattach itself or overwrite a newer file selected for the same slot.
	check(`(() => {
  overgo.api.upload=async function(path,file,options){if(file.name==='late.txt'){window.lateUploadSignal=options.signal;const stored=await attachmentUpload.call(this,path,file);return new Promise(resolve=>window.releaseLateUpload=()=>resolve(stored));}return attachmentUpload.call(this,path,file,options);};
  pickFiles([new File(['late'],'late.txt',{type:'text/plain'})]);return true;
})()`)
	settle(`typeof releaseLateUpload==='function' && fileRow('late.txt').dataset.state==='uploading'`)
	check(`(() => {fileButton('late.txt','Cancel').click();pickFiles([new File(['newer'],'newer.txt',{type:'text/plain'})]);return lateUploadSignal.aborted && fileRow('late.txt').dataset.state==='cancelled';})()`)
	settle(`fileRow('newer.txt').dataset.state==='ready'`)
	check(`(async () => {window.newerFile=fileSlot().value;releaseLateUpload();await Promise.resolve();await Promise.resolve();fileButton('late.txt','Remove').click();return fileSlot().value===newerFile&&!document.querySelector('.send-button').disabled;})()`)
	check(`(() => {window.releaseLateUpload=null;pickFiles([new File(['manual input'],'late.txt',{type:'text/plain'})]);return true;})()`)
	settle(`typeof releaseLateUpload==='function' && fileRow('late.txt').dataset.state==='uploading'`)
	check(`(() => {fileSlot().value=uploadedFile;fileSlot().dispatchEvent(new Event('input'));releaseLateUpload();return true;})()`)
	settle(`fileRow('late.txt').dataset.state==='error' && fileRow('late.txt').textContent.includes('input was changed')`)
	check(`(() => {fileButton('late.txt','Remove').click();return fileSlot().value===uploadedFile;})()`)
	check(`(() => {
  window.releaseLateUpload=null;pickFiles([new File(['late mode'],'late.txt',{type:'text/plain'})]);return true;
})()`)
	settle(`typeof releaseLateUpload==='function' && fileRow('late.txt').dataset.state==='uploading'`)
	check(`(() => {const mode=document.querySelector('.composer select[aria-label=mode]');mode.value='chat';mode.dispatchEvent(new Event('change'));return lateUploadSignal.aborted;})()`)
	settle(`fileRow('late.txt').dataset.state==='error' && fileRow('newer.txt').dataset.state==='error'`)
	check(`(async () => {releaseLateUpload();await Promise.resolve();await Promise.resolve();fileButton('late.txt','Retry').click();fileButton('newer.txt','Retry').click();return true;})()`)
	settle(`fileRows().every(row=>row.dataset.state==='ready') && !document.querySelector('.send-button').disabled`)
	check(`(() => {
  for(const row of fileRows())row.querySelector('[aria-label^="Remove "]').click();
  const mode=document.querySelector('.composer select[aria-label=mode]');mode.value='image';mode.dispatchEvent(new Event('change'));return true;
})()`)
	settle(`!!fileSlot()`)
	check(`(() => {overgo.api.upload=attachmentUpload;pickFiles([new File(['real bytes'],'unknown-type')]);return true;})()`)
	settle(`fileRow('unknown-type').dataset.state==='refused' && fileRow('unknown-type').textContent.includes('application/octet-stream')`)
	check(`fileSlot().value==='' && document.querySelector('.send-button').disabled`)
	check(`(() => {
  fileButton('unknown-type','Remove').click();fileSlot().value=uploadedFile;
  window.attachmentGenerate=overgo.generate;overgo.generate=async function*(){yield {type:'media',kind:'image',url:'/artifacts/content?id='+encodeURIComponent(uploadedFile),artifact:uploadedFile,caption:'Controlled stored output'};yield {type:'done'};};return true;
})()`)
	say(t, ctx, browser, "Show the controlled stored output")
	settle(`!!document.querySelector('.msg.media img') && document.querySelector('.msg.media img').naturalWidth>0 && !document.querySelector('.send-button').disabled`)
	check(`(() => {overgo.generate=attachmentGenerate;const mode=document.querySelector('.composer select[aria-label=mode]');mode.value='chat';mode.dispatchEvent(new Event('change'));[...document.querySelectorAll('.msg.media button')].find(button=>button.textContent==='use as input').click();return true;})()`)
	settle(`fileRows().length===1 && fileRows()[0].dataset.state==='ready'`)
	check(`(() => {window.reusedFileName=fileRows()[0].querySelector('.attachment-name').textContent;window.attachmentOldComposer=document.querySelector('.composer');overgo.openConversation(null);return true;})()`)
	settle(`!!document.querySelector('.composer') && document.querySelector('.composer')!==attachmentOldComposer && fileRow(reusedFileName).dataset.state==='reattach'`)
	check(`(() => {
  window.attachmentBlob=overgo.api.blob;let fail=true;overgo.api.blob=async function(path,options){if(fail){fail=false;throw new Error('Stored file temporarily unavailable');}return attachmentBlob.call(this,path,options);};fileButton(reusedFileName,'Retry').click();return true;
})()`)
	settle(`fileRow(reusedFileName).dataset.state==='error' && fileRow(reusedFileName).textContent.includes('Stored file temporarily unavailable')`)
	check(`(() => {fileButton(reusedFileName,'Retry').click();return true;})()`)
	settle(`fileRow(reusedFileName).dataset.state==='ready' && !!fileRow(reusedFileName).querySelector('img') && !document.querySelector('.send-button').disabled`)
	check(`(() => {overgo.api.blob=attachmentBlob;const source=[...fileRow(reusedFileName).querySelectorAll('a')].find(link=>link.textContent==='Source file');return source.getAttribute('href').includes(encodeURIComponent(uploadedFile));})()`)
	check(`(() => {fileButton(reusedFileName,'Remove').click();pickFiles(Array.from({length:8},(_,index)=>new File(['text'],'phone-'+index+'.txt',{type:'text/plain'})));return true;})()`)
	settle(`fileRows().length===8 && !document.querySelector('.send-button').disabled`)
	for _, viewport := range []webuilane.Viewport{{Name: "phone", Width: 390, Height: 844}, {Name: "keyboard", Width: 390, Height: 320}, {Name: "small", Width: 320, Height: 568}} {
		if findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "attachment-workflow"); err != nil || len(findings) != 0 {
			t.Fatalf("attachment %s layout: %v, %v", viewport.Name, findings, err)
		}
		check(`(() => {const send=document.querySelector('.send-button').getBoundingClientRect(),input=document.querySelector('.composer textarea').getBoundingClientRect();return send.bottom<=innerHeight&&input.top>=0&&input.bottom<=innerHeight&&document.querySelector('.chat-log').clientHeight>0&&document.querySelector('.composer-extras').scrollHeight>document.querySelector('.composer-extras').clientHeight;})()`)
	}
	check(`overgo.errors.length===0`)
	t.Log("attachment workflow leg: native multi-file/paste, read/decode refusal and retry, stable previews/focus, real authenticated intake/preview, upload cancellation with late completion, mode rebinding, unknown-MIME refusal and bounded phone controls passed; declared image/slot fixtures do not claim generation or physical-device evidence")
}
