package server

import (
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserAudioWhileModelsLoad holds a file chosen while a task's
// models are still loading. The server here serves transcription alone, and
// its answer to the page's request for the models is held back; the page shows
// that it is loading them. A recording chosen then waits: it is not judged by
// the chat model's rule, which has no audio projector and would refuse it.
// When the models arrive the recording is stored into the task's audio input
// and can be sent. Before the page waited, the recording was refused for good
// whenever it was chosen in that window, which a busy machine made wide enough
// for the audio input leg to fall into.
func TestWebUIBrowserAudioWhileModelsLoad(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the audio while models load leg uses Chromium through cmd/webui-lane")
	fixture := newTranscriptionHTTPFixture(t, nil)
	handler, err := New(Config{Repository: fixture.store, RuntimePolicy: testRuntimePolicy(), Workflows: fixture.workspace}, GenerationRefused{Reason: "no chat model is loaded"})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	asked, held := make(chan struct{}), make(chan struct{})
	ask, release := sync.OnceFunc(func() { close(asked) }), sync.OnceFunc(func() { close(held) })
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Every request for the models waits, whichever part of the page asks.
		if r.URL.Path == "/generation/capabilities" {
			ask()
			select {
			case <-held:
			case <-r.Context().Done():
				return
			}
		}
		handler.ServeHTTP(w, r)
	}))
	defer server.Close()
	defer release()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	select {
	case <-asked:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	settle(`!!document.querySelector('.composer input[type=file]') && /Loading models/.test(document.querySelector('.composer').textContent)`)
	check(`(()=>{const bytes=Uint8Array.from(atob(` + strconv.Quote(base64.StdEncoding.EncodeToString(fixture.wave)) + `),c=>c.charCodeAt(0)),file=new File([bytes],'recording.wav',{type:'audio/wav'}),transfer=new DataTransfer();transfer.items.add(file);const picker=document.querySelector('.composer input[type=file]');picker.files=transfer.files;picker.dispatchEvent(new Event('change'));return true;})()`)
	// The models have not come, and the recording is neither refused nor stored.
	check(`(()=>{
  const rows = document.querySelectorAll('.attachment-row'), composer = document.querySelector('.composer').textContent;
  if (rows.length !== 1 || document.querySelector('.attachment-name').textContent !== 'recording.wav') throw new Error('the composer holds ' + rows.length + ' files: ' + composer);
  if (/no audio projector|does not accept/.test(composer)) throw new Error('the recording was judged by the chat model while the models load: ' + composer);
  if (!/Loading models/.test(composer) || document.querySelector('[aria-label="Transcription options"] input')) throw new Error('the models are not still loading: ' + composer);
  if (!/Waiting for the models to load/.test(document.querySelector('.attachment-row').textContent) || !document.querySelector('.composer .send-button').disabled) throw new Error('the recording does not say it waits, or can be sent before it is placed: ' + composer);
  return true;
})()`)
	release()
	settle(`(()=>{
  const composer = document.querySelector('.composer').textContent;
  if (/no audio projector|does not accept|Model input changed/.test(composer)) throw new Error('with the models loaded the recording reads: ' + composer);
  const input = document.querySelector('[aria-label="Transcription options"] input');
  return !!input && input.value.startsWith('file:') && !document.querySelector('.composer .send-button').disabled && document.querySelectorAll('.attachment-row').length === 1;
})()`)
	webuilane.Leg(t, "audio while models load leg", "a recording chosen while the transcription models were loading waited for them and was stored into the task's audio input, never judged by the chat model's rule")
}
