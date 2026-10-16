package server

import (
	"net/http/httptest"
	"testing"

	"overgo/internal/webuilane"
)

func TestWebUIBrowserSpeechPlayback(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "speech playback requires the browser lane")
	for _, ending := range []string{"drain", "stop", "dispose", "drain-stop"} {
		t.Run(ending, func(t *testing.T) {
			handler, fixture := newSpeechOutputFixture(t)
			gate := make(chan struct{})
			fixture.gate = gate
			server := httptest.NewServer(handler)
			t.Cleanup(server.Close)
			ctx, browser, check, settle := openBrowserLeg(t, server.URL)
			t.Cleanup(func() { _ = browser.Close() })
			settle(`!!document.querySelector('.composer select[aria-label="mode"] option[value="speech"]')`)
			check(`(() => {
  const Native = window.AudioContext;
  window.audioProbe = {created:0, starts:[], ends:0, stops:0, disconnects:0, closes:0, live:0, maxLive:0};
  window.AudioContext = class extends Native {
    constructor(...args) { super(...args); audioProbe.created++; }
    createBufferSource() {
      const source=super.createBufferSource(), start=source.start.bind(source), stop=source.stop.bind(source), disconnect=source.disconnect.bind(source);
      source.start=(at)=>{const pcm=source.buffer.getChannelData(0);audioProbe.starts.push({at, duration:source.buffer.duration, sample:pcm[0], exact:Object.is(pcm[1],-0)&&pcm[2]===Math.fround(2**-149), scheduled:performance.now()});audioProbe.live++;audioProbe.maxLive=Math.max(audioProbe.maxLive,audioProbe.live);return start(at);};
      source.stop=()=>{audioProbe.stops++;return stop();};
      source.disconnect=()=>{audioProbe.disconnects++;return disconnect();};
      source.addEventListener('ended',()=>{audioProbe.ends++;audioProbe.live--;});
      return source;
    }
    close() { audioProbe.closes++;audioProbe.liveAtClose=audioProbe.live;return super.close(); }
  };
  const fetch=window.fetch.bind(window);
  window.fetch=(url,options)=>{if(url==='/generation/run')audioProbe.submitted=performance.now();return fetch(url,options);};
  const mode=document.querySelector('.composer select[aria-label="mode"]');mode.value='speech';mode.dispatchEvent(new Event('change'));
  return true;
})()`)
			settle(`!document.querySelector('.mode-controls').textContent.includes('Loading models')`)
			say(t, ctx, browser, "Read the complete fixture.")
			settle(`audioProbe.starts.length===1 && document.body.textContent.includes('Playing speech')`)
			if _, found, err := fixture.repository.Artifact(t.Context(), fixture.content.Descriptor.ID); err != nil || found {
				t.Fatalf("first playback waited for publication: %v %v", found, err)
			}
			check(`!document.querySelector('.audio-result') && audioProbe.closes===0`)
			switch ending {
			case "drain":
				close(gate)
				settle(`audioProbe.closes===1 && !!document.querySelector('.audio-result audio')`)
				check(`audioProbe.starts.length===6 && audioProbe.ends===6 && audioProbe.disconnects===6 && audioProbe.stops===0 && audioProbe.liveAtClose===0 && audioProbe.maxLive<=2`)
				check(`audioProbe.starts.every((item,index)=>item.exact && Math.abs(item.sample-(index+1)/10)<1e-7 && (!index || item.at>=audioProbe.starts[index-1].at+audioProbe.starts[index-1].duration))`)
				check(`!document.querySelector('.audio-result audio').autoplay && !!document.querySelector('.audio-download')`)
				var measurement struct {
					First     float64 `json:"first"`
					Scheduled float64 `json:"scheduled"`
					Buffers   int     `json:"buffers"`
				}
				if err := browser.Evaluate(ctx, `({first:audioProbe.starts[0].scheduled-audioProbe.submitted,scheduled:audioProbe.starts.at(-1).scheduled-audioProbe.submitted,buffers:audioProbe.maxLive})`, &measurement); err != nil {
					t.Fatal(err)
				}
				t.Logf("fixture first scheduled audio %.3fms, final buffer scheduled %.3fms, peak scheduled buffers %d; generator intentionally held after first chunk, no model throughput claim", measurement.First, measurement.Scheduled, measurement.Buffers)
				say(t, ctx, browser, "Read the complete fixture.")
				settle(`audioProbe.closes===2 && document.querySelectorAll('.audio-result').length===2`)
				check(`audioProbe.starts.length===6 && [...document.querySelectorAll('.audio-result audio')].at(-1).autoplay`)
				if fixture.produced.Load() != 6 {
					t.Fatal("cached recording regenerated")
				}
			case "stop":
				pressKey(t, ctx, browser, "Escape", 27)
				settle(`audioProbe.closes===1 && document.body.textContent.includes('Stopped')`)
			case "dispose":
				check(`(()=>{overgo.openConversation(null);return true;})()`)
				settle(`audioProbe.closes===1`)
			case "drain-stop":
				close(gate)
				<-fixture.released
				settle(`audioProbe.starts.length===6 && audioProbe.ends<6 && audioProbe.closes===0`)
				pressKey(t, ctx, browser, "Escape", 27)
				settle(`audioProbe.closes===1 && document.body.textContent.includes('Stopped')`)
				check(`audioProbe.stops>0 && audioProbe.disconnects===6 && !document.querySelector('.audio-result')`)
				if _, found, err := fixture.repository.Artifact(t.Context(), fixture.content.Descriptor.ID); err != nil || !found {
					t.Fatalf("playback stop invalidated published output: %v %v", found, err)
				}
			}
			if ending == "stop" || ending == "dispose" {
				<-fixture.released
				close(gate)
				check(`audioProbe.starts.length===1 && audioProbe.disconnects===1 && !document.querySelector('.audio-result')`)
				if _, found, err := fixture.repository.Artifact(t.Context(), fixture.content.Descriptor.ID); err != nil || found {
					t.Fatalf("cancelled recording published: %v %v", found, err)
				}
			}
		})
	}
	webuilane.Leg(t, "speech playback lifecycle", "real AudioContext first playback precedes publication, two buffers bound scheduling, sample bits survive transport, completion drains, Stop/disposal release, late output is refused, stopping final playback preserves published output, cached output uses the stored audio player")
}
