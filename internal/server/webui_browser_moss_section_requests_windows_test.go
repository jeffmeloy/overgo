//go:build windows

package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/media"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// sectionOrder is the MOSS Ultra workflow's six sections in its own order, by the slugs section_requests.json states
// them under.
var sectionOrder = []string{"text-to-image", "image-to-image", "depth-control", "text-to-image-turbo", "image-to-image-turbo", "depth-control-turbo"}

// beyondPreset is a request's members less those equal to what its preset fills (latentimage.PresetSettings), so a
// request sent with its preset's fills shown is the request that names the preset and what it sets beside it.
func beyondPreset(t *testing.T, request []byte, settings map[string]json.RawMessage) map[string]json.RawMessage {
	t.Helper()
	var members map[string]json.RawMessage
	testcheck.NoError(t, json.Unmarshal(request, &members))
	var preset string
	testcheck.NoError(t, json.Unmarshal(members["preset"], &preset))
	var filled map[string]json.RawMessage
	testcheck.NoError(t, json.Unmarshal(settings[preset], &filled))
	for name, value := range members {
		if fill, held := filled[name]; held && jsonEqual(t, fill, value) {
			delete(members, name)
		}
	}
	return members
}

// jsonEqual says two JSON values are the same value.
func jsonEqual(t *testing.T, a, b json.RawMessage) bool {
	t.Helper()
	var left, right any
	testcheck.NoError(t, json.Unmarshal(a, &left))
	testcheck.NoError(t, json.Unmarshal(b, &right))
	return bytes.Equal(testcheck.Must(json.Marshal(left))(t), testcheck.Must(json.Marshal(right))(t))
}

// TestWebUIBrowserMOSSSectionRequests holds the request MOSS's generation page sends for each of the MOSS Ultra
// workflow's six sections to the one that section's test draws from (moss-section-requests), without the card: the
// sections' requests are stated once (fixtures/comfy/moss-ultra/section_requests.json), each its preset and what the
// section sets beside it, and latentimage's TestEachSectionStatesTheRequestItsRecordingAsks holds each, filled by its
// preset, to its recording. Here each section's preset is chosen on the page, what its stated request sets beside it is
// set (its prompt, seed, size, denoise and picture, a stored file's identity), the run is sent, and the request the
// session takes, less what its preset fills, is the stated one member for member, with nothing more. The page had been
// held field by field (the workbench's shown settings, a schedule and sharpening set part by part), never as the whole
// request a section's test draws.
func TestWebUIBrowserMOSSSectionRequests(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the section requests leg uses Chromium through cmd/webui-lane")
	fixture := newMOSSGenerationFixture(t, nil)
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
	testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/section-picture", Artifacts: []artifact.Descriptor{picture.Descriptor}, Contents: []artifact.Content{picture}}))(t)
	var stated struct {
		Picture  string                     `json:"picture"`
		Sections map[string]json.RawMessage `json:"sections"`
	}
	statedFile := testcheck.Must(os.ReadFile(filepath.Join(testutil.RepoRoot(t), "fixtures", "comfy", "moss-ultra", "section_requests.json")))(t)
	testcheck.NoError(t, strictjson.DecodeBytes(statedFile, &stated))
	// The stated picture, the recorder's, is the page's stored picture here.
	named := testcheck.Must(json.Marshal(stated.Picture))(t)
	stored := testcheck.Must(json.Marshal(picture.Descriptor.ID.String()))(t)
	settings := testcheck.Must(latentimage.PresetSettings(fixture.root))(t)

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
    field:(name)=>[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent===name).querySelector('input,select,textarea'),
    group:(name)=>panel.querySelector('fieldset[data-control="'+name+'"]'),
    part:(group,name)=>[...bench.group(group).querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent===name).querySelector('input,select,textarea'),
    choose:(name)=>{const preset=bench.field('section');preset.value=name;preset.dispatchEvent(new Event('change',{bubbles:true}));},
    status:()=>bench.panel.querySelector('[aria-label="run status"]'),
    button:()=>[...bench.panel.querySelectorAll('button')].find(button=>button.textContent==='Run'),
    run:()=>{bench.panel.querySelectorAll('.workbench-output').forEach(shown=>shown.remove());bench.status().replaceChildren();bench.button().click();},
    ran:()=>!!bench.panel.querySelector('.workbench-output') && !bench.button().disabled,
    refused:()=>!bench.button().disabled && !bench.panel.querySelector('.workbench-output') && bench.status().textContent.trim()!=='',
  };
  panel.querySelector('.advanced-controls').open=true;
  return true;
})()`)
	for _, slug := range sectionOrder {
		raw, held := stated.Sections[slug]
		if !held {
			t.Fatalf("section_requests.json states no request for %s", slug)
		}
		want := bytes.ReplaceAll(raw, named, stored)
		var section struct {
			Preset  string   `json:"preset"`
			Prompt  string   `json:"prompt"`
			Width   int      `json:"width"`
			Height  int      `json:"height"`
			Seed    int64    `json:"seed"`
			Image   string   `json:"image"`
			Denoise *float64 `json:"denoise"`
			Control *struct {
				Picture  string   `json:"picture"`
				Strength *float64 `json:"strength"`
			} `json:"control"`
		}
		testcheck.NoError(t, json.Unmarshal(want, &section))
		values := string(testcheck.Must(json.Marshal(section))(t))
		check(`(() => {
  const section=` + values + `;
  // What a person typed stays through a change of preset, so what the last section set is cleared first.
  bench.group('control').value='';
  bench.field('picture').value='';
  bench.field('denoise').value='';
  bench.choose(section.preset);
  const set=(field,value)=>{field.value=value;field.dispatchEvent(new Event('input',{bubbles:true}));};
  set(bench.field('prompt'),section.prompt);
  set(bench.field('seed'),String(section.seed));
  set(bench.field('width'),String(section.width));
  set(bench.field('height'),String(section.height));
  if (section.denoise!==null) set(bench.field('denoise'),String(section.denoise));
  if (section.image) set(bench.field('picture'),section.image);
  if (section.control) {
    set(bench.part('control','picture'),section.control.picture);
    if (section.control.strength!==null) set(bench.part('control','strength'),String(section.control.strength));
  }
  bench.run();
  return true;
})()`)
		settle(`bench.ran() || bench.refused()`)
		check(`bench.ran() || (() => { throw new Error('the run of ' + bench.field('section').value + ' was refused: ' + bench.status().textContent); })()`)
		inputs := fixture.requests()
		if len(inputs) == 0 {
			t.Fatalf("%s's run reached no session", slug)
		}
		sent := testcheck.Must(json.Marshal(inputs[len(inputs)-1]))(t)
		got, beyond := beyondPreset(t, sent, settings), beyondPreset(t, want, settings)
		if gotJSON, wantJSON := testcheck.Must(json.Marshal(got))(t), testcheck.Must(json.Marshal(beyond))(t); !bytes.Equal(gotJSON, wantJSON) {
			t.Errorf("for %s the page sent, beyond its preset,\n%s\nwhere its section's test draws\n%s", slug, gotJSON, wantJSON)
		}
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-section-requests", "MOSS's generation page sends, for each of the MOSS Ultra workflow's six sections, its preset and what the section sets beside it, the request its section's test draws from.")
}
