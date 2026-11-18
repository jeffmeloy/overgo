//go:build windows

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// workbenchSection is a MOSS Ultra section's settings as the workflow's own request records them, as the workbench's
// page shows them once its preset is chosen.
type workbenchSection struct {
	Steps    string `json:"steps"`
	Guidance string `json:"guidance"`
	Schedule string `json:"schedule"`
	Denoise  string `json:"denoise"`
	Adapters string `json:"adapters"`
	Width    string `json:"width"`
	Height   string `json:"height"`
	Sharpen  string `json:"sharpen"`
}

// recordedSections reads each section's settings from the workflow's requests (fixtures/comfy/moss-ultra/requests):
// its KSampler's steps, guidance, sampler, scheduler and denoise (none where it samples the whole schedule), the
// adapters its LoRA loader stacks, its canvas as its ResolutionSelector sizes it, and its sharpening.
func recordedSections(t *testing.T) map[string]workbenchSection {
	t.Helper()
	sections := map[string]struct{ slug, sampler string }{
		"ultra-text-to-image": {"text-to-image", "598"}, "ultra-text-to-image-turbo": {"text-to-image-turbo", "331"},
		"ultra-image-to-image": {"image-to-image", "506"}, "ultra-image-to-image-turbo": {"image-to-image-turbo", "198"},
		"ultra-depth-control": {"depth-control", "443"}, "ultra-depth-control-turbo": {"depth-control-turbo", "48"},
	}
	recorded := map[string]workbenchSection{}
	for preset, section := range sections {
		var nodes map[string]struct {
			Class  string         `json:"class_type"`
			Inputs map[string]any `json:"inputs"`
		}
		path := filepath.Join(testutil.RepoRoot(t), "fixtures", "comfy", "moss-ultra", "requests", section.slug+".json")
		testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(path))(t), &nodes))
		sampler := nodes[section.sampler].Inputs
		text := func(value any) string { return string(testcheck.Must(json.Marshal(value))(t)) }
		// The page writes an object's fields in its request's order, so the settings are encoded so.
		settings := workbenchSection{Steps: text(sampler["steps"]), Guidance: text(sampler["cfg"]),
			Schedule: text(struct {
				Sampler any `json:"sampler"`
				Name    any `json:"name"`
			}{sampler["sampler_name"], sampler["scheduler"]})}
		if denoise := sampler["denoise"].(float64); denoise < 1 {
			settings.Denoise = text(denoise)
		}
		type adapter struct {
			Name     string `json:"name"`
			Strength any    `json:"strength"`
		}
		var adapters []adapter
		for _, node := range nodes {
			switch in := node.Inputs; node.Class {
			case "ResolutionSelector":
				aspect, _, _ := strings.Cut(in["aspect_ratio"].(string), " ")
				width, height, err := media.Canvas{Aspect: aspect, Megapixels: in["megapixels"].(float64), Multiple: int(in["multiple"].(float64))}.Size()
				testcheck.NoError(t, err)
				settings.Width, settings.Height = text(width), text(height)
			case "ImageSharpen":
				settings.Sharpen = text(struct {
					Radius any `json:"radius"`
					Sigma  any `json:"sigma"`
					Alpha  any `json:"alpha"`
				}{in["sharpen_radius"], in["sigma"], in["alpha"]})
			case "Power Lora Loader (rgthree)":
				for slot := 1; ; slot++ {
					lora, held := in["lora_"+string(rune('0'+slot))].(map[string]any)
					if !held {
						break
					}
					if lora["on"] == true {
						adapters = append(adapters, adapter{strings.TrimSuffix(lora["lora"].(string), ".safetensors"), lora["strength"]})
					}
				}
			}
		}
		if adapters != nil {
			settings.Adapters = text(adapters)
		}
		recorded[preset] = settings
	}
	return recorded
}

// TestTheWorkbenchOffersTheUltraPresets holds MOSS's workbench (moss-ultra-compare/workbench), the generation page with
// MOSS chosen, to the MOSS Ultra workflow: each of its six section presets, chosen, shows the section's recorded steps,
// guidance, sampler and scheduler, denoise, adapters, canvas size and sharpening, and the backup negative, every one a
// control the person can change; the page offers the three prompt modes (the prompt, the enhancer, the layout); a drag
// across the layout's canvas, shaped as the picture, adds a region of its shares; the depth control's picture is
// previewed as its run reads it; and an image to image run's picture is shown and compared, B over A, with the picture
// it started from.
func TestTheWorkbenchOffersTheUltraPresets(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the workbench leg uses Chromium through cmd/webui-lane")
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
	testcheck.Must(artifact.Publish(t.Context(), fixture.store, artifact.Batch{Key: "fixture/workbench-picture", Artifacts: []artifact.Descriptor{picture.Descriptor}, Contents: []artifact.Content{picture}}))(t)
	recorded := string(testcheck.Must(json.Marshal(recordedSections(t)))(t))
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-generation.active fieldset[data-control="regions"] .box-canvas')`)
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  window.bench={
    panel,
    field:(name)=>[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent===name).querySelector('input,select,textarea'),
    group:(name)=>panel.querySelector('fieldset[data-control="'+name+'"]'),
    choose:(name)=>{const preset=bench.field('preset');preset.value=name;preset.dispatchEvent(new Event('change',{bubbles:true}));},
  };
  panel.querySelector('.advanced-controls').open=true;
  return true;
})()`)
	// Each section's preset shows its recorded settings.
	check(`(() => {
  const recorded=` + recorded + `;
  for (const [preset, want] of Object.entries(recorded)) {
    bench.choose(preset);
    const got={steps:bench.field('steps').value, guidance:bench.field('guidance').value, schedule:bench.group('schedule').value, denoise:bench.field('denoise').value,
      adapters:bench.group('adapters').value, width:bench.field('width').value, height:bench.field('height').value, sharpen:bench.group('sharpen').value};
    for (const key of Object.keys(want)) if (got[key]!==want[key]) throw new Error(preset+' shows '+key+' '+got[key]+'; the workflow records '+want[key]);
    if (!bench.field('negative prompt').value) throw new Error(preset+' shows no backup negative');
  }
  return true;
})()`)
	// The three prompt modes: the prompt, the enhancer and the layout.
	check(`(() => { const enhance=bench.field('enhance'); return !!bench.field('prompt') && [...enhance.options].some(option=>option.value==='true') && !!bench.group('layout'); })()`)
	// A drag across the layout's canvas, shaped as the picture, adds a region of its shares.
	check(`(() => {
  bench.choose('ultra-text-to-image');
  const canvas=bench.group('regions').querySelector('.box-canvas');
  if (canvas.style.aspectRatio!==bench.field('width').value+' / '+bench.field('height').value) throw new Error('the canvas is shaped '+canvas.style.aspectRatio);
  const bounds=canvas.getBoundingClientRect();
  const at=(x,y)=>({clientX:bounds.left+x*bounds.width, clientY:bounds.top+y*bounds.height, bubbles:true, pointerId:1, isPrimary:true});
  canvas.dispatchEvent(new PointerEvent('pointerdown',at(0.1,0.2)));
  canvas.dispatchEvent(new PointerEvent('pointermove',at(0.5,0.6)));
  canvas.dispatchEvent(new PointerEvent('pointerup',at(0.5,0.6)));
  const regions=JSON.parse(bench.group('regions').value);
  const near=(a,b)=>Math.abs(a-b)<0.002;
  return regions.length===1 && near(regions[0].x,0.1) && near(regions[0].y,0.2) && near(regions[0].w,0.4) && near(regions[0].h,0.4) && canvas.querySelectorAll('.box').length===1 && bench.group('layout').value.includes('"regions"');
})()`)
	// The depth control's picture is previewed as its run reads it.
	check(`(() => {
  bench.choose('ultra-depth-control');
  bench.field('prompt').value='a fox';
  bench.group('regions').value='';
  [...bench.group('control').querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='picture').querySelector('input').value='` + picture.Descriptor.ID.String() + `';
  [...bench.panel.querySelectorAll('button')].find(button=>button.textContent==='Preview depth map').click();
  return true;
})()`)
	settle(`(() => { const map=bench.panel.querySelector('.depth-preview img'); return !!map && map.src.startsWith('blob:') && map.naturalWidth===2; })()`)
	// An image to image run's picture is shown and compared, B over A, with the picture it started from.
	check(`(() => {
  bench.group('control').value='';
  bench.choose('ultra-image-to-image-turbo');
  bench.field('picture').value='` + picture.Descriptor.ID.String() + `';
  [...bench.panel.querySelectorAll('button')].find(button=>button.textContent==='Run').click();
  return true;
})()`)
	settle(`(() => { const compare=bench.panel.querySelector('.ab-compare'); return !!compare && [...compare.querySelectorAll('img')].every(shown=>shown.src.startsWith('blob:') && shown.complete); })()`)
	check(`(() => {
  const compare=bench.panel.querySelector('.ab-compare');
  const [a,b]=compare.querySelectorAll('select');
  const divide=compare.querySelector('input[type=range]');
  divide.value='30'; divide.dispatchEvent(new Event('input',{bubbles:true}));
  return a.value==='` + picture.Descriptor.ID.String() + `' && b.value!==a.value && compare.querySelector('.ab-b').style.clipPath.includes('30%') && !!bench.panel.querySelector('.workbench-output');
})()`)
	inputs := fixture.requests()
	if len(inputs) != 1 {
		t.Fatalf("the workbench ran %d requests; want the image to image run", len(inputs))
	}
	run := inputs[0]
	if run.Image != picture.Descriptor.ID.String() || run.Denoise != 0.72 || run.Steps != 8 || run.Adapters == nil || len(*run.Adapters) != 2 || run.Schedule == nil || run.Sharpen == nil || run.Control != nil {
		t.Errorf("the image to image run asked %+v; want the section's settings and the picture it started from", run)
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-workbench", "MOSS's generation page as the MOSS Ultra workflow's workbench: the six section presets show their recorded settings, the three prompt modes, a region drawn on the layout's canvas, the depth control's preview, and an image to image run's picture compared with the picture it started from.")
}
