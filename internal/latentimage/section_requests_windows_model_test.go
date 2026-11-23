//go:build modeltest && windows

package latentimage

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// mossPipelineClass is the pipeline class MOSS's catalog entry is kept under (profiles.json).
const mossPipelineClass = "MOSSImagePipeline"

// statedSections are the MOSS Ultra workflow's six sections' requests as a page sends them
// (fixtures/comfy/moss-ultra/section_requests.json, moss-section-requests): each section's preset and what the section
// sets beside it, its picture named as the recorder gave it.
type statedSections struct {
	Picture  string                     `json:"picture"`
	Sections map[string]json.RawMessage `json:"sections"`
}

// statedRequest is the section's request as section_requests.json states it, filled by its preset as its session
// fills it (Request.withPreset), its picture the recorder's.
func statedRequest(t *testing.T, slug string) Request {
	t.Helper()
	var stated statedSections
	testcheck.NoError(t, strictjson.DecodeBytes(testcheck.Must(os.ReadFile(filepath.Join(testutil.RepoRoot(t), "fixtures", "comfy", "moss-ultra", "section_requests.json")))(t), &stated))
	raw, held := stated.Sections[slug]
	if !held {
		t.Fatalf("section_requests.json states no request for %s", slug)
	}
	var request Request
	testcheck.NoError(t, strictjson.DecodeBytes(raw, &request))
	picture := base64.StdEncoding.EncodeToString(ultraPictureBytes(t))
	if request.Image == stated.Picture {
		request.Image = picture
	}
	if request.Control != nil && request.Control.Picture == stated.Picture {
		control := *request.Control
		control.Picture = picture
		request.Control = &control
	}
	catalog := mossCatalog()
	return testcheck.Must(request.withPreset(catalog))(t)
}

// TestEachSectionStatesTheRequestItsRecordingAsks holds the six sections' stated requests, which the page sends for
// each section's preset (TestWebUIBrowserMOSSSectionRequests), to the requests the section tests draw from, as each
// section's recording asks it (workflowSections): each stated request filled by its preset is its section's, member for
// member. Nothing had held a page's request for a preset to its section's: the depth control section's own request
// had named its adapter apart from its loader and gone unsharpened where the workflow sharpens it.
func TestEachSectionStatesTheRequestItsRecordingAsks(t *testing.T) {
	for _, section := range workflowSections(t) {
		derived := testcheck.Must(json.Marshal(section.request(t, readComfySection(t, "pass-1", section.slug))))(t)
		stated := testcheck.Must(json.Marshal(statedRequest(t, section.slug)))(t)
		if string(stated) != string(derived) {
			t.Errorf("%s's stated request, filled by its preset, is\n%s\nwhere its recording asks\n%s", section.slug, abridged(stated), abridged(derived))
		}
	}
}

// abridged is a request's JSON with its pictures' bytes cut to their first characters.
func abridged(request []byte) string {
	var fields map[string]any
	if json.Unmarshal(request, &fields) != nil {
		return string(request)
	}
	cut := func(value any) any {
		if text, ok := value.(string); ok && len(text) > 24 {
			return text[:24] + "..."
		}
		return value
	}
	if image, held := fields["image"]; held {
		fields["image"] = cut(image)
	}
	if control, ok := fields["control"].(map[string]any); ok {
		control["picture"] = cut(control["picture"])
	}
	written, err := json.Marshal(fields)
	if err != nil {
		return string(request)
	}
	return string(written)
}

// TestEachDrawnPromptIsTheWorkflowsPromptUsed holds what a picture reports it was drawn from (the generator's prompt and
// negative of its filled request, EncodedImage.Prompt, draw-result-carries) to what the MOSS Ultra workflow shows beside
// the same section ("Prompt used" 436, "Negative prompt used" 592), for text to image in each prompt mode as ComfyUI
// recorded it: the real prompt (real-1), the enhancer's text (mode-2) and the layout its builder's fields make
// (builder-1), where the request gives no prompt.
func TestEachDrawnPromptIsTheWorkflowsPromptUsed(t *testing.T) {
	catalog := mossCatalog()
	for _, golden := range goldenCases {
		if golden.slug != "text-to-image" {
			continue
		}
		recording := readComfySection(t, golden.run, golden.slug)
		filled, err := golden.request(t, recording).withPreset(catalog)
		prompt, negative := filled.prompt(), filled.NegativePrompt
		testcheck.NoError(t, err)
		if prompt != recording.Texts["436"][0] || negative != recording.Texts["592"][0] {
			t.Errorf("%s reports it was drawn from %q and %q; the workflow showed %q and %q", golden.name, prompt, negative, recording.Texts["436"][0], recording.Texts["592"][0])
		}
	}
}

// mossCatalog is MOSS's catalog entry, read by its pipeline class as a selected profile names it.
func mossCatalog() Catalog {
	return Catalog{entry: sync.OnceValues(func() (catalogProfile, error) { return lookupCatalogProfile(mossPipelineClass) })}
}
