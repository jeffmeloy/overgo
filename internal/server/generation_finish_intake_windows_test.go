package server

import (
	"encoding/json"
	"testing"

	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/testcheck"
)

// TestAClipAskingNoUpscaleIsTakenIn holds the intake to the requests a clip's finish was added beside
// (rtx-vsr/clips): a clip request that asks for no upscale, as every page request did before, is taken in as it was,
// given its format's default and no upscale level, and its own validation passes; one that asks for an upscale is
// given the level a request naming none runs at. The oscillator's video, which builds without a device, carries the
// same finish as every clip request.
func TestAClipAskingNoUpscaleIsTakenIn(t *testing.T) {
	t.Parallel()
	catalog := BindGenerationCatalog(mediacapability.Catalog, mediacapability.Controls, mediacapability.OutputContent)
	controls, refusal := catalog.Controls(modelrecipe.ModuleOscillatorVideoPrepare, "")
	if refusal != "" {
		t.Fatal(refusal)
	}
	for asked, wantLevel := range map[string]string{
		`{"class":1,"frames":4,"scale":1}`:                                    "",
		`{"class":1,"frames":4,"scale":1,"upscale":0}`:                        "",
		`{"class":1,"frames":4,"scale":1,"upscale":2}`:                        "clean-high",
		`{"class":1,"frames":4,"scale":1,"upscale":4}`:                        "clean-high",
		`{"class":1,"frames":4,"scale":1,"upscale":2,"upscale_level":"high"}`: "high",
	} {
		filled := testcheck.Must(withDefaults(controls, json.RawMessage(asked)))(t)
		var request oscillatorimage.VideoRequest
		testcheck.NoError(t, json.Unmarshal(filled, &request))
		if err := oscillatorimage.ValidateVideoRequest(request); err != nil {
			t.Errorf("the intake took %s in as %s, which is refused: %v", asked, filled, err)
		}
		if string(request.UpscaleLevel) != wantLevel || request.Format != "mp4" {
			t.Errorf("the intake took %s in as %s; want the level %q and an MP4", asked, filled, wantLevel)
		}
	}
}
