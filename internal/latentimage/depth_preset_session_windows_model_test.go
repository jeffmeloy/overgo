//go:build modeltest && windows

package latentimage

import (
	"context"
	"encoding/base64"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestADepthPresetRequestWithNoEndSamples holds a session loaded for the MOSS Ultra workflow's depth control preset to
// sampling a request that leaves its control's end to the preset (depth-preset-session, SQA-418): on the workflow's
// files, a request at 1024 x 576 naming ultra-depth-control and a depth map, with no end, is prepared and sampled by the
// session it loaded, one naming the preset's 0.5 is too, and a session loaded for an end of 1 still refuses a request
// that leaves its end to the preset's 0.5. The preset fills an omitted end with a copy of the request's control, once as
// the session loads and again as it prepares, and prepareRotary had compared the two copies' pointers and refused it.
func TestADepthPresetRequestWithNoEndSamples(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	_, depthMap := depthControlSection(t, readComfySection(t, "pass-1", "depth-control"))
	asking := func(end *float64) Request {
		return Request{Prompt: "a fox", Preset: "ultra-depth-control", Width: 1024, Height: 576, Steps: 2, Seed: 7,
			Control: &DepthControl{Map: base64.StdEncoding.EncodeToString(depthMap), End: end}}
	}
	half, whole := 0.5, 1.0
	generator := testcheck.Must(LoadGenerator(ctx, root, profile, asking(nil)))(t)
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	for index, end := range []*float64{nil, &half} {
		request := asking(end)
		if index > 0 {
			testcheck.NoError(t, generator.Reset(ctx, request))
		}
		session, err := generator.prepare(ctx, request)
		if err != nil {
			t.Fatalf("a depth preset request with end %v was refused by the session it loaded: %v", end, err)
		}
		testcheck.Must(generator.integrate(ctx, session))(t)
		testcheck.Must(generator.decode(ctx, session))(t)
	}
	testcheck.NoError(t, generator.Reset(ctx, asking(&whole)))
	if _, err := generator.prepare(ctx, asking(nil)); err == nil {
		t.Fatal("a session loaded for a control ending at 1 admitted a request that leaves its end to the preset's 0.5")
	}
}
