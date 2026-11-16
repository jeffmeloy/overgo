package evaluation

import (
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
)

// TestOCRAndMotionObservationsBindTheirConsumedMedia holds a suite's readings and motion to the media its cases judge
// (bindObservations): compiled over the bytes a reader read and an encoder measured, the bound suite keeps them, the
// readings under the reference its cases name the picture by, and records their metrics; compiled over media whose
// bytes changed after the observation, or with a measure that names no content, the stale observation is held out and
// the suite still compiles; and the suite's method, how its cases are asked, is the same with its readings as without,
// since what an instrument observed is no part of the asking.
func TestOCRAndMotionObservationsBindTheirConsumedMedia(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	environment := testcheck.Must(runrecord.NewEnvironment(runrecord.Environment{
		Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture",
	}))(t)
	runtime := &mediaInputRuntime{producer: planID(t, artifact.KindRun, "answer producer")}
	campaign := testcheck.Must(NewCampaign(store, runtime, modelrecipe.ProgramIdentity{
		Model:      planID(t, artifact.KindModel, "candidate"),
		Definition: planID(t, artifact.KindModelDefinition, "candidate definition"),
		Recipe:     planID(t, artifact.KindRecipe, "candidate recipe"),
	}, environment, planTestCommit))(t)
	judge := MediaRuntime{Model: planID(t, artifact.KindModel, "judge"), Recipe: planID(t, artifact.KindRecipe, "judge recipe")}
	read, changed := planID(t, artifact.KindFile, "the bytes observed"), planID(t, artifact.KindFile, "the bytes drawn again")
	// compile binds a suite over media of content and gives the bound suite and the method its cases are asked by.
	compile := func(suite GeneratedAnswerSuite, decode MediaDecode, content artifact.ID) (GeneratedAnswerSuite, artifact.ID, []string) {
		t.Helper()
		runtime.input = MediaInput{Content: content, Runtime: judge, Decode: decode}
		compiled := testcheck.Must(campaign.CompileSuite(t.Context(), testcheck.Must(json.Marshal(suite))(t)))(t)
		var bound GeneratedAnswerSuite
		for _, held := range compiled.plan.authorities {
			if held.Descriptor.ID == compiled.plan.body.CaseProfile {
				testcheck.NoError(t, json.Unmarshal(held.Data, &bound))
			}
		}
		var metrics []string
		for _, metric := range generatedAnswerMetrics(bound, 1, 1) {
			metrics = append(metrics, metric.Name)
		}
		return bound, compiled.plan.body.Media.Method, metrics
	}

	reader := MediaRuntime{Model: planID(t, artifact.KindModel, "reader"), Recipe: planID(t, artifact.KindRecipe, "reader recipe"), Projection: planID(t, artifact.KindRecipe, "reader projection")}
	pictures := GeneratedAnswerSuite{Kind: GeneratedAnswerKind, Schema: "fixture", Source: "fixture", Judge: "label",
		Cases: []GeneratedAnswerCase{
			{Name: "title", Prompt: mediaPrompt(MediaImage, "sign.png", "Which text does it hold?\nA. open late\nB. closed"), MaxTokens: exactMaxTokens, Answers: []string{"A"}},
		},
		Readings: &PictureReadings{Reader: reader, Read: map[string]PictureReading{"sign.png": {Content: read.DigestHex(), Text: "open late"}}},
	}
	image := MediaDecode{Format: "image"}
	bound, method, metrics := compile(pictures, image, read)
	_, reference, _, _ := ParseMediaPrompt(bound.Cases[0].Prompt)
	if bound.Readings == nil || len(bound.Readings.Read) != 1 || bound.Readings.Read[reference].Text != "open late" || !slices.Contains(metrics, CharacterMetric) {
		t.Fatalf("readings of the bytes judged bound as %+v with metrics %v; want them under %s, the case's reference, recorded", bound.Readings, metrics, reference)
	}
	stale, _, staleMetrics := compile(pictures, image, changed)
	if stale.Readings != nil || slices.Contains(staleMetrics, CharacterMetric) {
		t.Fatalf("readings of bytes since drawn again bound as %+v with metrics %v; want them held out", stale.Readings, staleMetrics)
	}
	unread := pictures
	unread.Readings = nil
	if _, unreadMethod, _ := compile(unread, image, read); unreadMethod != method {
		t.Fatal("the readings moved the method the cases are asked by")
	}

	measure := FrameSimilarity{Neighbours: []float64{0.9, 0.8}, Mean: 0.85, Lowest: 0.8, Content: read.DigestHex()}
	clips := GeneratedAnswerSuite{Kind: GeneratedAnswerKind, Schema: "fixture", Source: "fixture", Judge: "label",
		Cases: []GeneratedAnswerCase{
			{Name: "dog", Group: "a dog runs", Prompt: mediaPrompt(MediaVideo, "dog.mp4", "Does a dog run?"), MaxTokens: exactMaxTokens, Answers: []string{"yes"}},
		},
		Motion: &ClipMotion{Encoder: "fixture", Clips: map[string]FrameSimilarity{"a dog runs": measure}},
	}
	video := MediaDecode{Format: "gif", FPS: 16, Frames: 81}
	if bound, _, metrics := compile(clips, video, read); bound.Motion == nil || !slices.Contains(metrics, FrameSimilarityMetric) {
		t.Fatalf("motion of the clip judged bound as %+v with metrics %v; want it kept and recorded", bound.Motion, metrics)
	}
	if stale, _, metrics := compile(clips, video, changed); stale.Motion != nil || slices.Contains(metrics, FrameSimilarityMetric) {
		t.Fatalf("motion of a clip since drawn again bound as %+v with metrics %v; want it held out", stale.Motion, metrics)
	}
	measure.Content = ""
	clips.Motion.Clips["a dog runs"] = measure
	if unnamed, _, _ := compile(clips, video, read); unnamed.Motion != nil {
		t.Fatal("a motion measure naming no content bound to the clip judged; want it held out")
	}
}
