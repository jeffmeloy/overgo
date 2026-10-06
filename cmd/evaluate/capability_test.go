package main

import (
	"testing"

	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/seriesforecast"
)

// TestADirectoryThatIsNoFractaleModelOpensNoThoughtBankSession holds the
// dispatch to what it claims: a path that is not a Fractale generation model
// is reported as not one, with nothing opened, so evaluation falls through to
// the model's own runner.
func TestADirectoryThatIsNoFractaleModelOpensNoThoughtBankSession(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	session, opened, err := openThoughtBankSession(t.Context(), store, manifest{}, t.TempDir())
	if session != nil || opened || err != nil {
		t.Fatalf("an empty directory opened (%v, %t, %v), want no session and no error", session, opened, err)
	}
}

// TestCapabilityTasksAreTheOnesASessionExists holds the manifest's task names
// to the capabilities evaluate opens, and a capability to the runtime it is:
// a forecasting model forecasts no more steps than its horizon and generates
// and scores no text.
func TestCapabilityTasksAreTheOnesASessionExists(t *testing.T) {
	t.Parallel()
	if len(capabilitySessions) == 0 {
		t.Fatal("no capability session is named")
	}
	for _, task := range []string{"seq2seq", "forecast", "vqa"} {
		if capabilitySessions[task] == nil {
			t.Errorf("task %q has no capability session", task)
		}
	}
	vqa := &vqaRuntime{}
	if _, _, err := vqa.Generate(t.Context(), "Which description matches?", inference.GenerateOptions{}); err == nil {
		t.Error("a prompt that names no image was answered")
	}
	if _, _, err := vqa.Generate(t.Context(), "<overgo-media kind=video name=a.mp4>\nWhich?", inference.GenerateOptions{}); err == nil {
		t.Error("a video was answered by a model that reads one image")
	}
	if _, _, err := vqa.Generate(t.Context(), "<overgo-media kind=image name=../a.png>\nWhich?", inference.GenerateOptions{}); err == nil {
		t.Error("an image outside the media root was answered")
	}
	if _, err := vqa.ScoreContinuations(t.Context(), "text", []string{"candidate"}); err == nil {
		t.Error("a vqa evaluation scored a continuation")
	}
	runtime := forecastRuntime{model: &seriesforecast.Model{Dims: seriesforecast.Dims{Horizon: 4, Quantiles: 2}}}
	for _, steps := range []int{0, 5} {
		if _, err := runtime.Forecast([]float32{1}, steps); err == nil {
			t.Errorf("%d steps were forecast by a model whose horizon is 4", steps)
		}
	}
	if _, _, err := runtime.Generate(t.Context(), "text", inference.GenerateOptions{}); err == nil {
		t.Error("a forecasting model generated text")
	}
	if _, err := runtime.ScoreContinuations(t.Context(), "text", []string{"candidate"}); err == nil {
		t.Error("a forecasting model scored a continuation")
	}
}
