package latentvideo

import (
	"slices"
	"testing"
)

func TestLiveEditSourceCodecBoundary(t *testing.T) {
	profile := SourceCodecProfile{
		InputChannels: 3, LatentChannels: 2, Stride: [3]int{4, 8, 8},
		LatentStats: VAELatentStats{Mean: []float32{1, -2}, Std: []float32{2, 4}},
	}
	plan, err := CompileSourceCodecBoundary(profile, SourceVideoShape{Channels: 3, Frames: 5, Height: 16, Width: 24})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Latent != (SourceVideoShape{Channels: 2, Frames: 2, Height: 2, Width: 3}) ||
		!slices.Equal(plan.SourceChunks, []int{1, 4}) {
		t.Fatalf("source plan = %+v", plan)
	}
	meanOutput := make([]float32, 24)
	want := make([]float32, 24)
	for index := range 12 {
		meanOutput[index] = 1 + 2*float32(index)
		meanOutput[12+index] = -2 + 4*float32(index)
		want[index], want[12+index] = float32(index), float32(index)
	}
	got := make([]float32, len(meanOutput))
	if err := NormalizeSourceLatent(got, meanOutput, plan); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("normalized latent = %v, want %v", got, want)
	}

	spatial := plan.Source.Height * plan.Source.Width
	source := make([]float32, plan.Source.Channels*plan.Source.Frames*spatial)
	for index := range source {
		source[index] = float32(index)
	}
	chunk := make([]float32, plan.Source.Channels*plan.SourceChunks[1]*spatial)
	if err := copyChannelFrames(chunk, plan.SourceChunks[1], 0, source, plan.Source.Frames, 1, plan.Source.Channels, plan.SourceChunks[1], spatial); err != nil {
		t.Fatal(err)
	}
	for channel := range plan.Source.Channels {
		wantStart := (channel*plan.Source.Frames + 1) * spatial
		gotStart := channel * plan.SourceChunks[1] * spatial
		if !slices.Equal(chunk[gotStart:gotStart+plan.SourceChunks[1]*spatial], source[wantStart:wantStart+plan.SourceChunks[1]*spatial]) {
			t.Fatalf("channel %d frame span changed layout", channel)
		}
	}
}
