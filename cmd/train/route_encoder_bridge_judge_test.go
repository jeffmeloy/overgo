package main

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"overgo/internal/media"
	"overgo/internal/rae"
)

// TestEncoderBridgeJudgmentScoresAFrameDrawnAsItIsAtNoError holds the encoder-bridge
// judge to giving a verdict when a side drew a frame exactly as it is: the frame counts
// at no error on that side, beside what the other side lost, in the means of both sides
// and in the paired difference with its standard error. The judge once refused such a
// frame for want of a ratio to give it, and the route lost the fitted bridge with it.
func TestEncoderBridgeJudgmentScoresAFrameDrawnAsItIsAtNoError(t *testing.T) {
	t.Parallel()
	// The stand-in decoder draws its tokens as the picture, and a bridge of one weight of
	// one and no offset maps tokens to themselves, so each side draws exactly the picture
	// its tokens are and a side is off by as many levels as its tokens were moved.
	moved := func(levels uint8) []float32 {
		picture := make([]float32, 0, 4)
		for _, level := range []uint8{0, 100, 200, 250} {
			picture = append(picture, media.U8ToUnitF32(level+levels))
		}
		return picture
	}
	picture := moved(0)
	frames := []bridgeFrame{
		{picture: picture, target: picture, source: picture},  // both sides exact
		{picture: picture, target: picture, source: moved(1)}, // own exact, bridged one level off
		{picture: picture, target: moved(2), source: picture}, // bridged exact, own two levels off
	}
	decode := func(_ context.Context, tokens []float32) ([]float32, error) { return tokens, nil }
	verdict, err := judgeEncoderBridge(t.Context(), decode, rae.EncoderBridge{Weight: []float32{1}, Bias: []float32{0}}, frames)
	if err != nil {
		t.Fatalf("the judgment was refused: %v", err)
	}
	// The verdict is read as the route prints it, by its keys.
	encoded, err := json.Marshal(verdict)
	if err != nil {
		t.Fatalf("the verdict %+v cannot be printed: %v", verdict, err)
	}
	var printed map[string]float64
	if err := json.Unmarshal(encoded, &printed); err != nil {
		t.Fatal(err)
	}
	// Own errors of 0, 0 and 4 and bridged errors of 0, 1 and 0 differ frame by frame by
	// 0, 1 and -4: a mean of -1 whose squared distances from it sum to 14 over three frames.
	for key, want := range map[string]float64{
		"frames":                               3,
		"own_encoder_mean_squared_level_error": 4.0 / 3,
		"bridged_mean_squared_level_error":     1.0 / 3,
		"bridged_less_own_level_error":         -1,
		"bridged_less_own_standard_error":      math.Sqrt(14.0 / 6),
	} {
		if got, written := printed[key]; !written || math.Abs(got-want) > 1e-9 {
			t.Errorf("%s is printed as %v (written %t) in %s, want %v", key, got, written, encoded, want)
		}
	}
	if residual, written := printed["feature_residual"]; !written || residual <= 0 {
		t.Errorf("the feature residual is printed as %v (written %t) in %s, want the share the moved tokens leave", residual, written, encoded)
	}
	for key := range printed {
		if strings.HasSuffix(key, "_db") {
			t.Errorf("%s is printed in decibels, a unit that has no value for a frame drawn as it is", key)
		}
	}
}
