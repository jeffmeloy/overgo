package latentvideo

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/media"
	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// TestAnMP4IsConvertedAndTaggedAsBT709 holds every MP4 overgo writes to the colours drawn (clip-encode/colour-709):
// FFmpeg converted its RGB by its BT.601 default and wrote no colour tags, so a player, reading untagged HD as BT.709,
// shifted reds and greens. A clip of saturated red, green, blue and yellow quadrants is written through WriteClip; FFmpeg
// reads its stream as yuv420p limited range BT.709, and overgo's own decoder returns each quadrant's middle within
// three 8-bit levels of what was drawn.
func TestAnMP4IsConvertedAndTaggedAsBT709(t *testing.T) {
	t.Parallel()
	ffmpeg, err := media.FFmpeg()
	if err != nil {
		testskip.NotApplicable(t, "no FFmpeg to write an MP4 with: "+err.Error())
	}
	const side, frames, fps, tolerance = 64, 2, 8, 3
	quadrants := [4][3]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}}
	quadrant := func(x, y int) int { return y/(side/2)*2 + x/(side/2) }
	plane := side * side
	frame := make([]float32, 3*plane)
	for y := range side {
		for x := range side {
			for channel, value := range quadrants[quadrant(x, y)] {
				frame[channel*plane+y*side+x] = value
			}
		}
	}
	video := testcheck.Must(WriteClip(t.Context(), mp4Clip, ClipFinish{}, fps, UnitPixels, func(sink VideoFrameSink) error {
		for index := range frames {
			if err := sink(index, frame, side, side); err != nil {
				return err
			}
		}
		return nil
	}))(t)
	path := filepath.Join(t.TempDir(), "patches.mp4")
	testcheck.NoError(t, os.WriteFile(path, video.Data, 0o600))
	var described bytes.Buffer
	_, _ = processcontrol.Run(context.Background(), processcontrol.Command{Path: ffmpeg, Args: []string{"-hide_banner", "-i", path}, Stderr: &described})
	if !strings.Contains(described.String(), "yuv420p(tv, bt709") {
		t.Errorf("FFmpeg reads the clip as\n%s\nwant a yuv420p stream tagged limited range BT.709", described.String())
	}
	pictures := testcheck.Must(media.DecodeVideoFile(t.Context(), path, ffmpeg, fps, frames))(t)
	for corner, drawn := range quadrants {
		x, y := side/4+corner%2*side/2, side/4+corner/2*side/2
		red, green, blue, _ := pictures[0].At(x, y).RGBA()
		for channel, read := range []uint32{red >> 8, green >> 8, blue >> 8} {
			want := int(drawn[channel] * 255)
			if difference := int(read) - want; difference > tolerance || difference < -tolerance {
				t.Errorf("quadrant %d channel %d decodes as %d, drawn as %d", corner, channel, read, want)
			}
		}
	}
}
