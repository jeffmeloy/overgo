package latentvideo

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/processcontrol"
)

// MP4Encoder streams planar frames through one FFmpeg process.
type MP4Encoder struct {
	supervised                          *processcontrol.Supervised
	wait                                context.Context
	input                               io.WriteCloser
	output                              bytes.Buffer
	errors                              media.DiagnosticBuffer
	raw, prior                          []byte
	pixels                              PixelRange
	fps, height, width, frames, changed int
}

func NewMP4Encoder(ctx context.Context, ffmpeg string, fps, height, width int, pixels PixelRange) (*MP4Encoder, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if !checked.Nonzero(ffmpeg) || !checked.PositiveInts(fps) || !pixels.valid() {
		return nil, errors.New("latent video: incomplete MP4 encoder config")
	}
	if err := media.ValidateSpatialGeometry(height, width); err != nil {
		return nil, fmt.Errorf("latent video: MP4 geometry: %w", err)
	}
	if _, err := os.Stat(ffmpeg); err != nil {
		return nil, fmt.Errorf("latent video: FFmpeg: %w", err)
	}
	encoder := &MP4Encoder{fps: fps, height: height, width: width, pixels: pixels, raw: make([]byte, media.RGBChannels*height*width), wait: ctx}
	frameInput, frameSource := io.Pipe()
	supervised, err := processcontrol.Start(ctx, processcontrol.Command{
		Path: ffmpeg,
		Args: []string{
			"-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
			"-f", "rawvideo", "-pix_fmt", "rgb24", "-s", fmt.Sprintf("%dx%d", width, height),
			"-r", strconv.Itoa(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
			"-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
			"-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1",
		},
		Stdin:  frameInput,
		Stdout: &encoder.output,
		Stderr: &encoder.errors,
	})
	if err != nil {
		_ = frameSource.Close()
		return nil, err
	}
	encoder.supervised = supervised
	encoder.input = frameSource
	return encoder, nil
}

func (s *MP4Encoder) Add(_ int, frame []float32, height, width int) error {
	if s == nil || s.input == nil {
		return errors.New("latent video: invalid MP4 frame")
	}
	if !checked.Equal(height, s.height) {
		return errors.New("latent video: invalid MP4 frame geometry")
	}
	if !checked.Equal(width, s.width) {
		return errors.New("latent video: invalid MP4 frame geometry")
	}
	if err := media.EncodePlanarRGB8Into(s.raw, frame, height, width, func(value float32) uint8 {
		return encodeVideoByte(value, s.pixels)
	}); err != nil {
		return err
	}
	plane := height * width
	for index := range plane {
		base := index * media.RGBChannels
		if s.prior != nil && !bytes.Equal(s.raw[base:base+media.RGBChannels], s.prior[base:base+media.RGBChannels]) {
			s.changed++
		}
	}
	if _, err := s.input.Write(s.raw); err != nil {
		return err
	}
	if s.prior == nil {
		s.prior = make([]byte, len(s.raw))
	}
	copy(s.prior, s.raw)
	s.frames++
	return nil
}

func (s *MP4Encoder) Finish() (EncodedVideo, error) {
	if s == nil || s.input == nil || !checked.PositiveInts(s.frames) {
		return EncodedVideo{}, errors.New("latent video: no MP4 frames emitted")
	}
	if err := s.input.Close(); err != nil {
		return EncodedVideo{}, err
	}
	s.input = nil
	receipt, err := s.supervised.Wait(s.wait)
	s.supervised = nil
	if err != nil || receipt.ExitCode != 0 {
		return EncodedVideo{}, fmt.Errorf("latent video: FFmpeg exit %d: %v: %s", receipt.ExitCode, err, strings.TrimSpace(s.errors.String()))
	}
	return EncodedVideo{
		Data: s.output.Bytes(), MediaType: media.MP4MediaType, Frames: s.frames, Channels: media.RGBChannels,
		Height: s.height, Width: s.width, FPS: s.fps, ChangedPixels: s.changed,
	}, nil
}

// Close stops an unfinished encoder.
func (s *MP4Encoder) Close() error {
	if s == nil {
		return nil
	}
	if s.input != nil {
		_ = s.input.Close()
		s.input = nil
	}
	if s.supervised == nil {
		return nil
	}
	_ = s.supervised.Terminate()
	_, err := s.supervised.Wait(s.wait)
	s.supervised = nil
	return err
}
