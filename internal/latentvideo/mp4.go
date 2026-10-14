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

// chromaSubsampling is how many pixels a side of one chroma sample covers in
// the 4:2:0 frames an MP4 clip is written in (FFmpeg's yuv420p), so a frame's
// width and height are multiples of it. It holds while clips are written as
// yuv420p.
const chromaSubsampling = 2

// mp4Encoder streams planar frames through one FFmpeg process, which starts
// at the first frame.
type mp4Encoder struct {
	ffmpeg                              string
	supervised                          *processcontrol.Supervised
	wait                                context.Context
	input                               io.WriteCloser
	output                              bytes.Buffer
	errors                              media.DiagnosticBuffer
	raw, prior                          []byte
	pixels                              PixelRange
	fps, height, width, frames, changed int
}

// NewMP4Encoder writes a clip as H.264 in a fragmented MP4 through the FFmpeg
// at ffmpeg.
//   - FFmpeg starts at the first frame, whose size every later frame keeps, because a generator's frames are its decoder's size, which a request off the decoder's stride does not name
//   - a frame of an odd width or height is refused, because each 4:2:0 chroma sample covers two pixels a side.
func NewMP4Encoder(ctx context.Context, ffmpeg string, fps int, pixels PixelRange) (*mp4Encoder, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if !checked.Nonzero(ffmpeg) || !checked.PositiveInts(fps) || !pixels.valid() {
		return nil, errors.New("latent video: incomplete MP4 encoder config")
	}
	if _, err := os.Stat(ffmpeg); err != nil {
		return nil, fmt.Errorf("latent video: FFmpeg: %w", err)
	}
	return &mp4Encoder{ffmpeg: ffmpeg, fps: fps, pixels: pixels, wait: ctx}, nil
}

// start starts FFmpeg for frames of height by width.
//   - the frame pipe closes when FFmpeg exits, so a frame offered to an FFmpeg that failed is refused and not left waiting
func (s *mp4Encoder) start(height, width int) error {
	if err := media.ValidateSpatialGeometry(height, width); err != nil {
		return fmt.Errorf("latent video: MP4 geometry: %w", err)
	}
	if height%chromaSubsampling != 0 || width%chromaSubsampling != 0 {
		return fmt.Errorf("latent video: an MP4 frame is %dx%d, and its sides must be multiples of %d", width, height, chromaSubsampling)
	}
	frameInput, frameSource := io.Pipe()
	supervised, err := processcontrol.Start(s.wait, processcontrol.Command{
		Path: s.ffmpeg,
		Args: []string{
			"-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
			"-f", "rawvideo", "-pix_fmt", "rgb24", "-s", fmt.Sprintf("%dx%d", width, height),
			"-r", strconv.Itoa(s.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
			"-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
			"-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1",
		},
		Stdin:  frameInput,
		Stdout: &s.output,
		Stderr: &s.errors,
	})
	if err != nil {
		_ = frameSource.Close()
		return err
	}
	go func() {
		<-supervised.Done()
		_ = frameInput.CloseWithError(errors.New("latent video: FFmpeg exited before the clip's last frame"))
	}()
	s.supervised, s.input = supervised, frameSource
	s.height, s.width, s.raw = height, width, make([]byte, media.RGBChannels*height*width)
	return nil
}

// Add takes the next frame of the clip.
func (s *mp4Encoder) Add(index int, frame []float32, height, width int) error {
	return s.add(context.Background(), index, frame, height, width)
}

// add writes the frame to FFmpeg and counts the pixels that changed since the
// frame before; the first frame starts FFmpeg at its size.
func (s *mp4Encoder) add(_ context.Context, index int, frame []float32, height, width int) error {
	if s == nil {
		return errors.New("latent video: invalid MP4 frame")
	}
	if index != s.frames {
		return errors.New("latent video: frame index is out of order")
	}
	if s.frames == 0 && s.supervised == nil {
		if err := s.start(height, width); err != nil {
			return err
		}
	}
	if s.input == nil {
		return errors.New("latent video: invalid MP4 frame")
	}
	if !checked.Equal(height, s.height) || !checked.Equal(width, s.width) {
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
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(s.errors.String()))
	}
	if s.prior == nil {
		s.prior = make([]byte, len(s.raw))
	}
	copy(s.prior, s.raw)
	s.frames++
	return nil
}

func (s *mp4Encoder) Finish() (EncodedVideo, error) {
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
func (s *mp4Encoder) Close() error {
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
