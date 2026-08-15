package latentvideo

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

const encodedMP4MediaType = "video/mp4"

// MP4Encoder streams planar frames through one FFmpeg process.
type MP4Encoder struct {
	command                             *exec.Cmd
	input                               io.WriteCloser
	output                              bytes.Buffer
	errors                              boundedEncoderError
	raw, prior                          []byte
	pixels                              PixelRange
	fps, height, width, frames, changed int
}

func NewMP4Encoder(ctx context.Context, ffmpeg string, fps, height, width int, pixels PixelRange) (*MP4Encoder, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if ffmpeg == "" || fps <= 0 || height <= 0 || width <= 0 || !pixels.valid() {
		return nil, errors.New("latent video: incomplete MP4 encoder config")
	}
	if _, err := os.Stat(ffmpeg); err != nil {
		return nil, fmt.Errorf("latent video: FFmpeg: %w", err)
	}
	encoder := &MP4Encoder{fps: fps, height: height, width: width, pixels: pixels, raw: make([]byte, 3*height*width)}
	encoder.command = exec.CommandContext(ctx, ffmpeg,
		"-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
		"-f", "rawvideo", "-pix_fmt", "rgb24", "-s", fmt.Sprintf("%dx%d", width, height),
		"-r", strconv.Itoa(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
		"-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
		"-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1",
	)
	encoder.command.Stdout, encoder.command.Stderr = &encoder.output, &encoder.errors
	input, err := encoder.command.StdinPipe()
	if err != nil {
		return nil, err
	}
	encoder.input = input
	if err := encoder.command.Start(); err != nil {
		_ = input.Close()
		return nil, err
	}
	return encoder, nil
}

func (s *MP4Encoder) Add(_ int, frame []float32, height, width int) error {
	if s == nil || s.input == nil || height != s.height || width != s.width || len(frame) != len(s.raw) {
		return errors.New("latent video: invalid MP4 frame")
	}
	plane := height * width
	for index := range plane {
		for channel := range 3 {
			value := frame[channel*plane+index]
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return errors.New("latent video: non-finite MP4 frame")
			}
			s.raw[index*3+channel] = encodeVideoByte(value, s.pixels)
		}
		if s.prior != nil && (s.raw[index*3] != s.prior[index*3] || s.raw[index*3+1] != s.prior[index*3+1] || s.raw[index*3+2] != s.prior[index*3+2]) {
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
	if s == nil || s.input == nil || s.frames == 0 {
		return EncodedVideo{}, errors.New("latent video: no MP4 frames emitted")
	}
	if err := s.input.Close(); err != nil {
		return EncodedVideo{}, err
	}
	s.input = nil
	err := s.command.Wait()
	s.command = nil
	if err != nil {
		return EncodedVideo{}, fmt.Errorf("latent video: FFmpeg: %w: %s", err, strings.TrimSpace(s.errors.String()))
	}
	return EncodedVideo{
		Data: s.output.Bytes(), MediaType: encodedMP4MediaType, Frames: s.frames, Channels: 3,
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
	if s.command == nil || s.command.Process == nil {
		return nil
	}
	_ = s.command.Process.Kill()
	err := s.command.Wait()
	s.command = nil
	return err
}

type boundedEncoderError struct{ data []byte }

func (w *boundedEncoderError) Write(data []byte) (int, error) {
	written := len(data)
	remaining := 64<<10 - len(w.data)
	if remaining > 0 {
		w.data = append(w.data, data[:min(len(data), remaining)]...)
	}
	return written, nil
}

func (w *boundedEncoderError) String() string { return string(w.data) }
