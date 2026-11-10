package latentvideo

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/processcontrol"
)

// chromaSubsampling is how many pixels a side of one chroma sample covers in
// the 4:2:0 frames an MP4 clip is written in (FFmpeg's yuv420p), so a frame's
// width and height are multiples of it. It holds while clips are written as
// yuv420p.
const chromaSubsampling = 2

// x264Codec is a clip's encoder unless it is finished: x264's slow preset at a constant rate factor of 14, one thread.
// On two VidGen clips cut to 81 frames at 832 x 480 it read back at 40.1 and 40.4 dB in 2.6 and 2.4 MB, where ultrafast
// at 18 read 38.4 and 38.6 in 3.1 and 2.7 MB (ultrafast turns off the deblocking filter and adaptive quantization),
// against a ceiling of 42.4 and 43.8 dB that 4:2:0 itself sets. One encoder thread, so the same frames are the same bytes
// on any machine: -threads binds the input or output it precedes, so it follows the input and sits with the encoder's
// own options; before -i it set the rawvideo reader's threads and x264 ran 15 frame threads and 2 lookahead threads
// (x264's own record in the clip).
var x264Codec = []string{"-c:v", "libx264", "-threads", "1", "-preset", "slow", "-crf", "14"}

// nvencCodec encodes a finished clip, up to 3328 x 1920, on the card's NVENC as HEVC at a constant quality, as the
// owner's upscaler does (upscale_video.py open_encoder: VBR, b:v 0, an 80 Mb/s ceiling, its 4x example's quality 18),
// at the slowest preset, and tagged hvc1 so players that require the tag open it. On this RTX 4090 D two encodes of one
// 80-frame 3328 x 1920 clip were the same bytes (2026-11-09), so a finished clip stays one clip.
var nvencCodec = []string{"-c:v", "hevc_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", "18", "-b:v", "0", "-maxrate", "80M",
	"-bufsize", "160M", "-tag:v", "hvc1"}

// bt709Conversion converts a clip's RGB to 4:2:0 YUV by BT.709's limited-range coefficients and tags the stream with
// BT.709's primaries, transfer, matrix and range, so a player shows the colours drawn: with no colour options FFmpeg's
// scaler converts by BT.601 and writes no tags, and a player shows untagged HD as BT.709 (clip-encode/colour-709).
var bt709Conversion = []string{"-vf", "scale=out_color_matrix=bt709:out_range=tv", "-pix_fmt", "yuv420p",
	"-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"}

// nvencEncoder is the name FFmpeg lists NVENC's HEVC encoder by.
const nvencEncoder = "hevc_nvenc"

// mp4Encoder streams planar frames through one FFmpeg process, which starts
// at the first frame.
type mp4Encoder struct {
	ffmpeg                              string
	encoding                            media.Encoding
	supervised                          *processcontrol.Supervised
	wait                                context.Context
	input                               io.WriteCloser
	output                              bytes.Buffer
	errors                              media.DiagnosticBuffer
	raw, prior                          []byte
	pixels                              PixelRange
	fps, height, width, frames, changed int
	// codec is the encoder's options, x264Codec where none is set.
	codec []string
}

// newFinishedMP4Encoder writes a finished clip on NVENC where the FFmpeg at ffmpeg lists it, and with x264 otherwise,
// saying which and why.
func newFinishedMP4Encoder(ctx context.Context, ffmpeg string, fps int, pixels PixelRange) (*mp4Encoder, string, error) {
	encoder, err := NewMP4Encoder(ctx, ffmpeg, fps, pixels)
	if err != nil {
		return nil, "", err
	}
	listed, err := ffmpegEncoders(ffmpeg)
	if err != nil {
		return nil, "", err
	}
	if !slices.Contains(listed, nvencEncoder) {
		return encoder, "libx264: the FFmpeg lists no " + nvencEncoder, nil
	}
	encoder.codec = nvencCodec
	return encoder, nvencEncoder, nil
}

// ffmpegEncoders are the encoders an FFmpeg lists (ffmpeg -encoders), read once for each executable.
var ffmpegEncoders = func() func(string) ([]string, error) {
	var mu sync.Mutex
	read := map[string][]string{}
	return func(ffmpeg string) ([]string, error) {
		mu.Lock()
		defer mu.Unlock()
		if listed, known := read[ffmpeg]; known {
			return listed, nil
		}
		var output, diagnostic bytes.Buffer
		receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{Path: ffmpeg,
			Args: []string{"-hide_banner", "-encoders"}, Stdout: &output, Stderr: &diagnostic})
		if err == nil && receipt.ExitCode != 0 {
			err = fmt.Errorf("latent video: %s -encoders exited %d: %s", ffmpeg, receipt.ExitCode, diagnostic.String())
		}
		if err != nil {
			return nil, err
		}
		var listed []string
		for line := range strings.Lines(output.String()) {
			// Each encoder's line is its capability flags, then its name.
			if fields := strings.Fields(line); len(fields) > 1 {
				listed = append(listed, fields[1])
			}
		}
		read[ffmpeg] = listed
		return listed, nil
	}
}()

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
	// The executable is named by its bytes, since a path names whatever is installed there later.
	tool, err := artifact.IdentifyFile(artifact.KindFile, ffmpeg)
	if err != nil {
		return nil, fmt.Errorf("latent video: FFmpeg: %w", err)
	}
	return &mp4Encoder{ffmpeg: ffmpeg, encoding: media.Encoding{Tool: tool}, fps: fps, pixels: pixels, wait: ctx}, nil
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
	codec := s.codec
	if len(codec) == 0 {
		codec = x264Codec
	}
	s.encoding.Args = append(append(append([]string{
		"-nostdin", "-hide_banner", "-loglevel", "error",
		"-f", "rawvideo", "-pix_fmt", "rgb24", "-s", fmt.Sprintf("%dx%d", width, height),
		"-r", strconv.Itoa(s.fps), "-i", "pipe:0", "-an",
	}, codec...), bt709Conversion...), "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1")
	frameInput, frameSource := io.Pipe()
	supervised, err := processcontrol.Start(s.wait, processcontrol.Command{
		Path:   s.ffmpeg,
		Args:   s.encoding.Args,
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
		Height: s.height, Width: s.width, FPS: s.fps, ChangedPixels: s.changed, Encoding: s.encoding,
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
