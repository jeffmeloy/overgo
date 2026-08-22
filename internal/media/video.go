package media

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/gif"
	"image/png"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/tensor"
)

const (
	GIFMediaType = "image/gif"
	MP4MediaType = "video/mp4"
)

// GIFFrameDelay converts frames per second to GIF centisecond delay units.
func GIFFrameDelay(framesPerSecond int) int { return max(1, 100/framesPerSecond) }

// ValidateEncodedRGBVideo validates a typed encoded-video publication.
func ValidateEncodedRGBVideo(data []byte, mediaType, expectedMediaType string, frames, channels, height, width, fps int) error {
	if len(data) == 0 || mediaType != expectedMediaType || !checked.PositiveInts(frames, height, width, fps) || channels != RGBChannels {
		return errors.New("media: invalid encoded RGB video")
	}
	return nil
}

// ValidatePlanarVideo validates flat channel-major video storage.
func ValidatePlanarVideo[T any](pixels []T, channels, frames, height, width int) error {
	if !checked.PositiveInts(channels, frames, height, width) {
		return errors.New("media: invalid planar video geometry")
	}
	if err := checked.Length(pixels, channels, frames, height, width); err != nil {
		return fmt.Errorf("media: planar video: %w", err)
	}
	return nil
}

// CopyPlanarFrames copies a contiguous temporal span between channel-major
// planar video tensors.
func CopyPlanarFrames[T any](destination []T, destinationFrames, destinationStart int, source []T, sourceFrames, sourceStart, channels, frames, spatial int) error {
	if !checked.PositiveInts(destinationFrames, sourceFrames, channels, spatial) || !checked.NonNegativeInts(destinationStart, sourceStart, frames) {
		return errors.New("media: invalid planar frame copy")
	}
	destinationEnd, ok := checked.AddInt(destinationStart, frames)
	if !ok || destinationEnd > destinationFrames {
		return errors.New("media: destination frame copy is out of range")
	}
	sourceEnd, ok := checked.AddInt(sourceStart, frames)
	if !ok || sourceEnd > sourceFrames {
		return errors.New("media: source frame copy is out of range")
	}
	destinationElements := channels * destinationFrames * spatial
	sourceElements := channels * sourceFrames * spatial
	if len(destination) < destinationElements || len(source) < sourceElements {
		return errors.New("media: planar frame storage is too short")
	}
	for channel := range channels {
		destinationOffset := (channel*destinationFrames + destinationStart) * spatial
		sourceOffset := (channel*sourceFrames + sourceStart) * spatial
		copy(destination[destinationOffset:destinationOffset+frames*spatial], source[sourceOffset:sourceOffset+frames*spatial])
	}
	return nil
}

func DecodeGIF(reader io.Reader, maximum int) ([]image.Image, error) {
	if reader == nil || !checked.PositiveInts(maximum) {
		return nil, errors.New("media: GIF reader or frame limit is invalid")
	}
	decoded, err := gif.DecodeAll(reader)
	if err != nil {
		return nil, err
	}
	if len(decoded.Image) == tensor.FirstOffset || !checked.PositiveInts(decoded.Config.Width, decoded.Config.Height) {
		return nil, errors.New("media: GIF has no frames")
	}
	bounds := image.Rect(tensor.FirstOffset, tensor.FirstOffset, decoded.Config.Width, decoded.Config.Height)
	background := color.Color(color.Transparent)
	if palette, ok := decoded.Config.ColorModel.(color.Palette); ok && int(decoded.BackgroundIndex) < len(palette) {
		background = palette[decoded.BackgroundIndex]
	}
	canvas := image.NewRGBA(bounds)
	draw.Draw(canvas, bounds, image.NewUniform(background), image.Point{}, draw.Src)
	frames := make([]image.Image, tensor.FirstOffset, len(decoded.Image))
	for index, frame := range decoded.Image {
		if frame == nil || !frame.Bounds().In(bounds) {
			return nil, errors.New("media: GIF frame geometry is invalid")
		}
		before := CloneRGBA(canvas)
		draw.Draw(canvas, frame.Bounds(), frame, frame.Bounds().Min, draw.Over)
		frames = append(frames, CloneRGBA(canvas))
		disposal := byte(gif.DisposalNone)
		if index < len(decoded.Disposal) {
			disposal = decoded.Disposal[index]
		}
		switch disposal {
		case gif.DisposalBackground:
			draw.Draw(canvas, frame.Bounds(), image.NewUniform(background), image.Point{}, draw.Src)
		case gif.DisposalPrevious:
			canvas = before
		}
	}
	return sampleFrames(frames, maximum), nil
}

func sampleFrames(frames []image.Image, maximum int) []image.Image {
	if len(frames) <= maximum {
		return frames
	}
	result := make([]image.Image, maximum)
	if maximum == tensor.SingletonExtent {
		result[tensor.FirstOffset] = frames[tensor.FirstOffset]
		return result
	}
	for index := range result {
		source := index * (len(frames) - tensor.SingletonExtent) / (maximum - tensor.SingletonExtent)
		result[index] = frames[source]
	}
	return result
}

func DecodeEncodedVideo(ctx context.Context, data []byte, configuredFFmpeg string, fps float64, maximum int) ([]image.Image, error) {
	if len(data) == tensor.FirstOffset {
		return nil, errors.New("media: encoded video is empty")
	}
	if err := validateVideoDecode(fps, maximum); err != nil {
		return nil, err
	}
	if bytes.HasPrefix(data, []byte("GIF87a")) || bytes.HasPrefix(data, []byte("GIF89a")) {
		return DecodeGIF(bytes.NewReader(data), maximum)
	}
	ffmpeg, err := ResolveFFmpeg(configuredFFmpeg)
	if err != nil {
		return nil, err
	}
	return decodeFFmpegFrames(ctx, ffmpeg, "pipe:0", bytes.NewReader(data), fps, maximum)
}

func DecodeVideoFile(ctx context.Context, path, configuredFFmpeg string, fps float64, maximum int) ([]image.Image, error) {
	if err := validateVideoDecode(fps, maximum); err != nil {
		return nil, err
	}
	absolutePath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	if strings.EqualFold(filepath.Ext(path), ".gif") {
		file, err := os.Open(absolutePath)
		if err != nil {
			return nil, err
		}
		frames, decodeErr := DecodeGIF(file, maximum)
		return frames, errors.Join(decodeErr, file.Close())
	}
	ffmpeg, err := ResolveFFmpeg(configuredFFmpeg)
	if err != nil {
		return nil, err
	}
	return decodeFFmpegFrames(ctx, ffmpeg, absolutePath, nil, fps, maximum)
}

func validateVideoDecode(fps float64, maximum int) error {
	if !checked.PositiveInts(maximum) || !checked.PositiveFinite64(fps) {
		return errors.New("media: video FPS or frame limit is invalid")
	}
	return nil
}

func decodeFFmpegFrames(ctx context.Context, ffmpeg, input string, stdin io.Reader, fps float64, maximum int) ([]image.Image, error) {
	filter := "fps=" + strconv.FormatFloat(fps, 'g', -tensor.SingletonExtent, float64Bits)
	command := exec.CommandContext(
		ctx, ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
		"-i", input, "-vf", filter, "-frames:v", strconv.Itoa(maximum),
		"-f", "image2pipe", "-vcodec", "png", "pipe:1",
	)
	command.Stdin = stdin
	stdout, err := command.StdoutPipe()
	if err != nil {
		return nil, err
	}
	var stderr DiagnosticBuffer
	command.Stderr = &stderr
	if err := command.Start(); err != nil {
		return nil, err
	}
	reader := bufio.NewReader(stdout)
	frames := make([]image.Image, tensor.FirstOffset, maximum)
	for len(frames) < maximum {
		frame, decodeErr := png.Decode(reader)
		if errors.Is(decodeErr, io.EOF) || errors.Is(decodeErr, io.ErrUnexpectedEOF) {
			break
		}
		if decodeErr != nil {
			_ = command.Process.Kill()
			_ = command.Wait()
			return nil, decodeErr
		}
		frames = append(frames, frame)
	}
	if err := command.Wait(); err != nil {
		return nil, fmt.Errorf("media: FFmpeg: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if len(frames) == tensor.FirstOffset {
		return nil, errors.New("media: FFmpeg produced no video frames")
	}
	return frames, nil
}

func ResolveFFmpeg(configured string) (string, error) {
	if strings.TrimSpace(configured) != "" {
		if _, err := os.Stat(configured); err != nil {
			return "", fmt.Errorf("media: FFmpeg executable: %w", err)
		}
		return configured, nil
	}
	if path, err := exec.LookPath("ffmpeg"); err == nil {
		return path, nil
	}
	if runtime.GOOS == "windows" {
		if programFiles := os.Getenv("ProgramFiles"); programFiles != "" {
			candidate := filepath.Join(programFiles, "DownloadHelper CoApp", "ffmpeg.exe")
			if _, err := os.Stat(candidate); err == nil {
				return candidate, nil
			}
		}
	}
	return "", errors.New("media: FFmpeg is unavailable")
}
