package projector

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

const maxFFmpegErrorBytes = 64 << 10

// DecodeEncodedVideo: bounded GIF or FFmpeg frame decode.
func DecodeEncodedVideo(
	ctx context.Context,
	data []byte,
	configuredFFmpeg string,
	fps float64,
	maxFrames int,
) ([]image.Image, error) {
	if len(data) == 0 {
		return nil, errors.New("projector: encoded video is empty")
	}
	if maxFrames <= 0 || fps <= 0 || math.IsNaN(fps) || math.IsInf(fps, 0) {
		return nil, errors.New("projector: video FPS or frame limit is invalid")
	}
	if bytes.HasPrefix(data, []byte("GIF87a")) || bytes.HasPrefix(data, []byte("GIF89a")) {
		return DecodeGIFVideo(bytes.NewReader(data), maxFrames)
	}
	ffmpeg, err := ResolveFFmpeg(configuredFFmpeg)
	if err != nil {
		return nil, err
	}
	filter := "fps=" + strconv.FormatFloat(fps, 'g', -1, 64)
	command := exec.CommandContext(
		ctx, ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
		"-i", "pipe:0", "-vf", filter, "-frames:v", strconv.Itoa(maxFrames),
		"-f", "image2pipe", "-vcodec", "png", "pipe:1",
	)
	command.Stdin = bytes.NewReader(data)
	stdout, err := command.StdoutPipe()
	if err != nil {
		return nil, err
	}
	var stderr boundedVideoError
	command.Stderr = &stderr
	if err := command.Start(); err != nil {
		return nil, err
	}
	reader := bufio.NewReader(stdout)
	frames := make([]image.Image, 0, maxFrames)
	for len(frames) < maxFrames {
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
		return nil, fmt.Errorf("projector: FFmpeg: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if len(frames) == 0 {
		return nil, errors.New("projector: FFmpeg produced no video frames")
	}
	return frames, nil
}

// ResolveFFmpeg: configured, PATH, or known Windows executable.
func ResolveFFmpeg(configured string) (string, error) {
	if strings.TrimSpace(configured) != "" {
		if _, err := os.Stat(configured); err != nil {
			return "", fmt.Errorf("projector: FFmpeg executable: %w", err)
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
	return "", errors.New("projector: FFmpeg is unavailable")
}

type boundedVideoError struct {
	data []byte
}

func (w *boundedVideoError) Write(data []byte) (int, error) {
	written := len(data)
	remaining := maxFFmpegErrorBytes - len(w.data)
	if remaining > 0 {
		w.data = append(w.data, data[:min(len(data), remaining)]...)
	}
	return written, nil
}

func (w *boundedVideoError) String() string {
	return string(w.data)
}
