package latentvideo

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/processcontrol"
)

// CaptionedClip binds one on-disk clip to its corpus caption, so DiT
// training conditions on real video and real text rather than fixture
// latents.
type CaptionedClip struct {
	Name    string
	Path    string
	Caption string
}

// vidGenCaptionFile is the corpus's own caption manifest name; the
// format is the published VidGen convention, not an overgo invention.
const vidGenCaptionFile = "VidGen_1M_video_caption.json"

// ListCaptionedClips pairs the corpus directory's clips with their
// recorded captions, in deterministic name order, up to limit. The
// caption manifest is streamed -- a million-row corpus never loads
// whole -- and a clip without a caption is skipped rather than trained
// on with fabricated conditioning.
func ListCaptionedClips(root string, limit int) ([]CaptionedClip, error) {
	if !checked.PositiveInts(limit) {
		return nil, errors.New("vidgen: clip limit must be positive")
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	wanted := map[string]string{}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".mp4") {
			continue
		}
		wanted[strings.TrimSuffix(name, ".mp4")] = filepath.Join(root, name)
	}
	if len(wanted) == 0 {
		return nil, fmt.Errorf("vidgen: %q holds no clips", root)
	}
	manifest, err := os.Open(filepath.Join(root, vidGenCaptionFile))
	if err != nil {
		return nil, err
	}
	defer manifest.Close()
	decoder := json.NewDecoder(manifest)
	if _, err := decoder.Token(); err != nil {
		return nil, fmt.Errorf("vidgen: caption manifest: %w", err)
	}
	captions := map[string]string{}
	for decoder.More() {
		var row struct {
			Vid     string `json:"vid"`
			Caption string `json:"caption"`
		}
		if err := decoder.Decode(&row); err != nil {
			return nil, fmt.Errorf("vidgen: caption manifest: %w", err)
		}
		if _, present := wanted[row.Vid]; present && strings.TrimSpace(row.Caption) != "" {
			captions[row.Vid] = row.Caption
		}
	}
	names := make([]string, 0, len(captions))
	for name := range captions {
		names = append(names, name)
	}
	slices.Sort(names)
	clips := make([]CaptionedClip, 0, min(limit, len(names)))
	for _, name := range names {
		if len(clips) == limit {
			break
		}
		clips = append(clips, CaptionedClip{Name: name, Path: wanted[name], Caption: captions[name]})
	}
	if len(clips) == 0 {
		return nil, fmt.Errorf("vidgen: no clip in %q carries a caption", root)
	}
	return clips, nil
}

// DecodeClipSource decodes one clip into the planar [channel][frame]
// [height][width] source tensor in [0, 1] -- the exact convention the
// adaptive causal encoder consumes -- scaled to the requested shape and
// truncated to its frame count. A clip shorter than the requested
// frame count is refused rather than padded with invented frames.
func DecodeClipSource(ctx context.Context, ffmpeg, path string, shape SourceVideoShape) ([]float32, error) {
	if !checked.Equal(shape.Channels, 3) || !checked.PositiveInts(shape.Frames, shape.Height, shape.Width) {
		return nil, errors.New("vidgen: source shape requires 3 channels and positive extents")
	}
	spatial, err := checkedProduct(shape.Height, shape.Width)
	if err != nil {
		return nil, err
	}
	elements, err := checkedProduct(3, shape.Frames, spatial)
	if err != nil {
		return nil, err
	}
	stdout, sink := io.Pipe()
	var stderr bytes.Buffer
	supervised, err := processcontrol.Start(ctx, processcontrol.Command{
		Path: ffmpeg,
		Args: []string{
			"-v", "error", "-i", path,
			"-vf", fmt.Sprintf("scale=%d:%d:flags=bilinear", shape.Width, shape.Height),
			"-frames:v", fmt.Sprintf("%d", shape.Frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
		},
		Stdout: sink,
		Stderr: &stderr,
	})
	if err != nil {
		_ = sink.Close()
		return nil, err
	}
	go func() {
		_, _ = supervised.Wait(ctx)
		_ = sink.Close()
	}()
	frameBytes := make([]byte, 3*spatial)
	pixels := make([]float32, elements)
	for frame := 0; frame < shape.Frames; frame++ {
		if _, err := io.ReadFull(stdout, frameBytes); err != nil {
			_ = supervised.Terminate()
			_, _ = supervised.Wait(ctx)
			return nil, fmt.Errorf("vidgen: decode %q frame %d/%d: %w: %s",
				filepath.Base(path), frame, shape.Frames, err, strings.TrimSpace(stderr.String()))
		}
		for pixel := range spatial {
			for channel := range 3 {
				pixels[(channel*shape.Frames+frame)*spatial+pixel] = float32(frameBytes[3*pixel+channel]) / 255
			}
		}
	}
	_, _ = io.Copy(io.Discard, stdout)
	receipt, err := supervised.Wait(ctx)
	if err != nil || receipt.ExitCode != 0 {
		return nil, fmt.Errorf("vidgen: decode %q exit %d: %v: %s", filepath.Base(path), receipt.ExitCode, err, strings.TrimSpace(stderr.String()))
	}
	return pixels, nil
}

// EncodeClipLatent runs one decoded clip through the checkpoint-derived
// causal encoder and the profile's channel-wise affine, yielding the
// normalized latent space the DiT trains in.
func EncodeClipLatent(checkpoint string, graph VAEEncoderPlan, plan SourceCodecPlan, source []float32) ([]float32, error) {
	meanLatent, err := EncodeSourceVideo(checkpoint, graph, plan, source)
	if err != nil {
		return nil, err
	}
	normalized := make([]float32, len(meanLatent))
	if err := NormalizeSourceLatent(normalized, meanLatent, plan); err != nil {
		return nil, err
	}
	return normalized, nil
}
