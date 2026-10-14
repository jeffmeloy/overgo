package latentvideo

import (
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/gif"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/cuda/driver"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/processmeasure"
	"overgo/internal/sampling"
	"overgo/internal/strictjson"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/workflowruntime"
)

const (
	videoProfileVersion uint16 = 1
)

//go:embed profiles.json
var videoProfileCatalogJSON []byte

// PixelRange names the decoder output interval.
type PixelRange string

const (
	// SignedUnitPixels maps [-1,1] to bytes.
	SignedUnitPixels PixelRange = "signed_unit"
	// UnitPixels maps [0,1] to bytes.
	UnitPixels PixelRange = "unit"
)

// gifPaletteColors is how many colours a served clip's frame is drawn in:
// all a GIF frame can hold.
const gifPaletteColors = 256

var (
	videoProfileContract = artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: "application/vnd.overgo.video-profile+json", Schema: "overgo/video-profile/v1",
	}
	videoProfileCodec = artifact.DocumentCodec[Profile]{
		Name: "video profile", Contract: videoProfileContract,
		Decode:       func(data []byte, value *Profile) error { return strictjson.DecodeBytes(data, value) },
		Encode:       func(value Profile) ([]byte, error) { return json.Marshal(value) },
		Canonicalize: func(value *Profile) error { return value.validate() },
		Clone:        artifact.DeepCopy[Profile](),
		Identity:     func(value Profile) artifact.ID { return value.ID },
		SetIdentity:  func(value *Profile, id artifact.ID) { value.ID = id },
	}
	encodedGIFContract = artifact.DocumentContract{
		Kind: artifact.KindOutput, MediaType: media.GIFMediaType, Schema: "overgo.encoded-video.gif.v1",
	}
	encodedMP4Contract = artifact.DocumentContract{
		Kind: artifact.KindOutput, MediaType: media.MP4MediaType, Schema: "overgo.encoded-video.mp4.v1",
	}
)

// Profile owns Wan execution facts absent from checkpoint metadata.
type Profile struct {
	ID          artifact.ID      `json:"-"`
	Version     uint16           `json:"version"`
	Policy      DenoiserPolicy   `json:"denoiser"`
	LatentStats VAELatentStats   `json:"vae_latent_stats"`
	SampleFPS   int              `json:"sample_fps"`
	Precision   precisionPolicy  `json:"precision"`
	Generation  generationPolicy `json:"generation"`
}

type precisionPolicy struct {
	MatmulWeights         string `json:"matmul_weights"`
	RoundAttentionStorage bool   `json:"round_attention_storage"`
}

type generationPolicy struct {
	Frames     int     `json:"frames"`
	Width      int     `json:"width"`
	Height     int     `json:"height"`
	Steps      int     `json:"steps"`
	Shift      float64 `json:"shift"`
	GuideScale float64 `json:"guide_scale"`
	Seed       uint64  `json:"seed"`
}

// declaredProfile is the supported Wan profile as the catalog declares it,
// before any artifact is read against it.
func declaredProfile() (Profile, error) {
	var catalog []Profile
	if err := strictjson.DecodeBytes(videoProfileCatalogJSON, &catalog); err != nil {
		return Profile{}, fmt.Errorf("latent video: decode profile catalog: %w", err)
	}
	profileSource, ok := checked.First(catalog)
	if !ok {
		return Profile{}, errors.New("latent video: profile catalog is empty")
	}
	return videoProfileCodec.New(profileSource)
}

// ResolveProfile validates the artifact against the supported Wan profile.
func ResolveProfile(modelDirectory string) (Profile, error) {
	profile, err := declaredProfile()
	if err != nil {
		return Profile{}, err
	}
	config, err := LoadDenoiserConfig(modelDirectory, profile.Policy)
	if err != nil {
		return Profile{}, err
	}
	if !checked.Equal(config.InDim, len(profile.LatentStats.Mean)) {
		return Profile{}, errors.New("latent video: profile latent channels differ from artifact")
	}
	if !checked.Equal(config.OutDim, len(profile.LatentStats.Std)) {
		return Profile{}, errors.New("latent video: profile latent channels differ from artifact")
	}
	return profile, nil
}

func readProfile(ctx context.Context, store artifact.Reader, id artifact.ID) (Profile, error) {
	return videoProfileCodec.Require(ctx, store, id)
}

func (p Profile) Content() (artifact.Content, error) { return videoProfileCodec.Content(p) }

func (p Profile) validate() error {
	if !checked.Equal(p.Version, videoProfileVersion) ||
		!checked.PositiveInts(
			p.SampleFPS, p.Policy.NumTrainTimesteps, p.Policy.SinusoidalPeriod,
			p.Generation.Frames, p.Generation.Width, p.Generation.Height, p.Generation.Steps,
		) || !checked.Equal(p.Precision.MatmulWeights, dtype.BF16.String()) ||
		!checked.PositiveFinite64(p.Policy.RotaryFrequencyBase) ||
		!checked.PositiveFinite64(p.Generation.Shift) ||
		!checked.PositiveFinite64(p.Generation.GuideScale) {
		return errors.New("latent video: incomplete profile")
	}
	if err := media.ValidateChannelMoments(p.LatentStats.Mean, p.LatentStats.Std, len(p.LatentStats.Mean)); err != nil {
		return fmt.Errorf("latent video: invalid latent statistics: %w", err)
	}
	return nil
}

// IsWan recognizes the production checkpoint boundary without loading weights.
func IsWan(path string) bool {
	for _, file := range []string{filepath.Join(path, "config.json"), filepath.Join(path, "diffusion_pytorch_model.safetensors"), AutoencoderWeights(path)} {
		if info, err := os.Stat(file); err != nil || info.IsDir() {
			return false
		}
	}
	return true
}

// IsLiveEdit recognizes a LiveEdit checkpoint directory.
func IsLiveEdit(path string) bool {
	info, err := os.Stat(filepath.Join(path, "ar-forcing_002000.pt"))
	return err == nil && !info.IsDir()
}

// EncodedVideo owns a deterministic clip artifact, a GIF or an MP4.
type EncodedVideo struct {
	Data          []byte `json:"data"`
	MediaType     string `json:"media_type"`
	Frames        int    `json:"frames"`
	Channels      int    `json:"channels"`
	Height        int    `json:"height"`
	Width         int    `json:"width"`
	FPS           int    `json:"fps"`
	ChangedPixels int    `json:"changed_pixels"`
	ResidentRun   int    `json:"resident_run"`
}

// VideoContent is the artifact a clip is published as, under the contract of
// its media type.
//   - a GIF or an MP4 of positive frames, size and rate, three colours a pixel
//   - an MP4 opens with its file type box, so bytes of another form labelled as one are refused.
func VideoContent(video EncodedVideo) (artifact.Content, error) {
	contract := encodedGIFContract
	if video.MediaType == media.MP4MediaType {
		contract = encodedMP4Contract
		if opening := video.Data[:min(len(video.Data), mp4FileTypeEnd)]; len(opening) < mp4FileTypeEnd || string(opening[mp4FileTypeEnd-len(mp4FileType):]) != mp4FileType {
			return artifact.Content{}, fmt.Errorf("latent video: a clip labelled MP4 opens with %q, where an MP4 opens with its %s box", opening, mp4FileType)
		}
	}
	if err := media.ValidateEncodedRGBVideo(video.Data, video.MediaType, contract.MediaType, video.Frames, video.Channels, video.Height, video.Width, video.FPS); err != nil {
		return artifact.Content{}, fmt.Errorf("latent video: invalid encoded clip of %q (a GIF or an MP4 is published): %w", video.MediaType, err)
	}
	return contract.OwnedContentBytes(video.Data)
}

// The box an MP4 opens with (ISO/IEC 14496-12 section 4.3): its type follows
// the box's 32-bit size, so it ends at byte 8. The format fixes both, so
// neither is reconsidered.
const (
	mp4FileType    = "ftyp"
	mp4FileTypeEnd = 8
)

// ClipFormat is the form a served clip is written in.
//   - "" or "gif" = a GIF, each frame cut to a palette of its own, the default
//   - "mp4" = H.264 in an MP4 through FFmpeg, at full colour where a GIF frame holds 256, and in fewer bytes
//   - "gif" marshals as "", so a request that names the default is the request that names none, and keeps its run key.
type ClipFormat string

const (
	gifClip ClipFormat = "gif"
	mp4Clip ClipFormat = "mp4"
)

// IsZero reports whether the format is the default GIF.
func (format ClipFormat) IsZero() bool { return format == "" || format == gifClip }

// Choices are the formats a page offers.
func (ClipFormat) Choices() []string { return []string{string(gifClip), string(mp4Clip)} }

func (format ClipFormat) validate() error {
	if !format.IsZero() && format != mp4Clip {
		return fmt.Errorf("latent video: clip format %q, want %q or %q", string(format), gifClip, mp4Clip)
	}
	return nil
}

// clipEncoder writes a served clip from the frames a generation hands it.
//   - Close releases what an unfinished clip holds and waits for its work, and is a no-op after Finish
type clipEncoder interface {
	add(ctx context.Context, index int, frame []float32, height, width int) error
	Finish() (EncodedVideo, error)
	Close() error
}

// newServedEncoder is the encoder a served clip of the format is written by.
//   - an MP4 is written by the FFmpeg media.FFmpeg resolves, and is refused where none resolves
func newServedEncoder(ctx context.Context, format ClipFormat, fps int, pixels PixelRange) (clipEncoder, error) {
	if err := format.validate(); err != nil {
		return nil, err
	}
	if format.IsZero() {
		return NewClipEncoder(fps, pixels)
	}
	ffmpeg, err := media.FFmpeg()
	if err != nil {
		return nil, fmt.Errorf("latent video: an MP4 clip is written through FFmpeg: %w", err)
	}
	return NewMP4Encoder(ctx, ffmpeg, fps, pixels)
}

// writeClip runs a generation that hands its frames to the encoder and
// returns the clip, or what the generation failed with and no clip.
//   - each frame taken and the clip's writing are timed as parts of the request the context carries a ledger for
//   - it returns only when the encoder holds no work, so a generation that fails or is cancelled leaves no worker or process behind it
func writeClip(ctx context.Context, encoder clipEncoder, generate func(VideoFrameSink) error) (video EncodedVideo, err error) {
	defer func() { err = errors.Join(err, encoder.Close()) }()
	if err := generate(func(index int, frame []float32, height, width int) error {
		defer processmeasure.Stage(ctx, "clip frame taken")()
		return encoder.add(ctx, index, frame, height, width)
	}); err != nil {
		return EncodedVideo{}, err
	}
	defer processmeasure.Stage(ctx, "clip written")()
	return encoder.Finish()
}

// gifEncoder streams borrowed planar frames into one encoded artifact.
type gifEncoder struct {
	fps    int
	pixels PixelRange
	// cut draws each frame in a palette cut from it; unset, every frame is
	// drawn in one fixed palette.
	cut bool
	// frames are the clip's frames in order and delays the time each is
	// shown. A frame is drawn by a worker of its own from the bytes Add took
	// of it: its palette and its dither read its own pixels and no other
	// frame's, so the frames are drawn as they arrive, beside the decoding of
	// the ones after them, where one worker drawing 21 of them in turn took
	// three seconds of a request. drawing is the workers still at it.
	frames  []*drawnFrame
	delays  []int
	drawing sync.WaitGroup
	// draw draws one frame, and slots bounds the frames being drawn at once,
	// each of which holds the bytes Add took of it: a frame waits in Add for
	// a worker to be free, which holds the decoding to the drawing's pace and
	// the copies kept to the workers'.
	draw    func(raw []uint8, height, width int, cut bool) *image.Paletted
	slots   chan struct{}
	prior   []uint8
	changed int
	height  int
	width   int
}

// drawnFrame is one frame of a clip, set by the worker that draws it.
type drawnFrame struct {
	picture *image.Paletted
}

// drawFrame draws a frame's bytes, three a pixel, in a palette: one cut from
// the frame, or the fixed one.
func drawFrame(raw []uint8, height, width int, cut bool) *image.Paletted {
	rgba := image.NewRGBA(image.Rectangle{Max: image.Pt(width, height)})
	for index := range height * width {
		base := index * media.RGBChannels
		rgba.SetRGBA(index%width, index/width, color.RGBA{
			R: raw[base+tensor.FirstOffset],
			G: raw[base+tensor.SingletonExtent],
			B: raw[base+tensor.PairedExtent],
			A: ^uint8(tensor.FirstOffset),
		})
	}
	colours := palette.Plan9
	if cut {
		colours = media.AdaptivePalette(rgba, gifPaletteColors)
	}
	paletted := image.NewPaletted(rgba.Bounds(), colours)
	draw.FloydSteinberg.Draw(paletted, rgba.Bounds(), rgba, image.Point{})
	return paletted
}

// NewGIFEncoder writes a clip whose every frame is drawn in one fixed
// palette. It is the encoder the retained clips were written with, and the
// one that rebuilds them byte for byte from their retained frames; a clip
// that is served is written by NewClipEncoder.
func NewGIFEncoder(fps int, pixels PixelRange) (*gifEncoder, error) {
	if !checked.PositiveInts(fps) || !pixels.valid() {
		return nil, errors.New("latent video: invalid GIF encoding policy")
	}
	encoder := &gifEncoder{fps: fps, pixels: pixels, draw: drawFrame, slots: make(chan struct{}, runtime.GOMAXPROCS(0))}
	if _, err := media.GIFFrameDelay(fps, len(encoder.frames)); err != nil {
		return nil, err
	}
	return encoder, nil
}

// NewClipEncoder writes a clip whose every frame is drawn in a palette cut
// from that frame. A fixed palette's dither is noise a fifth of the range
// deep: a clip comes back from it at 22 dB, and a clip handed to an editor
// carries that grain into the edit. Drawn in its own colours it comes back
// at 33 dB.
func NewClipEncoder(fps int, pixels PixelRange) (*gifEncoder, error) {
	encoder, err := NewGIFEncoder(fps, pixels)
	if err != nil {
		return nil, err
	}
	encoder.cut = true
	return encoder, nil
}

// Add takes the next frame of the clip. It keeps the frame's bytes, counts
// the pixels that changed since the frame before, and hands the drawing to a
// worker; the frame it was given is the caller's again when it returns.
func (s *gifEncoder) Add(index int, frame []float32, height, width int) error {
	return s.add(context.Background(), index, frame, height, width)
}

// add is Add for a request: a frame that waits for a worker stops waiting
// when the request is cancelled, and is not taken.
func (s *gifEncoder) add(ctx context.Context, index int, frame []float32, height, width int) error {
	if index != len(s.frames) {
		return errors.New("latent video: frame index is out of order")
	}
	delay, err := media.GIFFrameDelay(s.fps, index)
	if err != nil {
		return err
	}
	if len(s.frames) > 0 && (!checked.Equal(height, s.height) || !checked.Equal(width, s.width)) {
		return errors.New("latent video: frame geometry changed")
	}
	select {
	case s.slots <- struct{}{}:
	case <-ctx.Done():
		return context.Cause(ctx)
	}
	raw := make([]uint8, len(frame))
	if err := media.EncodePlanarRGB8Into(raw, frame, height, width, func(value float32) uint8 {
		return encodeVideoByte(value, s.pixels)
	}); err != nil {
		<-s.slots
		return err
	}
	s.height, s.width = height, width
	if s.prior != nil {
		for base := 0; base < len(raw); base += media.RGBChannels {
			if !bytes.Equal(raw[base:base+media.RGBChannels], s.prior[base:base+media.RGBChannels]) {
				s.changed++
			}
		}
	}
	s.prior = raw
	drawn := &drawnFrame{}
	s.frames, s.delays = append(s.frames, drawn), append(s.delays, delay)
	cut := s.cut
	s.drawing.Go(func() {
		defer func() { <-s.slots }()
		drawn.picture = s.draw(raw, height, width, cut)
	})
	return nil
}

// Close waits for the frames still being drawn, so no worker outlives the
// clip.
func (s *gifEncoder) Close() error {
	s.drawing.Wait()
	return nil
}

func (p PixelRange) valid() bool { return p == SignedUnitPixels || p == UnitPixels }

func encodeVideoByte(value float32, pixels PixelRange) byte {
	if pixels == SignedUnitPixels {
		return media.SignedUnitF32ToU8(value)
	}
	return media.UnitF32ToU8(value)
}

// Finish waits for the frames still being drawn and writes the clip.
func (s *gifEncoder) Finish() (EncodedVideo, error) {
	s.drawing.Wait()
	if !checked.NonemptyAll(s.frames) {
		return EncodedVideo{}, errors.New("latent video: no frames emitted")
	}
	animation := gif.GIF{Delay: s.delays}
	for _, frame := range s.frames {
		animation.Image = append(animation.Image, frame.picture)
	}
	var encoded bytes.Buffer
	if err := gif.EncodeAll(&encoded, &animation); err != nil {
		return EncodedVideo{}, err
	}
	return EncodedVideo{
		Data: encoded.Bytes(), MediaType: media.GIFMediaType, Frames: len(s.frames), Channels: media.RGBChannels,
		Height: s.height, Width: s.width, FPS: s.fps, ChangedPixels: s.changed,
	}, nil
}

// WanRequest carries text conditioning and sampling state in one of two
// forms. The tensor form names compiled contexts and a noise plan; the
// prompt form names a prompt (and optionally a negative prompt and a seed)
// and leaves any generation parameter at zero for the profile's generation
// policy, so a page composes it from typed controls and the runtime
// resolves the rest through the model's own text pipeline.
type WanRequest struct {
	Frames         int                       `json:"frames"`
	Width          int                       `json:"width"`
	Height         int                       `json:"height"`
	Steps          int                       `json:"steps"`
	Shift          float64                   `json:"shift"`
	GuideScale     float64                   `json:"guide_scale"`
	Prompt         string                    `json:"prompt,omitzero"`
	NegativePrompt string                    `json:"negative_prompt,omitzero"`
	Seed           uint64                    `json:"seed,omitzero"`
	Format         ClipFormat                `json:"format,omitzero"`
	CondContext    []float32                 `json:"cond_context,omitempty"`
	UncondContext  []float32                 `json:"uncond_context,omitempty"`
	InitialSample  []float32                 `json:"initial_sample,omitempty"`
	Noise          sampling.CounterNoisePlan `json:"noise,omitzero"`
}

func validateWanRequest(request WanRequest) error {
	if !checked.PositiveInts(request.Frames, request.Width, request.Height, request.Steps) {
		return errors.New("latent video: incomplete Wan request")
	}
	if !checked.PositiveFinite64(request.Shift) || !checked.NonNegativeFinite64(request.GuideScale) {
		return errors.New("latent video: incomplete Wan request")
	}
	if _, ok := checked.First(request.CondContext); !ok {
		return errors.New("latent video: incomplete Wan request")
	}
	if _, ok := checked.First(request.UncondContext); !ok {
		return errors.New("latent video: incomplete Wan request")
	}
	return nil
}

// SessionKey identifies reusable resident state for the request; a
// prompt-form request that leaves its geometry to the profile keys the
// profile's default geometry under the zero extents it names.
func (request WanRequest) SessionKey() (string, error) {
	return fmt.Sprintf("wan:%dx%dx%d", request.Frames, request.Height, request.Width), nil
}

type ReferenceEditCondition struct {
	TextContext     []float32 `json:"text_context"`
	InitialNoise    []float32 `json:"initial_noise"`
	FramesPerChunk  int       `json:"frames_per_chunk"`
	LocalAttention  int       `json:"local_attention_frames"`
	Timesteps       []int64   `json:"timesteps"`
	Sigmas          []float32 `json:"sigmas"`
	ContextTimestep int64     `json:"context_timestep"`
	Seed            int64     `json:"seed"`
	// Format rides in the condition because the edit recipe reads only its condition and its source.
	Format ClipFormat `json:"format,omitzero"`
}

type SourceVideo struct {
	Pixels   []float32 `json:"pixels"`
	Channels int       `json:"channels"`
	Frames   int       `json:"frames"`
	Height   int       `json:"height"`
	Width    int       `json:"width"`
}

// ReferenceEditRequest carries the reference edit in one of two forms: the
// compiled condition with decoded source pixels, or the clip form a page
// composes (a prompt, a seed and the source clip's artifact id), which the
// video capability resolves through the declared edit policy and the base
// model's text pipeline before the runtime sees it.
type ReferenceEditRequest struct {
	Condition      ReferenceEditCondition `json:"condition,omitzero"`
	Source         SourceVideo            `json:"source,omitzero"`
	Prompt         string                 `json:"prompt,omitzero"`
	Seed           int64                  `json:"seed,omitzero"`
	SourceArtifact artifact.ID            `json:"source_artifact,omitzero" label:"source clip" media:"video"`
	// Format is the clip form's format, which resolving moves into the condition.
	Format ClipFormat `json:"format,omitzero"`
}

func ValidateReferenceEditRequest(request ReferenceEditRequest) error {
	c, s := request.Condition, request.Source
	if !request.Format.IsZero() {
		return fmt.Errorf("latent video: a resolved LiveEdit request names format %q beside its condition, which is where it carries one", request.Format)
	}
	if err := c.Format.validate(); err != nil {
		return err
	}
	if _, ok := checked.First(c.TextContext); !ok {
		return errors.New("latent video: incomplete LiveEdit request")
	}
	if !checked.PositiveInts(c.FramesPerChunk, c.LocalAttention) {
		return errors.New("latent video: incomplete LiveEdit request")
	}
	if _, ok := checked.First(c.Timesteps); !ok {
		return errors.New("latent video: incomplete LiveEdit request")
	}
	if !checked.Equal(len(c.Timesteps), len(c.Sigmas)) {
		return errors.New("latent video: incomplete LiveEdit request")
	}
	if !checked.Equal(s.Channels, media.RGBChannels) {
		return errors.New("latent video: incomplete LiveEdit request")
	}
	if err := media.ValidatePlanarVideo(s.Pixels, s.Channels, s.Frames, s.Height, s.Width); err != nil {
		return errors.New("latent video: incomplete LiveEdit request")
	}
	return nil
}

// SessionKey identifies reusable resident state for the request: the clip's
// shape and the edit's schedule, which a resident runtime is compiled for. The
// instruction's context and the seed are not in it. A request gives them to a
// resident runtime (referenceEditRuntime.Run), so an edit with another
// instruction or another seed loads nothing.
func (request ReferenceEditRequest) SessionKey() (string, error) {
	content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.liveedit-session-policy.v2"), struct {
		Shape    [4]int    `json:"shape"`
		Chunk    [2]int    `json:"chunk"`
		Steps    []int64   `json:"steps"`
		Sigmas   []float32 `json:"sigmas"`
		Timestep int64     `json:"context_timestep"`
	}{
		Shape: [4]int{request.Source.Channels, request.Source.Frames, request.Source.Height, request.Source.Width},
		Chunk: [2]int{request.Condition.FramesPerChunk, request.Condition.LocalAttention}, Steps: request.Condition.Timesteps,
		Sigmas: request.Condition.Sigmas, Timestep: request.Condition.ContextTimestep,
	})
	if err != nil {
		return "", err
	}
	return "liveedit:" + content.Descriptor.ID.String(), nil
}

// editConditionRecord and sourceVideoRecord are what a run records of an
// edit's condition and of its source clip in place of their tensors: the
// settings whole, and each tensor as its length and the digest of its values'
// bytes.
type editConditionRecord struct {
	TextContext     tensorRecord `json:"text_context"`
	InitialNoise    tensorRecord `json:"initial_noise"`
	FramesPerChunk  int          `json:"frames_per_chunk"`
	LocalAttention  int          `json:"local_attention_frames"`
	Timesteps       []int64      `json:"timesteps"`
	Sigmas          []float32    `json:"sigmas"`
	ContextTimestep int64        `json:"context_timestep"`
	Seed            int64        `json:"seed"`
	Format          ClipFormat   `json:"format,omitzero"`
}

type sourceVideoRecord struct {
	Pixels   tensorRecord `json:"pixels"`
	Channels int          `json:"channels"`
	Frames   int          `json:"frames"`
	Height   int          `json:"height"`
	Width    int          `json:"width"`
}

type tensorRecord struct {
	Elements int    `json:"elements"`
	SHA256   string `json:"sha256"`
}

func recordTensor(values []float32) tensorRecord {
	digest := sha256.Sum256(driver.Bytes(values))
	return tensorRecord{Elements: len(values), SHA256: hex.EncodeToString(digest[:])}
}

// Record names a condition exactly without holding its tensors. A clip's
// pixels at the size an editor is trained at are 25 million values, and a
// document of them took longer to write and read back than the edit took to
// denoise; a digest of their bytes names them as exactly in a tenth of a
// second.
func (c ReferenceEditCondition) Record() editConditionRecord {
	return editConditionRecord{
		TextContext: recordTensor(c.TextContext), InitialNoise: recordTensor(c.InitialNoise),
		FramesPerChunk: c.FramesPerChunk, LocalAttention: c.LocalAttention,
		Timesteps: c.Timesteps, Sigmas: c.Sigmas, ContextTimestep: c.ContextTimestep, Seed: c.Seed, Format: c.Format,
	}
}

// Record names a source clip exactly without holding its pixels.
func (s SourceVideo) Record() sourceVideoRecord {
	return sourceVideoRecord{Pixels: recordTensor(s.Pixels), Channels: s.Channels, Frames: s.Frames, Height: s.Height, Width: s.Width}
}

type wanPlan struct{ request WanRequest }
type wanFeatures struct{ video EncodedVideo }
type referenceEditPlan struct{ request ReferenceEditRequest }
type referenceEditFeatures struct{ video EncodedVideo }

func registerWanRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generate func(context.Context, WanRequest) (EncodedVideo, error)) error {
	if err := workflowruntime.RegisterScalarStage(runtime, modelrecipe.ModuleLatentVideoPrepare, modelID,
		func(request WanRequest) (wanPlan, error) {
			return wanPlan{request: request}, ValidateWanRequestForm(request)
		}, nil); err != nil {
		return err
	}
	if err := workflowruntime.RegisterContextStage(runtime, modelrecipe.ModuleLatentVideoIntegrate, modelID,
		func(ctx context.Context, plan wanPlan) (wanFeatures, error) {
			video, err := generate(ctx, plan.request)
			return wanFeatures{video: video}, err
		}, nil); err != nil {
		return err
	}
	return workflowruntime.RegisterScalarStage(runtime, modelrecipe.ModuleLatentVideoDecode, modelID,
		func(features wanFeatures) (EncodedVideo, error) { return features.video, nil }, VideoContent)
}

func registerReferenceEditRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generate func(context.Context, ReferenceEditRequest) (EncodedVideo, error)) error {
	if err := workflowruntime.RegisterResolvedStage(runtime, modelrecipe.ModuleReferenceVideoPrepare, modelID,
		func(ctx context.Context, step workflowruntime.StepRequest) (referenceEditPlan, error) {
			defer processmeasure.Stage(ctx, "request read by its stage")()
			condition, err := workflowruntime.ScalarInput[ReferenceEditCondition](step, "condition")
			if err != nil {
				return referenceEditPlan{}, err
			}
			source, err := workflowruntime.ScalarInput[SourceVideo](step, "source")
			request := ReferenceEditRequest{Condition: condition, Source: source}
			return referenceEditPlan{request: request}, errors.Join(err, ValidateReferenceEditRequest(request))
		}, nil); err != nil {
		return err
	}
	if err := workflowruntime.RegisterContextStage(runtime, modelrecipe.ModuleReferenceVideoIntegrate, modelID,
		func(ctx context.Context, plan referenceEditPlan) (referenceEditFeatures, error) {
			video, err := generate(ctx, plan.request)
			return referenceEditFeatures{video: video}, err
		}, nil); err != nil {
		return err
	}
	return workflowruntime.RegisterScalarStage(runtime, modelrecipe.ModuleReferenceVideoDecode, modelID,
		func(features referenceEditFeatures) (EncodedVideo, error) { return features.video, nil }, VideoContent)
}
