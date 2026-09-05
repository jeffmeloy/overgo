package latentvideo

import (
	"bytes"
	"context"
	_ "embed"
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

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
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

var (
	videoProfileContract = artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: "application/vnd.overgo.video-profile+json", Schema: "overgo/video-profile/v1",
	}
	videoProfileCodec = artifact.DocumentCodec[Profile]{
		Name: "video profile", Contract: videoProfileContract,
		Decode:       func(data []byte, value *Profile) error { return strictjson.DecodeBytes(data, value) },
		Encode:       func(value Profile) ([]byte, error) { return json.Marshal(value) },
		Canonicalize: func(value *Profile) error { return value.validate() },
		Identity:     func(value Profile) artifact.ID { return value.ID },
		SetIdentity:  func(value *Profile, id artifact.ID) { value.ID = id },
	}
	encodedGIFContract = artifact.DocumentContract{
		Kind: artifact.KindOutput, MediaType: media.GIFMediaType, Schema: "overgo.encoded-video.gif.v1",
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

// ResolveProfile validates the artifact against the supported Wan profile.
func ResolveProfile(modelDirectory string) (Profile, error) {
	var catalog []Profile
	if err := strictjson.DecodeBytes(videoProfileCatalogJSON, &catalog); err != nil {
		return Profile{}, fmt.Errorf("latent video: decode profile catalog: %w", err)
	}
	profileSource, ok := checked.First(catalog)
	if !ok {
		return Profile{}, errors.New("latent video: profile catalog is empty")
	}
	profile, err := videoProfileCodec.New(profileSource)
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

func ReadProfile(ctx context.Context, store artifact.Reader, id artifact.ID) (Profile, error) {
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
	for _, name := range []string{"config.json", "diffusion_pytorch_model.safetensors", "Wan2.1_VAE.pth"} {
		if info, err := os.Stat(filepath.Join(path, name)); err != nil || info.IsDir() {
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

// EncodedVideo owns a deterministic GIF artifact.
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

func GIFContent(video EncodedVideo) (artifact.Content, error) {
	if err := media.ValidateEncodedRGBVideo(video.Data, video.MediaType, media.GIFMediaType, video.Frames, video.Channels, video.Height, video.Width, video.FPS); err != nil {
		return artifact.Content{}, fmt.Errorf("latent video: invalid encoded GIF: %w", err)
	}
	return encodedGIFContract.OwnedContentBytes(video.Data)
}

// GIFEncoder streams borrowed planar frames into one encoded artifact.
type GIFEncoder struct {
	fps       int
	pixels    PixelRange
	animation gif.GIF
	prior     []uint8
	changed   int
	height    int
	width     int
}

func NewGIFEncoder(fps int, pixels PixelRange) (*GIFEncoder, error) {
	if !checked.PositiveInts(fps) || !pixels.valid() {
		return nil, errors.New("latent video: invalid GIF encoding policy")
	}
	return &GIFEncoder{fps: fps, pixels: pixels}, nil
}

func (s *GIFEncoder) Add(_ int, frame []float32, height, width int) error {
	raw := make([]uint8, len(frame))
	if err := media.EncodePlanarRGB8Into(raw, frame, height, width, func(value float32) uint8 {
		return encodeVideoByte(value, s.pixels)
	}); err != nil {
		return err
	}
	if len(s.animation.Image) > 0 {
		if !checked.Equal(height, s.height) {
			return errors.New("latent video: frame geometry changed")
		}
		if !checked.Equal(width, s.width) {
			return errors.New("latent video: frame geometry changed")
		}
	}
	s.height, s.width = height, width
	rgba := image.NewRGBA(image.Rectangle{Max: image.Pt(width, height)})
	plane := height * width
	for index := range plane {
		base := index * media.RGBChannels
		if s.prior != nil && !bytes.Equal(raw[base:base+media.RGBChannels], s.prior[base:base+media.RGBChannels]) {
			s.changed++
		}
		rgba.SetRGBA(index%width, index/width, color.RGBA{
			R: raw[base+tensor.FirstOffset],
			G: raw[base+tensor.SingletonExtent],
			B: raw[base+tensor.PairedExtent],
			A: ^uint8(tensor.FirstOffset),
		})
	}
	s.prior = raw
	paletted := image.NewPaletted(rgba.Bounds(), palette.Plan9)
	draw.FloydSteinberg.Draw(paletted, rgba.Bounds(), rgba, image.Point{})
	s.animation.Image = append(s.animation.Image, paletted)
	s.animation.Delay = append(s.animation.Delay, media.GIFFrameDelay(s.fps))
	return nil
}

func (p PixelRange) valid() bool { return p == SignedUnitPixels || p == UnitPixels }

func encodeVideoByte(value float32, pixels PixelRange) byte {
	if pixels == SignedUnitPixels {
		return media.SignedUnitF32ToU8(value)
	}
	return media.UnitF32ToU8(value)
}

func (s *GIFEncoder) Finish() (EncodedVideo, error) {
	if !checked.NonemptyAll(s.animation.Image) {
		return EncodedVideo{}, errors.New("latent video: no frames emitted")
	}
	var encoded bytes.Buffer
	if err := gif.EncodeAll(&encoded, &s.animation); err != nil {
		return EncodedVideo{}, err
	}
	return EncodedVideo{
		Data: encoded.Bytes(), MediaType: media.GIFMediaType, Frames: len(s.animation.Image), Channels: media.RGBChannels,
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
	CondContext    []float32                 `json:"cond_context,omitempty"`
	UncondContext  []float32                 `json:"uncond_context,omitempty"`
	InitialSample  []float32                 `json:"initial_sample,omitempty"`
	Noise          sampling.CounterNoisePlan `json:"noise,omitzero"`
}

func ValidateWanRequest(request WanRequest) error {
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
}

type SourceVideo struct {
	Pixels   []float32 `json:"pixels"`
	Channels int       `json:"channels"`
	Frames   int       `json:"frames"`
	Height   int       `json:"height"`
	Width    int       `json:"width"`
}

type ReferenceEditRequest struct {
	Condition ReferenceEditCondition `json:"condition"`
	Source    SourceVideo            `json:"source"`
}

func ValidateReferenceEditRequest(request ReferenceEditRequest) error {
	c, s := request.Condition, request.Source
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

// SessionKey identifies reusable resident state for the request.
func (request ReferenceEditRequest) SessionKey() (string, error) {
	content, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.liveedit-session-policy.v1"), struct {
		Context  []float32 `json:"context"`
		Shape    [4]int    `json:"shape"`
		Chunk    [2]int    `json:"chunk"`
		Steps    []int64   `json:"steps"`
		Sigmas   []float32 `json:"sigmas"`
		Timestep int64     `json:"context_timestep"`
		Seed     int64     `json:"seed"`
	}{
		Context: request.Condition.TextContext,
		Shape:   [4]int{request.Source.Channels, request.Source.Frames, request.Source.Height, request.Source.Width},
		Chunk:   [2]int{request.Condition.FramesPerChunk, request.Condition.LocalAttention}, Steps: request.Condition.Timesteps,
		Sigmas: request.Condition.Sigmas, Timestep: request.Condition.ContextTimestep, Seed: request.Condition.Seed,
	})
	if err != nil {
		return "", err
	}
	return "liveedit:" + content.Descriptor.ID.String(), nil
}

func ReferenceEditInputContent(request ReferenceEditRequest) (map[recipe.PortName]artifact.Content, error) {
	condition, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.liveedit-conditioning.v1"), request.Condition)
	if err != nil {
		return nil, err
	}
	source, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.source-video.v1"), request.Source)
	if err != nil {
		return nil, err
	}
	return map[recipe.PortName]artifact.Content{"condition": condition, "source": source}, nil
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
		func(features wanFeatures) (EncodedVideo, error) { return features.video, nil }, GIFContent)
}

func registerReferenceEditRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, generate func(context.Context, ReferenceEditRequest) (EncodedVideo, error)) error {
	if err := workflowruntime.RegisterResolvedStage(runtime, modelrecipe.ModuleReferenceVideoPrepare, modelID,
		func(_ context.Context, step workflowruntime.StepRequest) (referenceEditPlan, error) {
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
		func(features referenceEditFeatures) (EncodedVideo, error) { return features.video, nil }, GIFContent)
}
