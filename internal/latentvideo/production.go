package latentvideo

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/gif"
	"math"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

const (
	videoProfileVersion uint16 = 1
	encodedGIFMediaType        = "image/gif"
)

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
		Kind: artifact.KindOutput, MediaType: encodedGIFMediaType, Schema: "overgo.encoded-video.gif.v1",
	}
)

// Profile owns Wan execution facts absent from checkpoint metadata.
type Profile struct {
	ID          artifact.ID    `json:"-"`
	Version     uint16         `json:"version"`
	Policy      DenoiserPolicy `json:"denoiser"`
	LatentStats VAELatentStats `json:"vae_latent_stats"`
	SampleFPS   int            `json:"sample_fps"`
	Precision   string         `json:"precision"`
}

// ResolveProfile validates the artifact against the supported Wan profile.
func ResolveProfile(modelDirectory string) (Profile, error) {
	profile, err := videoProfileCodec.New(Profile{
		Version: videoProfileVersion,
		Policy: DenoiserPolicy{
			NumTrainTimesteps: 1000, SinusoidalPeriod: 10000,
			RotaryFrequencyBase: 10000, VAEStride: [3]int{4, 8, 8},
		},
		LatentStats: VAELatentStats{
			Mean: []float32{-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921},
			Std:  []float32{2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916},
		},
		SampleFPS: 16, Precision: "bf16",
	})
	if err != nil {
		return Profile{}, err
	}
	config, err := LoadDenoiserConfig(modelDirectory, profile.Policy)
	if err != nil {
		return Profile{}, err
	}
	if config.InDim != len(profile.LatentStats.Mean) || config.OutDim != len(profile.LatentStats.Std) {
		return Profile{}, errors.New("latent video: profile latent channels differ from artifact")
	}
	return profile, nil
}

func ReadProfile(ctx context.Context, store artifact.Reader, id artifact.ID) (Profile, error) {
	return videoProfileCodec.Require(ctx, store, id)
}

func (p Profile) Content() (artifact.Content, error) { return videoProfileCodec.Content(p) }

func (p Profile) validate() error {
	if p.Version != videoProfileVersion || p.SampleFPS <= 0 || p.Precision != "bf16" ||
		p.Policy.NumTrainTimesteps <= 0 || p.Policy.SinusoidalPeriod <= 0 || p.Policy.RotaryFrequencyBase <= 0 {
		return errors.New("latent video: incomplete profile")
	}
	if len(p.LatentStats.Mean) == 0 || len(p.LatentStats.Mean) != len(p.LatentStats.Std) {
		return errors.New("latent video: invalid latent statistics")
	}
	for _, value := range p.LatentStats.Std {
		if value <= 0 || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return errors.New("latent video: invalid latent standard deviation")
		}
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
	if video.MediaType != encodedGIFMediaType || video.Frames <= 0 || video.Channels != 3 ||
		video.Height <= 0 || video.Width <= 0 || video.FPS <= 0 || len(video.Data) == 0 {
		return artifact.Content{}, errors.New("latent video: invalid encoded GIF")
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
	if fps <= 0 || !pixels.valid() {
		return nil, errors.New("latent video: invalid GIF encoding policy")
	}
	return &GIFEncoder{fps: fps, pixels: pixels}, nil
}

func (s *GIFEncoder) Add(_ int, frame []float32, height, width int) error {
	if height <= 0 || width <= 0 || len(frame) != 3*height*width {
		return errors.New("latent video: invalid planar RGB frame")
	}
	if len(s.animation.Image) > 0 && (height != s.height || width != s.width) {
		return errors.New("latent video: frame geometry changed")
	}
	s.height, s.width = height, width
	rgba := image.NewRGBA(image.Rect(0, 0, width, height))
	raw := make([]uint8, 3*height*width)
	plane := height * width
	for index := range plane {
		for channel := range 3 {
			value := frame[channel*plane+index]
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return errors.New("latent video: non-finite frame")
			}
			raw[index*3+channel] = encodeVideoByte(value, s.pixels)
		}
		if s.prior != nil && (raw[index*3] != s.prior[index*3] || raw[index*3+1] != s.prior[index*3+1] || raw[index*3+2] != s.prior[index*3+2]) {
			s.changed++
		}
		rgba.SetRGBA(index%width, index/width, color.RGBA{R: raw[index*3], G: raw[index*3+1], B: raw[index*3+2], A: 255})
	}
	s.prior = raw
	paletted := image.NewPaletted(rgba.Bounds(), palette.Plan9)
	draw.FloydSteinberg.Draw(paletted, rgba.Bounds(), rgba, image.Point{})
	s.animation.Image = append(s.animation.Image, paletted)
	s.animation.Delay = append(s.animation.Delay, max(1, 100/s.fps))
	return nil
}

func (p PixelRange) valid() bool { return p == SignedUnitPixels || p == UnitPixels }

func encodeVideoByte(value float32, pixels PixelRange) byte {
	scaled := float64(value) * 255
	if pixels == SignedUnitPixels {
		scaled = (float64(value) + 1) * 127.5
	}
	return byte(min(max(math.Round(scaled), 0), 255))
}

func (s *GIFEncoder) Finish() (EncodedVideo, error) {
	if len(s.animation.Image) == 0 {
		return EncodedVideo{}, errors.New("latent video: no frames emitted")
	}
	var encoded bytes.Buffer
	if err := gif.EncodeAll(&encoded, &s.animation); err != nil {
		return EncodedVideo{}, err
	}
	return EncodedVideo{
		Data: encoded.Bytes(), MediaType: encodedGIFMediaType, Frames: len(s.animation.Image), Channels: 3,
		Height: s.height, Width: s.width, FPS: s.fps, ChangedPixels: s.changed,
	}, nil
}

// WanRequest carries compiled text conditioning and sampling state.
type WanRequest struct {
	Frames        int       `json:"frames"`
	Width         int       `json:"width"`
	Height        int       `json:"height"`
	Steps         int       `json:"steps"`
	Shift         float64   `json:"shift"`
	GuideScale    float64   `json:"guide_scale"`
	CondContext   []float32 `json:"cond_context"`
	UncondContext []float32 `json:"uncond_context"`
	InitialSample []float32 `json:"initial_sample,omitempty"`
	Noise         NoisePlan `json:"noise"`
}

func ValidateWanRequest(request WanRequest) error {
	if request.Frames <= 0 || request.Width <= 0 || request.Height <= 0 || request.Steps <= 0 ||
		request.Shift <= 0 || request.GuideScale < 0 || len(request.CondContext) == 0 || len(request.UncondContext) == 0 {
		return errors.New("latent video: incomplete Wan request")
	}
	return nil
}

func WanSessionPolicy(request WanRequest) (string, error) {
	if err := ValidateWanRequest(request); err != nil {
		return "", err
	}
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
	if len(c.TextContext) == 0 || c.FramesPerChunk <= 0 || c.LocalAttention <= 0 ||
		len(c.Timesteps) == 0 || len(c.Timesteps) != len(c.Sigmas) || s.Channels != 3 || s.Frames <= 0 || s.Height <= 0 || s.Width <= 0 ||
		len(s.Pixels) != s.Channels*s.Frames*s.Height*s.Width {
		return errors.New("latent video: incomplete LiveEdit request")
	}
	return nil
}

func ReferenceEditSessionPolicy(request ReferenceEditRequest) (string, error) {
	if err := ValidateReferenceEditRequest(request); err != nil {
		return "", err
	}
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
			return wanPlan{request: request}, ValidateWanRequest(request)
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
