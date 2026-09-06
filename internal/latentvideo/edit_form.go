package latentvideo

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"image/gif"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/sampling"
	"overgo/internal/strictjson"
)

//go:embed edit_profiles.json
var editPolicyCatalogJSON []byte

// EditPolicy declares the reference-edit schedule a LiveEdit checkpoint
// runs under: the causal chunking, the denoising timesteps, and the flow
// schedule their sigmas compile from. The catalog restates the upstream
// few-step configuration (LiveEdit/configs/wan_mm-ar-forcing-local.yaml).
type EditPolicy struct {
	Checkpoint           string  `json:"checkpoint"`
	FramesPerChunk       int     `json:"frames_per_chunk"`
	LocalAttentionFrames int     `json:"local_attention_frames"`
	DenoisingTimesteps   []int64 `json:"denoising_timesteps"`
	InferenceSteps       int     `json:"num_inference_steps"`
	Shift                float32 `json:"shift"`
	SigmaMin             float32 `json:"sigma_min"`
	SigmaMax             float32 `json:"sigma_max"`
	ExtraStep            bool    `json:"extra_one_step"`
	ContextTimestep      int64   `json:"context_timestep"`
}

// The reference layout of a LiveEdit model directory: the edit checkpoint
// inside it and the Wan base model directory beside it.
const (
	liveEditCheckpoint = "ar-forcing_002000.pt"
	wanBaseDirectory   = "Wan2.1-T2V-1.3B"
)

// LiveEditArtifacts locates the two artifacts a LiveEdit model directory
// runs with: the Wan base model beside it and the edit checkpoint inside it.
func LiveEditArtifacts(modelDirectory string) (wanDirectory, checkpoint string) {
	return filepath.Join(filepath.Dir(modelDirectory), wanBaseDirectory), filepath.Join(modelDirectory, liveEditCheckpoint)
}

// ResolveEditPolicy reads the declared policy for the checkpoint a LiveEdit
// directory carries, refusing an absent checkpoint or an undeclared one.
func ResolveEditPolicy(modelDirectory string) (EditPolicy, error) {
	var catalog []EditPolicy
	if err := strictjson.DecodeBytes(editPolicyCatalogJSON, &catalog); err != nil {
		return EditPolicy{}, fmt.Errorf("latent video: decode edit policy catalog: %w", err)
	}
	_, checkpoint := LiveEditArtifacts(modelDirectory)
	for _, policy := range catalog {
		if policy.Checkpoint != filepath.Base(checkpoint) {
			continue
		}
		if _, err := os.Stat(checkpoint); err != nil {
			return EditPolicy{}, err
		}
		return policy, policy.validate()
	}
	return EditPolicy{}, fmt.Errorf("latent video: no edit policy declares checkpoint %s", filepath.Base(checkpoint))
}

func (policy EditPolicy) validate() error {
	if !checked.PositiveInts(policy.FramesPerChunk, policy.LocalAttentionFrames, policy.InferenceSteps) {
		return errors.New("latent video: incomplete edit policy")
	}
	if _, ok := checked.First(policy.DenoisingTimesteps); !ok || policy.ContextTimestep < 0 {
		return errors.New("latent video: incomplete edit policy")
	}
	return nil
}

// admitLatentFrames restates the edit denoiser's chunk arithmetic for a
// clip: its latent frames must fill whole chunks beyond the first and fit
// the local attention history, so a clip the policy cannot chunk is
// refused with the counts it admits.
func (policy EditPolicy) admitLatentFrames(latent int) error {
	chunk, window := policy.FramesPerChunk, policy.LocalAttentionFrames
	if _, aligned := checked.DivExactInt(latent, chunk); !aligned || !checked.GreaterInt(latent, chunk) || !checked.AtMostInt(latent, window+chunk) {
		return fmt.Errorf("latent video: the clip's %d latent frames do not fit the edit policy (multiples of %d above %d, at most %d)",
			latent, chunk, chunk, window+chunk)
	}
	return nil
}

// ClipForm reports whether the request names a prompt and a source clip
// artifact rather than a compiled condition and decoded pixels: the page
// form, which the video capability resolves before the runtime.
func (request ReferenceEditRequest) ClipForm() bool {
	return request.Prompt != "" && request.SourceArtifact.Valid() &&
		len(request.Condition.TextContext) == 0 && len(request.Source.Pixels) == 0
}

// ValidateReferenceEditRequestForm admits the complete request or the clip
// form: a prompt, a source clip artifact id, and a non-negative seed.
func ValidateReferenceEditRequestForm(request ReferenceEditRequest) error {
	if !request.ClipForm() {
		return ValidateReferenceEditRequest(request)
	}
	if request.Seed < 0 {
		return errors.New("latent video: negative LiveEdit seed")
	}
	return nil
}

// SourceVideoFromGIF decodes an animated GIF into the planar signed-unit
// source video the reference edit consumes, compositing each frame over
// the logical screen so a partial frame resolves against its predecessors.
func SourceVideoFromGIF(data []byte) (SourceVideo, error) {
	animation, err := gif.DecodeAll(bytes.NewReader(data))
	if err != nil {
		return SourceVideo{}, err
	}
	frames := len(animation.Image)
	if frames == 0 {
		return SourceVideo{}, errors.New("latent video: GIF carries no frames")
	}
	bounds := image.Rect(0, 0, animation.Config.Width, animation.Config.Height)
	if bounds.Empty() {
		bounds = animation.Image[0].Bounds()
	}
	height, width := bounds.Dy(), bounds.Dx()
	canvas := image.NewRGBA(bounds)
	plane := height * width
	pixels := make([]float32, media.RGBChannels*frames*plane)
	for frame, paletted := range animation.Image {
		draw.Draw(canvas, paletted.Bounds(), paletted, paletted.Bounds().Min, draw.Over)
		for y := range height {
			for x := range width {
				offset := canvas.PixOffset(bounds.Min.X+x, bounds.Min.Y+y)
				for channel := range media.RGBChannels {
					pixels[(channel*frames+frame)*plane+y*width+x] = media.U8ToSignedUnitF32(canvas.Pix[offset+channel])
				}
			}
		}
	}
	return SourceVideo{Pixels: pixels, Channels: media.RGBChannels, Frames: frames, Height: height, Width: width}, nil
}

// ResolveClipRequest completes a clip-form request against the model
// directory and the store: the source clip decodes from its GIF artifact,
// the text context comes from the base model's text pipeline over the
// prompt, and the condition follows the declared edit policy with sigmas
// compiled for its timesteps and the request's seed.
func ResolveClipRequest(ctx context.Context, reader artifact.Reader, modelDirectory string, request ReferenceEditRequest) (ReferenceEditRequest, error) {
	if !request.ClipForm() {
		return ReferenceEditRequest{}, errors.New("latent video: request is not a clip form")
	}
	if err := ValidateReferenceEditRequestForm(request); err != nil {
		return ReferenceEditRequest{}, err
	}
	content, found, err := artifact.ReadContent(ctx, reader, request.SourceArtifact)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	if !found {
		return ReferenceEditRequest{}, fmt.Errorf("latent video: source clip %s is absent", request.SourceArtifact)
	}
	if content.Descriptor.MediaType != media.GIFMediaType {
		return ReferenceEditRequest{}, fmt.Errorf("latent video: source clip %s is %s, not a GIF", request.SourceArtifact, content.Descriptor.MediaType)
	}
	source, err := SourceVideoFromGIF(content.Data)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	policy, err := ResolveEditPolicy(modelDirectory)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	wanDirectory, _ := LiveEditArtifacts(modelDirectory)
	profile, err := ResolveProfile(wanDirectory)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	volume, err := media.DownsampledVolume(source.Frames, source.Height, source.Width, profile.Policy.VAEStride)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	if err := policy.admitLatentFrames(volume.Frames); err != nil {
		return ReferenceEditRequest{}, err
	}
	base, err := LoadDenoiserConfig(wanDirectory, profile.Policy)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	conditioning, err := TextConditioning(WanTextConditioningSpec(wanDirectory, base.TextLen), request.Prompt)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	sigmas, err := sampling.CompileEditFlowSigmas(sampling.EditFlowConfig{
		InferenceSteps: policy.InferenceSteps, TrainTimesteps: profile.Policy.NumTrainTimesteps,
		Shift: policy.Shift, SigmaMin: policy.SigmaMin, SigmaMax: policy.SigmaMax, ExtraStep: policy.ExtraStep,
	}, policy.DenoisingTimesteps)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	resolved := ReferenceEditRequest{
		Condition: ReferenceEditCondition{
			TextContext: conditioning.Context, FramesPerChunk: policy.FramesPerChunk, LocalAttention: policy.LocalAttentionFrames,
			Timesteps: policy.DenoisingTimesteps, Sigmas: sigmas, ContextTimestep: policy.ContextTimestep, Seed: request.Seed,
		},
		Source: source,
	}
	return resolved, ValidateReferenceEditRequest(resolved)
}
