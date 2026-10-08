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
	"overgo/internal/tensor"
)

//go:embed edit_profiles.json
var editPolicyCatalogJSON []byte

// editPolicy declares the reference-edit schedule a LiveEdit checkpoint
// runs under: the causal chunking, the denoising timesteps, and the flow
// schedule their sigmas compile from, and the size of the clips the editor
// was trained to change. The catalog restates the upstream few-step
// configuration (LiveEdit/configs/wan_mm-ar-forcing-local.yaml), whose latent
// of 60 by 104 is a clip of 480 by 832.
type editPolicy struct {
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
	SourceWidth          int     `json:"source_width"`
	SourceHeight         int     `json:"source_height"`
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

// resolveEditPolicy reads the declared policy for the checkpoint a LiveEdit
// directory carries, refusing an absent checkpoint or an undeclared one.
func resolveEditPolicy(modelDirectory string) (editPolicy, error) {
	var catalog []editPolicy
	if err := strictjson.DecodeBytes(editPolicyCatalogJSON, &catalog); err != nil {
		return editPolicy{}, fmt.Errorf("latent video: decode edit policy catalog: %w", err)
	}
	_, checkpoint := LiveEditArtifacts(modelDirectory)
	for _, policy := range catalog {
		if policy.Checkpoint != filepath.Base(checkpoint) {
			continue
		}
		if _, err := os.Stat(checkpoint); err != nil {
			return editPolicy{}, err
		}
		return policy, policy.validate()
	}
	return editPolicy{}, fmt.Errorf("latent video: no edit policy declares checkpoint %s", filepath.Base(checkpoint))
}

func (policy editPolicy) validate() error {
	if !checked.PositiveInts(policy.FramesPerChunk, policy.LocalAttentionFrames, policy.InferenceSteps, policy.SourceWidth, policy.SourceHeight) {
		return errors.New("latent video: incomplete edit policy")
	}
	if _, ok := checked.First(policy.DenoisingTimesteps); !ok || policy.ContextTimestep < 0 {
		return errors.New("latent video: incomplete edit policy")
	}
	return nil
}

// EditSourceSize is the width and height the editor in a LiveEdit directory
// was trained to change a clip at, as its policy declares. It edits a clip of
// another size too, but at a quarter of this one it changes a clip's texture
// and not what the instruction asks, so a clip cut for it is cut to this.
func EditSourceSize(modelDirectory string) (width, height int, err error) {
	policy, err := resolveEditPolicy(modelDirectory)
	return policy.SourceWidth, policy.SourceHeight, err
}

// admitLatentFrames restates the edit denoiser's chunk arithmetic for a
// clip: its latent frames must fill whole chunks beyond the first and fit
// the local attention history, so a clip the policy cannot chunk is
// refused with the counts it admits.
func (policy editPolicy) admitLatentFrames(latent int) error {
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

// validateReferenceEditRequestForm admits the complete request or the clip
// form: a prompt, a source clip artifact id, and a non-negative seed.
func validateReferenceEditRequestForm(request ReferenceEditRequest) error {
	if !request.ClipForm() {
		return ValidateReferenceEditRequest(request)
	}
	if request.Seed < 0 {
		return errors.New("latent video: negative LiveEdit seed")
	}
	return nil
}

// sourceVideoFromGIF decodes an animated GIF into the planar source video
// the reference edit consumes, compositing each frame over the logical
// screen so a partial frame resolves against its predecessors. Its values
// run from 0 to 1, not from -1 to 1 as an autoencoder's pictures otherwise
// do: the edit checkpoint was trained on source clips read that way (the
// reference reads its source without centering it), and a source centered
// on zero is a clip it was never shown. Each plane is then smoothed over the
// reach of a palette's dither (smoothDither).
func sourceVideoFromGIF(data []byte) (SourceVideo, error) {
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
					pixels[(channel*frames+frame)*plane+y*width+x] = media.U8ToUnitF32(canvas.Pix[offset+channel])
				}
			}
		}
	}
	scratch := make([]float32, plane)
	for part := range media.RGBChannels * frames {
		smoothDither(pixels[part*plane:][:plane], scratch, height, width)
	}
	return SourceVideo{Pixels: pixels, Channels: media.RGBChannels, Frames: frames, Height: height, Width: width}, nil
}

// smoothDither replaces each value of one plane by the mean of its three by
// three neighbourhood, its own row and column counted twice, the plane's
// edge repeated beyond it. A clip arrives as palette pictures, and a palette
// picture of a real scene is dithered: its colours are right over a few
// pixels and wrong pixel by pixel. The editor reads that grain as content
// and edits what it was not asked to; on its reference clip, given through a
// GIF, it turns a strawberry into a dark fruit beside the grapes it was asked
// for. Smoothed over the dither's reach, the clip is closer to the scene
// than the dithered one, and the edit is the edit of the scene. scratch
// holds a plane.
func smoothDither(values, scratch []float32, height, width int) {
	const center, sum = tensor.PairedExtent, tensor.PairedExtent + tensor.PairedExtent
	for y := range height {
		row := values[y*width:][:width]
		for x := range width {
			scratch[y*width+x] = (row[max(x-1, 0)] + center*row[x] + row[min(x+1, width-1)]) / sum
		}
	}
	for y := range height {
		above, below := scratch[max(y-1, 0)*width:][:width], scratch[min(y+1, height-1)*width:][:width]
		for x := range width {
			values[y*width+x] = (above[x] + center*scratch[y*width+x] + below[x]) / sum
		}
	}
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
	if err := validateReferenceEditRequestForm(request); err != nil {
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
	source, err := sourceVideoFromGIF(content.Data)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	policy, err := resolveEditPolicy(modelDirectory)
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
	// The prompt is encoded by the base model's text encoder and projected by
	// the edit checkpoint's own projection, which its training moved from the
	// base model's: a context projected by the base's is not the one the edit
	// denoiser was trained to read.
	_, checkpointPath := LiveEditArtifacts(modelDirectory)
	checkpoint, err := CompileReferenceEditCheckpoint(checkpointPath, base)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	projection, projectionBytes, err := checkpoint.loadTextProjection()
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	conditioning, err := textConditioningWithWeights(ctx, WanTextConditioningSpec(wanDirectory, base.TextLen), request.Prompt, projection, projectionBytes)
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
