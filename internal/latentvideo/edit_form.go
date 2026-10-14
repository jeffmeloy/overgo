package latentvideo

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/processmeasure"
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
// form: a prompt, a source clip artifact id, a non-negative seed and a clip
// format.
func validateReferenceEditRequestForm(request ReferenceEditRequest) error {
	if !request.ClipForm() {
		return ValidateReferenceEditRequest(request)
	}
	if request.Seed < 0 {
		return errors.New("latent video: negative LiveEdit seed")
	}
	return request.Format.validate()
}

// sourceVideoFromGIF decodes an animated GIF into the planar source video
// the reference edit consumes, every frame of it composed as a video
// prompt's is (media.DecodeGIF): a frame that covers part of the picture is
// drawn over the ones before it, and what it says to clear or to put back is
// cleared or put back before the next. Its values run from 0 to 1, not from
// -1 to 1 as an autoencoder's pictures otherwise do: the edit checkpoint was
// trained on source clips read that way (the reference reads its source
// without centering it), and a source centered on zero is a clip it was
// never shown. Each plane is then smoothed over the reach of a palette's
// dither (smoothDither).
func sourceVideoFromGIF(data []byte) (SourceVideo, error) {
	frames, err := media.DecodeGIF(bytes.NewReader(data), math.MaxInt)
	if err != nil {
		return SourceVideo{}, err
	}
	source, err := planarSource(frames)
	if err != nil {
		return SourceVideo{}, err
	}
	plane := source.Height * source.Width
	scratch := make([]float32, plane)
	for part := range media.RGBChannels * source.Frames {
		smoothDither(source.Pixels[part*plane:][:plane], scratch, source.Height, source.Width)
	}
	return source, nil
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

// sourceFramesContract is a source clip that keeps every pixel: its frames,
// each a PNG, one after another.
var sourceFramesContract = artifact.DocumentContract{
	Kind: artifact.KindFile, MediaType: media.PNGFramesMediaType, Schema: "overgo.source-clip.png-frames.v1",
}

// SourceFramesContent is a clip an editor is to change, as an artifact that
// keeps every pixel of it at eight bits a colour: each frame a PNG, one after
// another. A GIF keeps 256 colours a frame, and a source that arrives as one
// is smoothed over its dither's reach (smoothDither), which takes with the
// grain detail the editor would have kept. A sender that holds the frames
// sends them this way, and the editor reads them as they are. The clip's
// values run from 0 to 1, in the planar layout the edit consumes.
func SourceFramesContent(source SourceVideo) (artifact.Content, error) {
	if !checked.Equal(source.Channels, media.RGBChannels) {
		return artifact.Content{}, errors.New("latent video: a source clip holds three colours a pixel")
	}
	if err := media.ValidatePlanarVideo(source.Pixels, source.Channels, source.Frames, source.Height, source.Width); err != nil {
		return artifact.Content{}, err
	}
	plane := source.Height * source.Width
	picture := image.NewRGBA(image.Rect(0, 0, source.Width, source.Height))
	draw.Draw(picture, picture.Bounds(), image.Opaque, image.Point{}, draw.Src)
	var written bytes.Buffer
	for frame := range source.Frames {
		for index := range plane {
			offset := picture.PixOffset(index%source.Width, index/source.Width)
			for channel := range media.RGBChannels {
				value := source.Pixels[(channel*source.Frames+frame)*plane+index]
				if !checked.Finite32(value) {
					return artifact.Content{}, errors.New("latent video: a source clip's value is not finite")
				}
				picture.Pix[offset+channel] = media.UnitF32ToU8(value)
			}
		}
		if err := png.Encode(&written, picture); err != nil {
			return artifact.Content{}, err
		}
	}
	return sourceFramesContract.OwnedContentBytes(written.Bytes())
}

// sourceVideoFromFrames decodes a clip carried as PNG frames, one after
// another, into the planar source video the reference edit consumes, its
// values from 0 to 1 as sourceVideoFromGIF gives them. Nothing is smoothed:
// the frames hold no dither. Every frame is the size of the first.
func sourceVideoFromFrames(data []byte) (SourceVideo, error) {
	reader := bytes.NewReader(data)
	var frames []image.Image
	for reader.Len() > 0 {
		frame, err := png.Decode(reader)
		if err != nil {
			return SourceVideo{}, fmt.Errorf("latent video: source frame %d: %w", len(frames), err)
		}
		frames = append(frames, frame)
	}
	return planarSource(frames)
}

// planarSource lays a clip's pictures out as the planar source video the
// reference edit consumes, its values from 0 to 1, and refuses a clip of no
// pictures or of pictures that are not all the size of the first.
func planarSource(frames []image.Image) (SourceVideo, error) {
	first, ok := checked.First(frames)
	if !ok {
		return SourceVideo{}, errors.New("latent video: the source clip carries no frames")
	}
	height, width := first.Bounds().Dy(), first.Bounds().Dx()
	plane := height * width
	pixels := make([]float32, media.RGBChannels*len(frames)*plane)
	canvas := image.NewRGBA(image.Rect(0, 0, width, height))
	for frame, picture := range frames {
		if picture.Bounds().Dy() != height || picture.Bounds().Dx() != width {
			return SourceVideo{}, fmt.Errorf("latent video: source frame %d is %dx%d where the first is %dx%d",
				frame, picture.Bounds().Dx(), picture.Bounds().Dy(), width, height)
		}
		draw.Draw(canvas, canvas.Bounds(), picture, picture.Bounds().Min, draw.Src)
		for index := range plane {
			offset := canvas.PixOffset(index%width, index/width)
			for channel := range media.RGBChannels {
				pixels[(channel*len(frames)+frame)*plane+index] = media.U8ToUnitF32(canvas.Pix[offset+channel])
			}
		}
	}
	return SourceVideo{Pixels: pixels, Channels: media.RGBChannels, Frames: len(frames), Height: height, Width: width}, nil
}

// admittedFrames is the most frames, at most count, of a clip whose latent
// frames the policy admits, or zero when it admits none of so few; stride is
// the autoencoder's temporal stride.
func (policy editPolicy) admittedFrames(count, stride int) int {
	latent := min((count-1)/stride+1, policy.LocalAttentionFrames+policy.FramesPerChunk)
	latent -= latent % policy.FramesPerChunk
	if latent <= policy.FramesPerChunk {
		return 0
	}
	return (latent-1)*stride + 1
}

// sourceVideoFromMP4 decodes an MP4 source clip through FFmpeg into the
// planar source video the reference edit consumes, its values from 0 to 1.
//   - frames are sampled at the profile's rate, scaled to cover the size the editor was trained at and cut to its middle (media.DecodeVideoFileCovering)
//   - the clip is cut to the most frames the policy admits, because an MP4 is a recording of any length and the editor takes a bounded one
//   - nothing is smoothed, because an MP4 frame holds no palette dither
func sourceVideoFromMP4(ctx context.Context, data []byte, policy editPolicy, profile Profile) (SourceVideo, error) {
	ffmpeg, err := media.FFmpeg()
	if err != nil {
		return SourceVideo{}, fmt.Errorf("latent video: an MP4 source clip is read through FFmpeg: %w", err)
	}
	stride := profile.Policy.VAEStride[0]
	most := policy.admittedFrames(math.MaxInt, stride)
	frames, err := media.DecodeEncodedVideoCovering(ctx, data, ffmpeg, float64(profile.SampleFPS), most, policy.SourceWidth, policy.SourceHeight)
	if err != nil {
		return SourceVideo{}, err
	}
	if kept := policy.admittedFrames(len(frames), stride); kept > 0 {
		frames = frames[:kept]
	}
	return planarSource(frames)
}

// sourceVideoFromContent decodes a source clip by the form it arrives in: PNG
// frames as they are, a GIF smoothed over its dither, or an MP4 cut for the
// editor.
func sourceVideoFromContent(ctx context.Context, content artifact.Content, policy editPolicy, profile Profile) (SourceVideo, error) {
	switch content.Descriptor.MediaType {
	case media.PNGFramesMediaType:
		return sourceVideoFromFrames(content.Data)
	case media.GIFMediaType:
		return sourceVideoFromGIF(content.Data)
	case media.MP4MediaType:
		return sourceVideoFromMP4(ctx, content.Data, policy, profile)
	}
	return SourceVideo{}, fmt.Errorf("latent video: source clip %s is %s, not a GIF, an MP4 or PNG frames", content.Descriptor.ID, content.Descriptor.MediaType)
}

// ResolveClipRequest completes a clip-form request against the model
// directory and the store: the source clip decodes from its artifact, PNG
// frames as they are, a GIF smoothed over its dither or an MP4 cut for the
// editor, the text context comes from the base model's text pipeline over the
// prompt, and the condition follows the declared edit policy with sigmas
// compiled for its timesteps, the request's seed and its clip format.
func ResolveClipRequest(ctx context.Context, reader artifact.Reader, modelDirectory string, request ReferenceEditRequest) (ReferenceEditRequest, error) {
	if !request.ClipForm() {
		return ReferenceEditRequest{}, errors.New("latent video: request is not a clip form")
	}
	if err := validateReferenceEditRequestForm(request); err != nil {
		return ReferenceEditRequest{}, err
	}
	// The policy and the profile say what a source clip is cut to, so they are read with it.
	read := processmeasure.Stage(ctx, "source clip read")
	policy, err := resolveEditPolicy(modelDirectory)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	wanDirectory, _ := LiveEditArtifacts(modelDirectory)
	profile, err := ResolveProfile(wanDirectory)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	content, found, err := artifact.ReadContent(ctx, reader, request.SourceArtifact)
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	if !found {
		return ReferenceEditRequest{}, fmt.Errorf("latent video: source clip %s is absent", request.SourceArtifact)
	}
	source, err := sourceVideoFromContent(ctx, content, policy, profile)
	read()
	if err != nil {
		return ReferenceEditRequest{}, err
	}
	defer processmeasure.Stage(ctx, "edit checkpoint compiled")()
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
			Format: request.Format,
		},
		Source: source,
	}
	return resolved, ValidateReferenceEditRequest(resolved)
}
