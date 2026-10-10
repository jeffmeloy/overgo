//go:build windows

package latentvideo

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/latentimage"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor"
	"overgo/internal/workflowruntime"
)

// AutoencoderRuntime keeps a Wan autoencoder's encoder and decoder on the
// device, so a picture or a clip makes the round trip on request rather than
// only inside a generator.
type AutoencoderRuntime struct {
	graph   vaeEncoderPlan
	stats   VAELatentStats
	encoder *vaeEncoderCUDASession
	decoder *vaeDecoderCUDASession
}

// LoadAutoencoderRuntime compiles both halves from the checkpoint in a Wan
// directory and loads them on the default device. A clip large enough to
// repay it is convolved through cuDNN on a machine that has it; a picture is
// not (libraryRoute).
func LoadAutoencoderRuntime(directory string) (runtime *AutoencoderRuntime, err error) {
	checkpoint := AutoencoderWeights(directory)
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		return nil, err
	}
	graph, err := CompileVAEEncoderPlan(catalog.Tensors)
	if err != nil {
		return nil, err
	}
	decoderPlan, err := CompileVAEDecoderPlan(catalog.Tensors)
	if err != nil {
		return nil, err
	}
	stats, err := declaredLatentStats(graph.LatentChannels)
	if err != nil {
		return nil, err
	}
	runtime = &AutoencoderRuntime{graph: graph, stats: stats}
	defer func() {
		if err != nil {
			err = errors.Join(err, runtime.Close(context.Background()))
			runtime = nil
		}
	}()
	if runtime.encoder, err = newVAEEncoderCUDASession(checkpoint, graph, device.DefaultOrdinal()); err != nil {
		return runtime, err
	}
	if runtime.decoder, err = NewVAEDecoderCUDASession(checkpoint, decoderPlan, device.DefaultOrdinal()); err == nil {
		runtime.encoder.codec.adoptLibrary(libraryClipValues)
		runtime.decoder.adoptLibrary(libraryClipValues)
	}
	return runtime, err
}

// RoundTrip encodes planar signed-unit frames into the normalized latent and
// decodes that latent back. It returns the frames that come out and the
// latent's extent. The height and width are multiples of the autoencoder's
// stride, and a clip's frame count is one the causal decoder gives back: one
// more than a multiple of the temporal stride.
func (r *AutoencoderRuntime) RoundTrip(ctx context.Context, source SourceVideo) (SourceVideo, SourceVideoShape, error) {
	if r == nil || r.encoder == nil || r.decoder == nil {
		return SourceVideo{}, SourceVideoShape{}, errors.New("latent video: the autoencoder runtime is closed")
	}
	plan, err := CompileSourceCodecBoundary(SourceCodecProfile{
		InputChannels: r.graph.InputChannels, LatentChannels: r.graph.LatentChannels,
		Stride: r.graph.Stride, LatentStats: r.stats,
	}, SourceVideoShape{Channels: source.Channels, Frames: source.Frames, Height: source.Height, Width: source.Width})
	if err != nil {
		return SourceVideo{}, SourceVideoShape{}, err
	}
	latent, _, err := r.encoder.Encode(ctx, plan, source.Pixels)
	if err != nil {
		return SourceVideo{}, SourceVideoShape{}, err
	}
	decoded := source
	decoded.Pixels = make([]float32, len(source.Pixels))
	plane, written := source.Height*source.Width, tensor.FirstOffset
	_, err = r.decoder.DecodeContext(ctx, r.stats, latent, plan.Latent.Frames, plan.Latent.Height, plan.Latent.Width,
		func(index int, frame []float32, height, width int) error {
			if index != written || index >= source.Frames || height != source.Height || width != source.Width {
				return errors.New("latent video: the autoencoder gave back a frame outside the source's extent")
			}
			written++
			return media.CopyPlanarFrames(decoded.Pixels, source.Frames, index, frame, tensor.SingletonExtent, tensor.FirstOffset, source.Channels, tensor.SingletonExtent, plane)
		})
	if err == nil && written != source.Frames {
		err = errors.New("latent video: the autoencoder gave back fewer frames than it took")
	}
	return decoded, plan.Latent, err
}

// Reconstruct passes one encoded picture through the autoencoder and
// returns what comes out as a PNG of the picture's own size.
func (r *AutoencoderRuntime) Reconstruct(ctx context.Context, request ReconstructionRequest) (latentimage.EncodedImage, error) {
	if r == nil || r.encoder == nil {
		return latentimage.EncodedImage{}, errors.New("latent video: the autoencoder runtime is closed")
	}
	source, height, width, err := sourcePicture(request.Image, r.graph.Stride[tensor.SingletonExtent], r.graph.Stride[tensor.PairedExtent])
	if err != nil {
		return latentimage.EncodedImage{}, err
	}
	decoded, _, err := r.RoundTrip(ctx, source)
	if err != nil {
		return latentimage.EncodedImage{}, err
	}
	return latentimage.EncodePlanarPNG(cropPicture(decoded, height, width), height, width)
}

// Reset readies the resident autoencoder for the next request; it holds
// nothing of the last one.
func (r *AutoencoderRuntime) Reset(context.Context, ReconstructionRequest) error {
	if r == nil || r.encoder == nil || r.decoder == nil {
		return errors.New("latent video: the autoencoder runtime is closed")
	}
	return nil
}

// PeakDeviceBytes reports the larger peak of the two halves' allocation
// ledgers.
func (r *AutoencoderRuntime) PeakDeviceBytes() (uint64, error) {
	if r == nil || r.encoder == nil || r.decoder == nil {
		return 0, errors.New("latent video: the autoencoder runtime is closed")
	}
	return peakAcrossSessions(r.encoder.codec, r.decoder)
}

// Close releases both halves from the device.
func (r *AutoencoderRuntime) Close(context.Context) error {
	if r == nil {
		return nil
	}
	var errs []error
	if r.encoder != nil {
		errs = append(errs, r.encoder.Close())
	}
	if r.decoder != nil {
		errs = append(errs, r.decoder.Close())
	}
	r.encoder, r.decoder = nil, nil
	return errors.Join(errs...)
}

// RegisterAutoencoderRuntime binds one loaded autoencoder to its round-trip
// stage; the stage's stored output is the PNG.
func RegisterAutoencoderRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, model *AutoencoderRuntime) error {
	if model == nil || model.encoder == nil || model.decoder == nil {
		return errors.New("latent video: incomplete autoencoder runtime binding")
	}
	return workflowruntime.RegisterContextStage(runtime, modelrecipe.ModuleReconstructImage, modelID, model.Reconstruct, latentimage.PNGContent)
}
