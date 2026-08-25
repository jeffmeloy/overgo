//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sync"

	"overgo/internal/checked"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/graphruntime"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// ResidentImagePipeline owns one device runtime across text and denoise stages.
type ResidentImagePipeline struct {
	Encoder  *EncoderProgram
	Fusion   *FusionProgram
	Denoiser *DenoiserProgram
	VAE      *VAEProgram

	runtime          *graphruntime.ResidentSession
	encoder          *graphruntime.ResidentProgram
	bridge           *graphruntime.ResidentProgram
	bridgePointers   []driver.DevicePtr
	bridgeOutput     *tensor.Tensor
	fusion           *graphruntime.ResidentProgram
	timestepPlan     *timestepProgram
	timestep         *graphruntime.ResidentProgram
	denoiser         *graphruntime.ResidentProgram
	denoiserInputs   [4]driver.DevicePtr
	upload           *graphruntime.ResidentProgram
	uploadInput      *tensor.Tensor
	uploadOutput     *tensor.Tensor
	vaeBridge        *graphruntime.ResidentProgram
	vaeOutput        *tensor.Tensor
	vae              *graphruntime.ResidentProgram
	timestepEncoding media.SinusoidalProgram
	residentBytes    uint64
	modelDir         string
	mu               sync.Mutex
	conditioning     *ResidentConditioning
	latent           *residentLatent
}

// ResidentConditioning owns fused text conditioning on the pipeline device.
type ResidentConditioning struct {
	pipeline *ResidentImagePipeline
	outputs  *executor.RetainedOutputs
	value    executor.DeviceValue
}

type residentLatent struct {
	outputs *executor.RetainedOutputs
	node    *tensor.Tensor
	value   executor.DeviceValue
}

func NewResidentImagePipeline(
	ctx context.Context,
	encoder *EncoderProgram,
	fusion *FusionProgram,
	denoiser *DenoiserProgram,
	vae *VAEProgram,
	timestepEncoding media.SinusoidalProgram,
	modelDir string,
	ordinal int,
) (pipeline *ResidentImagePipeline, err error) {
	if encoder == nil || fusion == nil || denoiser == nil || vae == nil || timestepEncoding.Validate() != nil {
		return nil, errors.New("resident image pipeline: program is nil")
	}
	if encoder.Seq < fusion.TextSeq || fusion.TextSeq != denoiser.TextSeq ||
		len(encoder.Selected) != fusion.T.TextLayers || encoder.E.Hidden != fusion.T.TextHidden {
		return nil, errors.New("resident image pipeline: text programs are incompatible")
	}
	runtime, err := graphruntime.NewResidentSession(ordinal)
	if err != nil {
		return nil, fmt.Errorf("resident image pipeline: device: %w", err)
	}
	pipeline = &ResidentImagePipeline{
		Encoder: encoder, Fusion: fusion, Denoiser: denoiser, VAE: vae, runtime: runtime,
		timestepEncoding: timestepEncoding, modelDir: modelDir,
	}
	defer func() {
		if err != nil {
			_ = pipeline.Close(ctx)
		}
	}()
	pipeline.encoder, err = compileResidentWeights(
		ctx, runtime, "resident image encoder", filepath.Join(modelDir, "text_encoder"),
		encoder.weightInputs, nil, encoder.Selected...,
	)
	if err != nil {
		return nil, err
	}
	bridgeInputs, bridgeOutput, err := selectedHiddenBridge(encoder, fusion.TextSeq)
	if err != nil {
		return nil, err
	}
	pipeline.bridgeOutput = bridgeOutput
	pipeline.bridge, err = runtime.Compile(
		ctx, "resident image text bridge", bridgeInputs, bridgeOutput,
	)
	if err != nil {
		return nil, err
	}
	pipeline.bridgePointers = make([]driver.DevicePtr, len(bridgeInputs))
	pipeline.fusion, err = compileResidentWeights(
		ctx, runtime, "resident image fusion", filepath.Join(modelDir, "transformer"),
		fusion.weightInputs, []*tensor.Tensor{fusion.InEncoder}, fusion.Fused,
	)
	if err != nil {
		return nil, err
	}
	if fusion.keyBias != nil {
		if err := runtime.BindF32(
			ctx, pipeline.fusion, "static:resident image fusion", fusion.keyBias, fusion.keyData,
		); err != nil {
			return nil, err
		}
	}
	pipeline.residentBytes = runtime.StaticBytes()
	return pipeline, nil
}

func (p *ResidentImagePipeline) prepareGeneration(ctx context.Context) (err error) {
	modelDir, denoiser, vae := p.modelDir, p.Denoiser, p.VAE
	p.timestepPlan, err = compileTimestepProgram(denoiser.T, denoiser.MatmulType, p.timestepEncoding)
	if err != nil {
		return err
	}
	p.timestep, err = compileResidentWeights(
		ctx, p.runtime, "resident image timestep", filepath.Join(modelDir, "transformer"),
		p.timestepPlan.weightInputs, nil,
		p.timestepPlan.Embedding, p.timestepPlan.Modulation,
	)
	if err != nil {
		return err
	}
	p.denoiser, err = compileResidentWeights(
		ctx, p.runtime, "resident image denoiser", filepath.Join(modelDir, "transformer"),
		denoiser.weightInputs,
		[]*tensor.Tensor{denoiser.InLatent, denoiser.InText, denoiser.InTemb, denoiser.InTembMod},
		denoiser.NextLatent,
	)
	if err != nil {
		return err
	}
	if denoiser.keyBias != nil {
		if err := p.runtime.BindF32(
			ctx, p.denoiser, "static:resident image denoiser", denoiser.keyBias, denoiser.keyData,
		); err != nil {
			return err
		}
	}
	p.uploadInput, p.uploadOutput, err = latentUploadGraph(denoiser)
	if err != nil {
		return err
	}
	p.upload, err = p.runtime.Compile(
		ctx, "resident image latent upload", nil, p.uploadOutput,
	)
	if err != nil {
		return err
	}
	vaeStatic := make(map[*tensor.Tensor][]float32, len(vae.feeds))
	for _, feed := range vae.feeds {
		vaeStatic[feed.node] = feed.data
	}
	p.vae, err = compileResidentStatic(
		ctx, p.runtime, "resident image VAE", vaeStatic, []*tensor.Tensor{vae.Latent}, vae.Output,
	)
	if err != nil {
		return err
	}
	vaeInput, vaeOutput, vaeStatic, err := latentVAEDecodeBridge(denoiser, vae)
	if err != nil {
		return err
	}
	p.vaeOutput = vaeOutput
	p.vaeBridge, err = compileResidentStatic(
		ctx, p.runtime, "resident image VAE bridge", vaeStatic, []*tensor.Tensor{vaeInput}, vaeOutput,
	)
	if err != nil {
		return err
	}
	for index := range vae.feeds {
		vae.feeds[index].data = nil
	}
	p.runtime.Seal()
	p.residentBytes = p.runtime.StaticBytes()
	return nil
}

func latentVAEDecodeBridge(
	denoiser *DenoiserProgram,
	vae *VAEProgram,
) (*tensor.Tensor, *tensor.Tensor, map[*tensor.Tensor][]float32, error) {
	patch, patchErr := media.SquarePatchExtent(denoiser.T.InChannels, vae.ZDim)
	if patchErr != nil || !checked.Equal(vae.H, denoiser.GH*patch) || !checked.Equal(vae.W, denoiser.GW*patch) {
		return nil, nil, nil, fmt.Errorf(
			"resident image pipeline: VAE geometry %dx%d/%d is incompatible with denoiser %dx%d/%d",
			vae.W, vae.H, vae.ZDim, denoiser.GW, denoiser.GH, denoiser.T.InChannels,
		)
	}
	patchArea, _ := checked.MulInt(patch, patch)
	builder := tensor.NewBuilder()
	input := builder.Input("packed_latent", dtype.F32, denoiser.InLatent.Shape)
	var windows *tensor.Tensor
	for local := range patchArea {
		channels := builder.Reshape(
			builder.GroupSlice(input, uint64(local), tensor.SingletonExtent, uint64(vae.ZDim), uint64(patchArea)),
			uint64(vae.ZDim), tensor.SingletonExtent, uint64(denoiser.ImgSeq),
		)
		if windows == nil {
			windows = channels
		} else {
			windows = builder.Concat(windows, channels, tensor.SingletonExtent)
		}
	}
	spatial := builder.Reshape(
		builder.WindowUnpartition2D(windows, uint32(vae.W), uint32(vae.H)),
		uint64(vae.ZDim), uint64(vae.H*vae.W),
	)
	mean := builder.Input("vae_latent_mean", dtype.F32, tensor.MustShape(uint64(vae.ZDim)))
	std := builder.Input("vae_latent_std", dtype.F32, tensor.MustShape(uint64(vae.ZDim)))
	output := builder.Add(builder.Multiply(spatial, std), mean)
	if err := builder.Err(); err != nil {
		return nil, nil, nil, fmt.Errorf("resident image pipeline: VAE bridge: %w", err)
	}
	return input, output, map[*tensor.Tensor][]float32{
		mean: vae.latentMean, std: vae.latentStd,
	}, nil
}

func latentUploadGraph(program *DenoiserProgram) (*tensor.Tensor, *tensor.Tensor, error) {
	builder := tensor.NewBuilder()
	input := builder.Input("initial_latent", dtype.F32, program.InLatent.Shape)
	output := builder.BF16Round(input)
	if err := builder.Err(); err != nil {
		return nil, nil, fmt.Errorf("resident image pipeline: latent upload: %w", err)
	}
	return input, output, nil
}

func selectedHiddenBridge(program *EncoderProgram, outputSeq int) ([]*tensor.Tensor, *tensor.Tensor, error) {
	if program == nil {
		return nil, nil, errors.New("resident image pipeline: encoder program is nil")
	}
	if !checked.PositiveInts(outputSeq) || !checked.AtMostInt(outputSeq, program.Seq) {
		return nil, nil, errors.New("resident image pipeline: encoder output sequence is invalid")
	}
	builder := tensor.NewBuilder()
	inputs := make([]*tensor.Tensor, len(program.Selected))
	var joined *tensor.Tensor
	offset := uint64(program.Seq-outputSeq) * uint64(program.E.Hidden)
	for index, selected := range program.Selected {
		inputs[index] = builder.Input(
			fmt.Sprintf("selected_%d", index), dtype.F32, selected.Shape,
		)
		cropped := builder.FlatSlice(
			inputs[index], offset, uint64(program.E.Hidden), uint64(outputSeq),
		)
		if joined == nil {
			joined = cropped
		} else {
			joined = builder.Concat(joined, cropped, tensor.FirstOffset)
		}
	}
	if joined == nil {
		return nil, nil, errors.New("resident image pipeline: encoder has no selected outputs")
	}
	output := builder.Reshape(joined, uint64(program.E.Hidden), uint64(len(inputs)*outputSeq))
	if err := builder.Err(); err != nil {
		return nil, nil, fmt.Errorf("resident image pipeline: text bridge: %w", err)
	}
	return inputs, output, nil
}

func (p *ResidentImagePipeline) Condition(
	ctx context.Context,
	embedRows []float32,
) (*ResidentConditioning, error) {
	if p == nil {
		return nil, errors.New("resident image pipeline: unavailable")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.runtime == nil {
		return nil, errors.New("resident image pipeline: unavailable")
	}
	if p.conditioning != nil {
		return nil, errors.New("resident image pipeline: conditioning is already retained")
	}
	embedValue, err := reference.BorrowedValue(p.Encoder.Embed.Shape, embedRows)
	if err != nil {
		return nil, fmt.Errorf("resident image pipeline: embed rows: %w", err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		p.Encoder.Embed: embedValue,
	}
	if p.Encoder.keyBias != nil {
		feeds[p.Encoder.keyBias] = reference.Value{
			Shape: p.Encoder.keyBias.Shape, Data: p.Encoder.keyBiasData,
		}
	}
	encoded, err := p.runtime.Retain(ctx, p.encoder, feeds, nil)
	if err != nil {
		return nil, fmt.Errorf("resident image pipeline: encode: %w", err)
	}
	for index, output := range p.Encoder.Selected {
		value, ok := encoded.Value(output)
		if !ok {
			_ = encoded.Release(ctx)
			return nil, fmt.Errorf("resident image pipeline: selected output %d is unavailable", index)
		}
		p.bridgePointers[index] = value.Pointer
	}
	bridged, err := p.runtime.Retain(ctx, p.bridge, nil, p.bridgePointers)
	if err != nil {
		_ = encoded.Release(ctx)
		return nil, fmt.Errorf("resident image pipeline: bridge: %w", err)
	}
	if err := encoded.Release(ctx); err != nil {
		_ = bridged.Release(ctx)
		return nil, fmt.Errorf("resident image pipeline: release encoded text: %w", err)
	}
	bridgeValue, ok := bridged.Value(p.bridgeOutput)
	if !ok {
		_ = bridged.Release(ctx)
		return nil, errors.New("resident image pipeline: bridged text is unavailable")
	}
	fused, err := p.runtime.Retain(ctx, p.fusion, nil, []driver.DevicePtr{bridgeValue.Pointer})
	if err != nil {
		_ = bridged.Release(ctx)
		return nil, fmt.Errorf("resident image pipeline: fuse: %w", err)
	}
	if err := bridged.Release(ctx); err != nil {
		_ = fused.Release(ctx)
		return nil, fmt.Errorf("resident image pipeline: release bridged text: %w", err)
	}
	fusedValue, ok := fused.Value(p.Fusion.Fused)
	if !ok {
		_ = fused.Release(ctx)
		return nil, errors.New("resident image pipeline: fused text is unavailable")
	}
	if err := p.runtime.Release(ctx, p.encoder); err != nil {
		_ = fused.Release(ctx)
		return nil, fmt.Errorf("resident image pipeline: retire encoder graph: %w", err)
	}
	p.encoder = nil
	p.residentBytes = p.runtime.StaticBytes()
	if err := p.prepareGeneration(ctx); err != nil {
		_ = fused.Release(ctx)
		return nil, fmt.Errorf("resident image pipeline: prepare generation: %w", err)
	}
	if err := p.runtime.Release(ctx, p.fusion); err != nil {
		_ = fused.Release(ctx)
		return nil, fmt.Errorf("resident image pipeline: retire fusion graph: %w", err)
	}
	p.fusion = nil
	p.residentBytes = p.runtime.StaticBytes()
	conditioning := &ResidentConditioning{pipeline: p, outputs: fused, value: fusedValue}
	p.conditioning = conditioning
	return conditioning, nil
}

func (p *ResidentImagePipeline) Begin(ctx context.Context, latentPatches []float32) error {
	if p == nil {
		return errors.New("resident image pipeline: unavailable")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.runtime == nil || p.latent != nil {
		return errors.New("resident image pipeline: latent session is unavailable")
	}
	if want := p.Denoiser.ImgSeq * p.Denoiser.T.InChannels; len(latentPatches) != want {
		return fmt.Errorf("resident image pipeline: latent len=%d want %d", len(latentPatches), want)
	}
	outputs, err := p.runtime.Retain(ctx, p.upload, map[*tensor.Tensor]reference.Value{
		p.uploadInput: {Shape: p.uploadInput.Shape, Data: latentPatches},
	}, nil)
	if err != nil {
		return fmt.Errorf("resident image pipeline: upload latent: %w", err)
	}
	value, ok := outputs.Value(p.uploadOutput)
	if !ok {
		_ = outputs.Release(ctx)
		return errors.New("resident image pipeline: uploaded latent is unavailable")
	}
	p.latent = &residentLatent{outputs: outputs, node: p.uploadOutput, value: value}
	return nil
}

func (p *ResidentImagePipeline) Advance(ctx context.Context, sigma, delta float64) error {
	if p == nil {
		return errors.New("resident image pipeline: unavailable")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.runtime == nil || p.conditioning == nil || p.latent == nil {
		return errors.New("resident image pipeline: denoise session is unavailable")
	}
	sinusoid, err := p.timestepPlan.sinusoid(sigma)
	if err != nil {
		return err
	}
	step, err := p.runtime.Retain(ctx, p.timestep, map[*tensor.Tensor]reference.Value{
		p.timestepPlan.Input: {Shape: p.timestepPlan.Input.Shape, Data: sinusoid},
	}, nil)
	if err != nil {
		return fmt.Errorf("resident image pipeline: timestep: %w", err)
	}
	embedding, embeddingOK := step.Value(p.timestepPlan.Embedding)
	modulation, modulationOK := step.Value(p.timestepPlan.Modulation)
	if !embeddingOK || !modulationOK {
		_ = step.Release(ctx)
		return errors.New("resident image pipeline: timestep outputs are unavailable")
	}
	program := p.Denoiser
	p.denoiserInputs = [4]driver.DevicePtr{
		p.latent.value.Pointer, p.conditioning.value.Pointer, embedding.Pointer, modulation.Pointer,
	}
	next, err := p.runtime.Retain(
		ctx, p.denoiser,
		map[*tensor.Tensor]reference.Value{
			program.InDelta: {Shape: program.InDelta.Shape, Data: []float32{float32(delta)}},
		},
		p.denoiserInputs[:],
	)
	if err != nil {
		_ = step.Release(ctx)
		return fmt.Errorf("resident image pipeline: denoise: %w", err)
	}
	nextValue, ok := next.Value(program.NextLatent)
	if !ok {
		_ = next.Release(ctx)
		_ = step.Release(ctx)
		return errors.New("resident image pipeline: next latent is unavailable")
	}
	if err := step.Release(ctx); err != nil {
		_ = next.Release(ctx)
		return fmt.Errorf("resident image pipeline: release timestep: %w", err)
	}
	if err := p.latent.outputs.Release(ctx); err != nil {
		_ = next.Release(ctx)
		return fmt.Errorf("resident image pipeline: release prior latent: %w", err)
	}
	p.latent = &residentLatent{outputs: next, node: program.NextLatent, value: nextValue}
	return nil
}

func (p *ResidentImagePipeline) Latent(ctx context.Context) ([]float32, error) {
	if p == nil {
		return nil, errors.New("resident image pipeline: unavailable")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.latent == nil {
		return nil, errors.New("resident image pipeline: latent is unavailable")
	}
	value, err := p.latent.outputs.CopyToHost(ctx, p.latent.node)
	if err != nil {
		return nil, fmt.Errorf("resident image pipeline: download latent: %w", err)
	}
	return value.Data, nil
}

// DecodeHWC downloads the VAE's native HWC output without a planar copy.
func (p *ResidentImagePipeline) DecodeHWC(ctx context.Context) ([]float32, int, int, error) {
	if p == nil {
		return nil, 0, 0, errors.New("resident image pipeline: unavailable")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.runtime == nil || p.latent == nil || p.vae == nil {
		return nil, 0, 0, errors.New("resident image pipeline: decode session is unavailable")
	}
	bridged, err := p.runtime.Retain(ctx, p.vaeBridge, nil, []driver.DevicePtr{p.latent.value.Pointer})
	if err != nil {
		return nil, 0, 0, fmt.Errorf("resident image pipeline: bridge VAE latent: %w", err)
	}
	value, ok := bridged.Value(p.vaeOutput)
	if !ok {
		_ = bridged.Release(ctx)
		return nil, 0, 0, errors.New("resident image pipeline: VAE latent is unavailable")
	}
	results, err := p.runtime.ExecuteDevice(ctx, p.vae, nil, []driver.DevicePtr{value.Pointer})
	releaseErr := bridged.Release(ctx)
	if err != nil || releaseErr != nil {
		return nil, 0, 0, errors.Join(err, releaseErr)
	}
	return results[p.VAE.Output].Data, p.VAE.OutH, p.VAE.OutW, nil
}

// Finish releases request-local latent state; compiled graphs remain resident.
func (p *ResidentImagePipeline) Finish(ctx context.Context) error {
	if p == nil {
		return errors.New("resident image pipeline: unavailable")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.runtime == nil || p.latent == nil {
		return errors.New("resident image pipeline: generation is unavailable")
	}
	err := p.latent.outputs.Release(ctx)
	if err == nil {
		p.latent = nil
	}
	return err
}

func (c *ResidentConditioning) Release(ctx context.Context) error {
	if c == nil || c.pipeline == nil {
		return nil
	}
	pipeline := c.pipeline
	pipeline.mu.Lock()
	defer pipeline.mu.Unlock()
	if pipeline.conditioning != c || c.outputs == nil {
		return nil
	}
	err := c.outputs.Release(ctx)
	if err == nil {
		pipeline.conditioning = nil
		c.outputs = nil
		c.value = executor.DeviceValue{}
		c.pipeline = nil
	}
	return err
}

func (p *ResidentImagePipeline) ResidentBytes() uint64 {
	if p == nil {
		return uint64(tensor.FirstOffset)
	}
	return p.residentBytes
}

func (p *ResidentImagePipeline) Close(ctx context.Context) error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.runtime == nil {
		return nil
	}
	var releaseErr error
	if p.latent != nil {
		releaseErr = p.latent.outputs.Release(ctx)
		p.latent = nil
	}
	if p.conditioning != nil {
		releaseErr = errors.Join(releaseErr, p.conditioning.outputs.Release(ctx))
		p.conditioning.outputs = nil
		p.conditioning.value = executor.DeviceValue{}
		p.conditioning.pipeline = nil
		p.conditioning = nil
	}
	err := errors.Join(releaseErr, p.runtime.Close(ctx))
	p.runtime = nil
	p.encoder, p.bridge, p.fusion, p.timestep, p.denoiser, p.upload = nil, nil, nil, nil, nil, nil
	p.vaeBridge, p.vae = nil, nil
	return err
}
