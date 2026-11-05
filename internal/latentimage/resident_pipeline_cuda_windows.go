//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
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
	// conditioned keeps the fused text of the session's recent prompts on the
	// host, and conditioningUpload places one on the device again for a prompt
	// met again, which then places neither the text encoder nor the fusion.
	conditioned        conditioningCache
	conditioningUpload *graphruntime.ResidentProgram
	conditioningInput  *tensor.Tensor
	conditioningOutput *tensor.Tensor
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
	// The cache holds two prompts' conditioning, so a session that alternates
	// two encodes each once.
	limit, ok := checked.ProductInt(tensor.PairedExtent, fusion.T.Hidden, fusion.TextSeq)
	if !ok {
		return nil, errors.New("resident image pipeline: conditioning cache limit overflows")
	}
	pipeline.conditioned = conditioningCache{limit: limit}
	pipeline.conditioningInput, pipeline.conditioningOutput, err = conditioningUploadGraph(fusion)
	if err != nil {
		return nil, err
	}
	pipeline.conditioningUpload, err = runtime.Compile(
		ctx, "resident image conditioning upload", nil, pipeline.conditioningOutput,
	)
	if err != nil {
		return nil, err
	}
	if err := pipeline.loadText(ctx); err != nil {
		return nil, err
	}
	pipeline.residentBytes = runtime.StaticBytes()
	return pipeline, nil
}

// loadText places the fusion of the pipeline's prompt on the device. It is
// there while a prompt's conditioning is made and retired after it, and the
// text encoder is placed a layer at a time as the prompt passes through it
// (encodeBlocks), so a session holds the denoiser and at most one layer of the
// encoder beside it.
func (p *ResidentImagePipeline) loadText(ctx context.Context) (err error) {
	p.fusion, err = compileResidentWeights(
		ctx, p.runtime, "resident image fusion", filepath.Join(p.modelDir, "transformer"),
		p.Fusion.weightInputs, []*tensor.Tensor{p.Fusion.InEncoder}, p.Fusion.Fused,
	)
	if err != nil || p.Fusion.keyBias == nil {
		return err
	}
	return p.runtime.BindF32(ctx, p.fusion, "static:resident image fusion", p.Fusion.keyBias, p.Fusion.keyData)
}

// bindDenoiser binds the pipeline's denoiser program to the transformer's
// weights, which the session shares by name: a program compiled for another
// prompt's mask takes the weights the program it replaces holds, and loads
// none. The replaced program is released before the new mask is bound, since
// the two masks have one name and a bound input is shared by its name.
func (p *ResidentImagePipeline) bindDenoiser(ctx context.Context, replaced *graphruntime.ResidentProgram) (*graphruntime.ResidentProgram, error) {
	denoiser := p.Denoiser
	program, err := compileResidentWeights(
		ctx, p.runtime, "resident image denoiser", filepath.Join(p.modelDir, "transformer"),
		denoiser.weightInputs,
		[]*tensor.Tensor{denoiser.InLatent, denoiser.InText, denoiser.InTemb, denoiser.InTembMod},
		denoiser.NextLatent,
	)
	if err == nil {
		err = p.runtime.Release(ctx, replaced)
	}
	if err != nil || denoiser.keyBias == nil {
		return program, err
	}
	return program, p.runtime.BindF32(ctx, program, "static:resident image denoiser", denoiser.keyBias, denoiser.keyData)
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
	if p.denoiser, err = p.bindDenoiser(ctx, nil); err != nil {
		return err
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
	// The session keeps the names its weights are shared by: the denoiser of
	// another prompt binds the same weights by them.
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

// Condition makes the conditioning of the prompt the pipeline was compiled
// for, from the rows of its tokens, and keeps it for the prompt met again.
func (p *ResidentImagePipeline) Condition(
	ctx context.Context,
	prompt string,
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
	conditioning, err := p.fuse(ctx, embedRows)
	if err != nil {
		return nil, err
	}
	if err := p.keep(ctx, prompt, conditioning); err != nil {
		return nil, err
	}
	if err := p.prepareGeneration(ctx); err != nil {
		_ = conditioning.outputs.Release(ctx)
		return nil, fmt.Errorf("resident image pipeline: prepare generation: %w", err)
	}
	return conditioning, p.retain(ctx, conditioning)
}

// retain retires the fusion once a prompt's conditioning is made and what it
// conditions is bound, and keeps the conditioning as the pipeline's.
func (p *ResidentImagePipeline) retain(ctx context.Context, conditioning *ResidentConditioning) error {
	if err := p.runtime.Release(ctx, p.fusion); err != nil {
		_ = conditioning.outputs.Release(ctx)
		return fmt.Errorf("resident image pipeline: retire fusion graph: %w", err)
	}
	p.fusion = nil
	p.residentBytes = p.runtime.StaticBytes()
	p.conditioning = conditioning
	return nil
}

// Recondition gives a session that has made a picture another prompt: the
// rows of its tokens and the three programs compiled under its mask. The
// prompt's conditioning replaces the one before it, placed from the host for
// a prompt the session keeps (recall), else made by the text encoder and the
// fusion placed on the device again and retired; the denoiser compiled for the
// prompt takes the weights the session holds, so none of them is loaded
// again. An error leaves the pipeline to be closed.
func (p *ResidentImagePipeline) Recondition(
	ctx context.Context,
	prompt string,
	encoder *EncoderProgram,
	fusion *FusionProgram,
	denoiser *DenoiserProgram,
	embedRows []float32,
) error {
	if p == nil || encoder == nil || fusion == nil || denoiser == nil {
		return errors.New("resident image pipeline: unavailable")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.runtime == nil || p.conditioning == nil || p.denoiser == nil || p.latent != nil {
		return errors.New("resident image pipeline: no finished generation to give another prompt")
	}
	if encoder.Seq != p.Encoder.Seq || fusion.TextSeq != p.Fusion.TextSeq || denoiser.TextSeq != p.Denoiser.TextSeq ||
		denoiser.GH != p.Denoiser.GH || denoiser.GW != p.Denoiser.GW {
		return errors.New("resident image pipeline: the prompt's programs are of another shape than the session's")
	}
	if err := p.conditioning.outputs.Release(ctx); err != nil {
		return fmt.Errorf("resident image pipeline: release conditioning: %w", err)
	}
	p.conditioning.outputs, p.conditioning.pipeline, p.conditioning = nil, nil, nil
	p.Encoder, p.Fusion, p.Denoiser = encoder, fusion, denoiser
	conditioning, err := p.recall(ctx, prompt, embedRows)
	if err != nil {
		return err
	}
	if p.denoiser, err = p.bindDenoiser(ctx, p.denoiser); err != nil {
		_ = conditioning.outputs.Release(ctx)
		return fmt.Errorf("resident image pipeline: another prompt's denoiser: %w", err)
	}
	return p.retain(ctx, conditioning)
}

// recall gives the pipeline's prompt its conditioning: the one the session
// keeps for it, placed from the host, else the one the text encoder and the
// fusion make, which the session then keeps.
func (p *ResidentImagePipeline) recall(ctx context.Context, prompt string, embedRows []float32) (*ResidentConditioning, error) {
	if fused, kept := p.conditioned.lookup(prompt); kept {
		outputs, err := p.runtime.Retain(ctx, p.conditioningUpload, map[*tensor.Tensor]reference.Value{
			p.conditioningInput: {Shape: p.conditioningInput.Shape, Data: fused},
		}, nil)
		if err != nil {
			return nil, fmt.Errorf("resident image pipeline: place kept conditioning: %w", err)
		}
		value, ok := outputs.Value(p.conditioningOutput)
		if !ok {
			return nil, errors.Join(errors.New("resident image pipeline: kept conditioning is unavailable"), outputs.Release(ctx))
		}
		return &ResidentConditioning{pipeline: p, outputs: outputs, value: value}, nil
	}
	if err := p.loadText(ctx); err != nil {
		return nil, err
	}
	conditioning, err := p.fuse(ctx, embedRows)
	if err != nil {
		return nil, err
	}
	return conditioning, p.keep(ctx, prompt, conditioning)
}

// keep holds a prompt's conditioning on the host for the prompt met again; an
// error releases the conditioning.
func (p *ResidentImagePipeline) keep(ctx context.Context, prompt string, conditioning *ResidentConditioning) error {
	fused, err := conditioning.outputs.CopyToHost(ctx, p.Fusion.Fused)
	if err != nil {
		return errors.Join(fmt.Errorf("resident image pipeline: keep conditioning: %w", err), conditioning.outputs.Release(ctx))
	}
	p.conditioned.store(prompt, fused.Data)
	return nil
}

// conditioningUploadGraph places a prompt's kept conditioning on the device as
// the fusion left it: a scale by one, which changes no value.
func conditioningUploadGraph(fusion *FusionProgram) (*tensor.Tensor, *tensor.Tensor, error) {
	builder := tensor.NewBuilder()
	input := builder.Input("kept_conditioning", dtype.F32, fusion.Fused.Shape)
	output := builder.Scale(input, tensor.UnitScale)
	if err := builder.Err(); err != nil {
		return nil, nil, fmt.Errorf("resident image pipeline: conditioning upload: %w", err)
	}
	return input, output, nil
}

// fuse runs the pipeline's prompt through the text encoder and the fusion,
// which loadText placed, and returns its conditioning retained on the device.
// The text encoder is retired when its hidden states are fused; the caller
// retires the fusion.
func (p *ResidentImagePipeline) fuse(ctx context.Context, embedRows []float32) (*ResidentConditioning, error) {
	embedValue, err := reference.BorrowedValue(p.Encoder.Embed.Shape, embedRows)
	if err != nil {
		return nil, fmt.Errorf("resident image pipeline: embed rows: %w", err)
	}
	selected, err := p.encodeBlocks(ctx, embedValue)
	if err != nil {
		return nil, fmt.Errorf("resident image pipeline: encode: %w", err)
	}
	encoded := func() error {
		var errs []error
		for _, rows := range selected {
			errs = append(errs, rows.outputs.Release(ctx))
		}
		return errors.Join(errs...)
	}
	for index, rows := range selected {
		value, ok := rows.outputs.Value(rows.node)
		if !ok {
			_ = encoded()
			return nil, fmt.Errorf("resident image pipeline: selected output %d is unavailable", index)
		}
		p.bridgePointers[index] = value.Pointer
	}
	bridged, err := p.runtime.Retain(ctx, p.bridge, nil, p.bridgePointers)
	if err != nil {
		_ = encoded()
		return nil, fmt.Errorf("resident image pipeline: bridge: %w", err)
	}
	if err := encoded(); err != nil {
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
	p.residentBytes = p.runtime.StaticBytes()
	return &ResidentConditioning{pipeline: p, outputs: fused, value: fusedValue}, nil
}

// encodedRows is a prompt's residual stream after a layer of the text
// encoder, kept on the device.
type encodedRows struct {
	outputs *executor.RetainedOutputs
	node    *tensor.Tensor
}

// encodeBlocks runs the prompt through the text encoder a layer at a time:
// each layer's weights are placed on the device, the prompt's rows pass
// through it and stay there for the next, and the layer is given back before
// the next is placed. It returns the rows after each selected layer, in the
// order of their slots, which the caller releases.
func (p *ResidentImagePipeline) encodeBlocks(ctx context.Context, embed reference.Value) (selected []encodedRows, err error) {
	slots, err := captureSlots(p.Encoder.E)
	if err != nil {
		return nil, err
	}
	selected = make([]encodedRows, len(p.Encoder.Selected))
	var rows encodedRows
	// kept says the rows are a selected layer's, which the caller is given.
	kept := func(rows encodedRows) bool {
		return slices.ContainsFunc(selected, func(held encodedRows) bool { return held.outputs == rows.outputs })
	}
	defer func() {
		if err == nil {
			return
		}
		if !kept(rows) {
			err = errors.Join(err, rows.outputs.Release(context.WithoutCancel(ctx)))
		}
		for _, held := range selected {
			err = errors.Join(err, held.outputs.Release(context.WithoutCancel(ctx)))
		}
	}()
	for layer := range p.Encoder.E.SelectLayers[len(p.Encoder.E.SelectLayers)-1] {
		if rows, err = p.encodeLayer(ctx, layer, embed, rows, kept(rows)); err != nil {
			return nil, err
		}
		if slot, ok := slots[layer+1]; ok {
			selected[slot] = rows
		}
	}
	return selected, nil
}

// encodeLayer passes the prompt's rows through one layer of the text encoder:
// from its embedding at the first, else from rows, which it releases unless
// kept, and returns the rows after the layer. The layer's weights are given
// back before it returns.
func (p *ResidentImagePipeline) encodeLayer(ctx context.Context, layer int, embed reference.Value, rows encodedRows, kept bool) (encodedRows, error) {
	block, err := p.Encoder.block(layer)
	if err != nil {
		return rows, err
	}
	feeds := map[*tensor.Tensor]reference.Value{}
	if block.keyBias != nil {
		feeds[block.keyBias] = reference.Value{Shape: block.keyBias.Shape, Data: p.Encoder.keyBiasData}
	}
	var dynamic []*tensor.Tensor
	var pointers []driver.DevicePtr
	if rows.outputs == nil {
		feeds[block.Hidden] = embed
	} else {
		value, ok := rows.outputs.Value(rows.node)
		if !ok {
			return rows, fmt.Errorf("resident image pipeline: the rows before layer %d are unavailable", layer)
		}
		dynamic, pointers = []*tensor.Tensor{block.Hidden}, []driver.DevicePtr{value.Pointer}
	}
	program, err := compileResidentWeights(
		ctx, p.runtime, "resident image encoder layer", filepath.Join(p.modelDir, "text_encoder"),
		block.weightInputs, dynamic, block.Output,
	)
	if err != nil {
		return rows, err
	}
	retained, err := p.runtime.Retain(ctx, program, feeds, pointers)
	if err = errors.Join(err, p.runtime.Release(ctx, program)); err != nil {
		return rows, errors.Join(err, retained.Release(ctx))
	}
	if !kept {
		err = rows.outputs.Release(ctx)
	}
	return encodedRows{outputs: retained, node: block.Output}, err
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
		_ = bridged.Release(context.WithoutCancel(ctx))
		return nil, 0, 0, errors.New("resident image pipeline: VAE latent is unavailable")
	}
	results, err := p.runtime.ExecuteDevice(ctx, p.vae, nil, []driver.DevicePtr{value.Pointer})
	releaseErr := bridged.Release(context.WithoutCancel(ctx))
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
	p.bridge, p.fusion, p.timestep, p.denoiser, p.upload, p.conditioningUpload = nil, nil, nil, nil, nil, nil
	p.conditioned = conditioningCache{}
	p.vaeBridge, p.vae = nil, nil
	return err
}
