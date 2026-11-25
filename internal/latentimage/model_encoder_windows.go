//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"overgo/internal/comfyquant"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/graphruntime"
	"overgo/internal/hfbpe"
	"overgo/internal/hfgguf"
	"overgo/internal/hfrepo"
	"overgo/internal/model"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"path/filepath"
	"slices"
	"sync"
)

type modelConditioner struct {
	source    *hfrepo.Repository
	catalog   hfgguf.LanguageCatalog
	root      string
	tokenizer *hfbpe.Tokenizer
	template  textTemplate
	// stored is the encoder file's tensors as the file stores them, beside source's decoded view, where it holds int8
	// ConvRot weights (a ComfyUI encoder file), and decoded the catalog's weights each of those decodes to, by name: the
	// card decodes them from their int8 rows (convRot, moss-first-run).
	stored  *safetensors.Source
	decoded map[string]string
	// planning guards kept, the encoder's plan once made (plan), and plans, how many times it was made.
	planning sync.Mutex
	kept     *model.ModelPlan
	plans    int
}

// plan is the encoder's model-wide layer and cache plan, made the first time it is asked for and kept for the loaded
// encoder: each block of each prompt had planned all its layers again (SQA-511).
func (c *modelConditioner) plan() (model.ModelPlan, error) {
	c.planning.Lock()
	defer c.planning.Unlock()
	if c.kept == nil {
		plan, err := planModelEncoder(c.catalog)
		if err != nil {
			return model.ModelPlan{}, err
		}
		c.kept, c.plans = &plan, c.plans+1
	}
	return *c.kept, nil
}

// program builds the encoder's layers from first to end for a prompt of tokens from its kept plan.
func (c *modelConditioner) program(tokens, first, end int) (*modelEncoderProgram, error) {
	plan, err := c.plan()
	if err != nil {
		return nil, err
	}
	return compileModelEncoderFrom(plan, c.catalog, tokens, first, end)
}

func loadModelConditioner(root string, template textTemplate) (_ *modelConditioner, err error) {
	source, err := hfrepo.Open(filepath.Join(root, "text_encoder"))
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, source.Close())
		}
	}()
	// A ComfyUI encoder file is read as the Hugging Face language model it holds (comfyLanguageView), its states computed
	// as ComfyUI computes a text encoder's (hfgguf.LanguageHiddenComfyUI).
	output := hfgguf.LanguageHidden
	if comfyEncoderFile(source.Tensors) {
		output = hfgguf.LanguageHiddenComfyUI
	}
	source.Tensors = comfyLanguageView(source.Tensors)
	catalog, err := hfgguf.InspectLanguageModel(source, output)
	if err != nil {
		return nil, err
	}
	tok, err := hfbpe.Load(filepath.Join(root, "tokenizer"))
	if err != nil {
		return nil, err
	}
	text, err := os.ReadFile(filepath.Join(root, "tokenizer", "chat_template.jinja"))
	if err != nil {
		return nil, err
	}
	if template.ChatTemplate != "" && template.ChatTemplate != string(text) {
		return nil, fmt.Errorf("model conditioning template differs from profile")
	}
	template.ChatTemplate = string(text)
	conditioner := &modelConditioner{source: source, catalog: catalog, root: root, tokenizer: tok, template: template}
	if err := conditioner.openStored(); err != nil {
		return nil, err
	}
	return conditioner, nil
}

// openStored opens the encoder file's tensors as the file stores them where it holds int8 ConvRot weights, and names
// each catalog weight the catalog reads as the BF16 one of them decodes to, with nothing else done to it: no transform
// and no other storage. A file holding none opens nothing.
func (c *modelConditioner) openStored() error {
	stored, err := safetensors.OpenSource(filepath.Join(c.root, "text_encoder"))
	if err != nil {
		return err
	}
	view := comfyLanguageView(stored)
	decoded, err := cardDecoded(view, c.catalog.Bindings())
	if err != nil || len(decoded) == 0 {
		return errors.Join(err, stored.Close())
	}
	c.stored, c.decoded = view, decoded
	return nil
}

// cardDecoded names, by its target, each binding the card decodes from its int8 rows, with the source it reads: an
// untransformed BF16 binding whose source the stored view holds in int8 ConvRot rotated in groups the card's
// decode turns (executor.DecodesInt8ConvRotGroup), each read from the weight's comfy_quant record alone, so no payload
// is read for one the card cannot decode, which the catalog's host decode reads (decode-groups, SQA-455): comfyquant
// admits any power of 4, and a weight rotated in groups of 4096 had its 50 MB of rows read and its place reserved before
// the card's decode refused it, failing the prompt the host decode had served.
func cardDecoded(stored *safetensors.Source, bindings []hfgguf.TensorBinding) (map[string]string, error) {
	decoded := map[string]string{}
	for _, binding := range bindings {
		if len(binding.Transforms) != 0 || binding.TargetStorage != gguf.DTypeBF16.String() {
			continue
		}
		weight, quantized, err := comfyquant.Find(stored, binding.Source)
		if err != nil {
			return nil, err
		}
		if quantized && executor.DecodesInt8ConvRotGroup(weight.Group()) {
			decoded[binding.Target] = binding.Source
		}
	}
	return decoded, nil
}

// convRot is the int8 ConvRot weight the encoder's file holds for the catalog's weight named, where the catalog reads
// that weight as the BF16 it decodes to (openStored), and whether it does.
func (c *modelConditioner) convRot(name string) (comfyquant.Weight, bool, error) {
	source, decoded := c.decoded[name]
	if !decoded {
		return comfyquant.Weight{}, false, nil
	}
	return comfyquant.Find(c.stored, source)
}

// close closes the encoder's files, both views of them.
func (c *modelConditioner) close() error {
	err := c.source.Close()
	if c.stored != nil {
		err = errors.Join(err, c.stored.Close())
	}
	return err
}

func (c *modelConditioner) input(prompt string) (textInput, []float32, error) {
	text, err := renderTextInput(c.tokenizer, prompt, c.template)
	if err != nil {
		return text, nil, err
	}
	rows, err := c.embedding(text)
	return text, rows, err
}

// embedding is a rendered prompt's token embedding rows, as the text encoder reads them.
func (c *modelConditioner) embedding(text textInput) ([]float32, error) {
	for _, binding := range c.catalog.Bindings() {
		if binding.Target == c.catalog.Weights.TokenEmbedding.Name {
			tensor, held := c.source.Tensors.Tensors[binding.Source]
			if !held {
				return nil, fmt.Errorf("model conditioning: the encoder holds no token embedding %s", binding.Source)
			}
			return safetensors.TensorRowsF32(tensor, int(c.catalog.Spec.EmbeddingLength), text.IDs)
		}
	}
	return nil, fmt.Errorf("model conditioning has no source token embedding binding")
}

func (c *modelConditioner) encode(ctx context.Context, session *graphruntime.ResidentSession, text textInput, embedding []float32, observe func(string, []float32) error) ([]float32, error) {
	return c.encodeHolding(ctx, session, nil, text, embedding, observe)
}

// encodeHolding is encode for a caller with more prompts to encode: held takes
// the program, whose bound weights the next prompt's program shares, and the
// caller releases it. A nil held releases it here, and the weights with it.
func (c *modelConditioner) encodeHolding(ctx context.Context, session *graphruntime.ResidentSession, held *[]*graphruntime.ResidentProgram, text textInput, embedding []float32, observe func(string, []float32) error) (_ []float32, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if err := unpadded(text); err != nil {
		return nil, err
	}
	p, err := c.program(len(text.IDs), 0, int(c.catalog.Spec.BlockCount))
	if err != nil {
		return nil, err
	}
	names := slices.Sorted(maps.Keys(p.boundaries))
	outputs := []*tensor.Tensor{p.output}
	if observe != nil {
		outputs = nil
		for _, name := range names {
			outputs = append(outputs, p.boundaries[name])
		}
	}
	program, err := p.resident(ctx, session, c.catalog, c.root, c.convRot, nil, outputs...)
	if err != nil {
		return nil, err
	}
	if held != nil {
		*held = append(*held, program)
	} else {
		defer func() { err = errors.Join(err, session.Release(context.WithoutCancel(ctx), program)) }()
	}
	p.feeds[p.input] = reference.Value{Shape: p.input.Shape, Data: embedding}
	result, err := session.Execute(ctx, program, p.feeds)
	if err != nil {
		return nil, err
	}
	if observe != nil {
		for _, name := range names {
			if err := observe(name, result[p.boundaries[name]].Data); err != nil {
				return nil, err
			}
		}
	}
	return result[p.output].Data, nil
}

// unpadded refuses a prompt the encoder cannot read: it attends every token.
func unpadded(text textInput) error {
	if len(text.Mask) != len(text.IDs) {
		return fmt.Errorf("model conditioning requires one mask value per token")
	}
	if slices.Contains(text.Mask, false) {
		return fmt.Errorf("model conditioning requires an unpadded prompt")
	}
	return nil
}

// encodeEach encodes prompts through the text encoder a block at a time. Each
// block's weights are placed on the device, every prompt passes through it,
// its hidden rows kept on the device for the next block, and the block is
// given back before the next is placed: the device holds one block of the
// encoder and the prompts' rows, where a program of the whole encoder holds
// all of it. The hidden states are the whole program's, bit for bit.
func (c *modelConditioner) encodeEach(ctx context.Context, session *graphruntime.ResidentSession, texts []textInput, embeddings [][]float32) (_ [][]float32, err error) {
	if len(texts) != len(embeddings) {
		return nil, fmt.Errorf("model conditioning has %d prompts and %d embeddings", len(texts), len(embeddings))
	}
	for _, text := range texts {
		if err := unpadded(text); err != nil {
			return nil, err
		}
	}
	rows := make([]*executor.RetainedOutputs, len(texts))
	outputs := make([]*tensor.Tensor, len(texts))
	defer func() {
		for _, retained := range rows {
			err = errors.Join(err, retained.Release(context.WithoutCancel(ctx)))
		}
	}()
	for block := range int(c.catalog.Spec.BlockCount) {
		programs := make([]*graphruntime.ResidentProgram, 0, len(texts))
		for index, text := range texts {
			if err := c.encodeBlock(ctx, session, block, text, embeddings[index], &rows[index], &outputs[index], &programs); err != nil {
				return nil, errors.Join(err, session.Release(context.WithoutCancel(ctx), programs...))
			}
		}
		// The block's weights are given back before the next block's are placed.
		if err := session.Release(ctx, programs...); err != nil {
			return nil, err
		}
	}
	hidden := make([][]float32, len(texts))
	for index := range texts {
		value, err := rows[index].CopyToHost(ctx, outputs[index])
		if err != nil {
			return nil, err
		}
		hidden[index] = value.Data
	}
	return hidden, nil
}

// encodeBlock passes one prompt through one block of the encoder: from its
// embedding at the first block, and from the rows the block before kept on
// the device after it, which it replaces with its own. The block's program
// joins programs, which hold its weights until the caller releases them.
func (c *modelConditioner) encodeBlock(ctx context.Context, session *graphruntime.ResidentSession, block int, text textInput, embedding []float32, rows **executor.RetainedOutputs, output **tensor.Tensor, programs *[]*graphruntime.ResidentProgram) error {
	p, err := c.program(len(text.IDs), block, block+1)
	if err != nil {
		return err
	}
	var dynamic []*tensor.Tensor
	var pointers []driver.DevicePtr
	if *rows == nil {
		p.feeds[p.input] = reference.Value{Shape: p.input.Shape, Data: embedding}
	} else {
		previous, ok := (*rows).Value(*output)
		if !ok {
			return errors.New("model conditioning: the block before kept no rows")
		}
		dynamic, pointers = []*tensor.Tensor{p.input}, []driver.DevicePtr{previous.Pointer}
	}
	program, err := p.resident(ctx, session, c.catalog, c.root, c.convRot, dynamic, p.output)
	if err != nil {
		return err
	}
	*programs = append(*programs, program)
	retained, err := session.Retain(ctx, program, p.feeds, pointers)
	if err != nil {
		return err
	}
	// The memory the prompt ran in is given back before the next prompt runs: two prompts' at once would hold more
	// than the whole encoder's program, which runs one prompt at a time.
	err = errors.Join((*rows).Release(ctx), session.ReleaseStageMemory(ctx))
	*rows, *output = retained, p.output
	return err
}

// resident compiles the program and binds its weights from the catalog, under sourceIdentity: each one convRot names
// (modelConditioner.convRot) decoded on the card from the int8 ConvRot rows its file holds into F32, bit for bit as the
// host's float32 stages decode it (graphruntime.ResidentSession.BindDecoded, moss-first-run), the rest read from the
// catalog, a BF16 one widened where the program reads it in F32 (ComfyUI's arithmetic, compileModelEncoder). The MOSS
// Ultra encoder's weights had been decoded on the host, one core, as the first prompt's blocks were placed: 41 s of its
// 51 s of preparation.
func (p *modelEncoderProgram) resident(ctx context.Context, session *graphruntime.ResidentSession, catalog hfgguf.LanguageCatalog, sourceIdentity string, convRot func(name string) (comfyquant.Weight, bool, error), dynamic []*tensor.Tensor, outputs ...*tensor.Tensor) (_ *graphruntime.ResidentProgram, err error) {
	if len(outputs) == 0 {
		outputs = []*tensor.Tensor{p.output}
	}
	program, err := session.Compile(ctx, "model conditioning", dynamic, outputs...)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, session.Release(context.WithoutCancel(ctx), program))
		}
	}()
	for _, node := range p.weights {
		if _, required := program.Graph.InputSlot(node); !required {
			continue
		}
		weight, quantized := comfyquant.Weight{}, false
		if convRot != nil {
			if weight, quantized, err = convRot(node.Name); err != nil {
				return nil, err
			}
		}
		if quantized {
			err = session.BindDecoded(ctx, program, sourceIdentity, node.Name, node, func() (executor.Int8ConvRotDecode, error) {
				stored, err := weight.Stored()
				shape := weight.Shape()
				return executor.Int8ConvRotDecode{Stored: stored, Rows: uint32(shape[0]), Inner: uint32(shape[1]), Group: uint32(weight.Group())}, err
			})
		} else {
			err = session.Bind(ctx, program, sourceIdentity, node.Name, node, func() ([]byte, error) {
				payload, err := catalog.Payload(node.Name)
				if err != nil {
					return nil, err
				}
				data, err := io.ReadAll(payload.Data)
				if err != nil || payload.Type != gguf.DTypeBF16 || node.Type != dtype.F32 {
					return data, err
				}
				return widenBF16(data), nil
			})
		}
		if err != nil {
			return nil, err
		}
	}
	return program, nil
}
