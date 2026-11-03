//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/graphruntime"
	"overgo/internal/hfbpe"
	"overgo/internal/hfgguf"
	"overgo/internal/hfrepo"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"path/filepath"
	"slices"
)

type modelConditioner struct {
	source    *hfrepo.Repository
	catalog   hfgguf.LanguageCatalog
	root      string
	tokenizer *hfbpe.Tokenizer
	template  textTemplate
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
	catalog, err := hfgguf.InspectLanguageModel(source, hfgguf.LanguageHidden)
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
	return &modelConditioner{source: source, catalog: catalog, root: root, tokenizer: tok, template: template}, nil
}

func (c *modelConditioner) input(prompt string) (textInput, []float32, error) {
	text, err := renderTextInput(c.tokenizer, prompt, c.template)
	if err != nil {
		return text, nil, err
	}
	for _, binding := range c.catalog.Bindings() {
		if binding.Target == c.catalog.Weights.TokenEmbedding.Name {
			rows, err := safetensors.ReadTensorRowsF32(filepath.Join(c.root, "text_encoder"), binding.Source, int(c.catalog.Spec.EmbeddingLength), text.IDs)
			return text, rows, err
		}
	}
	return text, nil, fmt.Errorf("model conditioning has no source token embedding binding")
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
	p, err := compileModelEncoder(c.catalog, len(text.IDs), 0, int(c.catalog.Spec.BlockCount))
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
	program, err := p.resident(ctx, session, c.catalog, c.root, nil, outputs...)
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
	p, err := compileModelEncoder(c.catalog, len(text.IDs), block, block+1)
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
	program, err := p.resident(ctx, session, c.catalog, c.root, dynamic, p.output)
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

func (p *modelEncoderProgram) resident(ctx context.Context, session *graphruntime.ResidentSession, catalog hfgguf.LanguageCatalog, sourceIdentity string, dynamic []*tensor.Tensor, outputs ...*tensor.Tensor) (_ *graphruntime.ResidentProgram, err error) {
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
		if err = session.Bind(ctx, program, sourceIdentity, node.Name, node, func() ([]byte, error) {
			payload, err := catalog.Payload(node.Name)
			if err != nil {
				return nil, err
			}
			return io.ReadAll(payload.Data)
		}); err != nil {
			return nil, err
		}
	}
	return program, nil
}
