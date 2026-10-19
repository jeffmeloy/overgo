//go:build windows

package latentimage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
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

func (c *modelConditioner) encode(ctx context.Context, session *graphruntime.ResidentSession, text textInput, embedding []float32, observe func(string, []float32) error) (_ []float32, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if len(text.Mask) != len(text.IDs) {
		return nil, fmt.Errorf("model conditioning requires one mask value per token")
	}
	for _, attended := range text.Mask {
		if !attended {
			return nil, fmt.Errorf("model conditioning requires an unpadded prompt")
		}
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
	program, err := p.resident(ctx, session, c.catalog, c.root, outputs...)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, session.Release(context.WithoutCancel(ctx), program)) }()
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

func (p *modelEncoderProgram) resident(ctx context.Context, session *graphruntime.ResidentSession, catalog hfgguf.LanguageCatalog, sourceIdentity string, outputs ...*tensor.Tensor) (_ *graphruntime.ResidentProgram, err error) {
	if len(outputs) == 0 {
		outputs = []*tensor.Tensor{p.output}
	}
	program, err := session.Compile(ctx, "model conditioning", nil, outputs...)
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
