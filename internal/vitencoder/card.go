package vitencoder

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// Card is an encoder whose blocks and final norm run on a CUDA device, every weight resident there in F32. The host
// decodes and prepares each picture and embeds its patches, as the encoder does, and the card gives the tokens Tokens
// gives.
type Card struct {
	encoder     *Encoder
	worker      *device.Worker
	executor    *executor.Executor
	compiled    *executor.CompiledGraph
	inputs      *executor.DeviceInputs
	allocations device.AllocationSet
	// embedded is the input rows the host feeds, and tokens the rows after the final norm, the hidden width by the
	// tokens, as the host lays them token by token.
	embedded, tokens *tensor.Tensor
}

// OpenCard puts the encoder's blocks and final norm on the device of ordinal.
func (e *Encoder) OpenCard(ctx context.Context, ordinal int) (*Card, error) {
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	card := &Card{encoder: e, worker: worker, allocations: device.NewAllocationSet(worker)}
	fail := func(cause error) (*Card, error) { return nil, errors.Join(cause, card.Close()) }
	if card.executor, err = executor.NewWithWorker(worker); err != nil {
		return fail(err)
	}
	g := e.geometry
	hidden, count, heads := uint64(g.Hidden), uint64(e.tokens), uint64(g.Heads)
	width, epsilon := hidden/heads, float32(g.Epsilon)
	b := tensor.NewBuilder()
	card.embedded = b.Input("embedded", dtype.F32, tensor.MustShape(hidden, count))
	weights := map[*tensor.Tensor][]float32{}
	// A weight is fed as the checkpoint holds it: a matrix [out, in] row-major is the graph's [in, out].
	weight := func(name string, values []float32, shape ...uint64) *tensor.Tensor {
		node := b.Input(name, dtype.F32, tensor.MustShape(shape...))
		weights[node] = values
		return node
	}
	linear := func(input *tensor.Tensor, name string, matrix, bias []float32, in, out uint64) *tensor.Tensor {
		return b.Add(b.MulMat(weight(name+".weight", matrix, in, out), input), weight(name+".bias", bias, out))
	}
	residual := func(stream, branch *tensor.Tensor, name string, scale []float32) *tensor.Tensor {
		if scale != nil {
			branch = b.Multiply(branch, weight(name, scale, hidden))
		}
		return b.Add(stream, branch)
	}
	stream := card.embedded
	for index, l := range e.layers {
		name := func(part string) string { return fmt.Sprintf("layer%d.%s", index, part) }
		normed := b.AffineLayerNorm(stream, weight(name("norm1.weight"), l.norm1Weight, hidden), weight(name("norm1.bias"), l.norm1Bias, hidden), epsilon)
		headed := func(part string, matrix, bias []float32) *tensor.Tensor {
			return b.Reshape(linear(normed, name(part), matrix, bias, hidden, hidden), width, heads, count)
		}
		attended := b.AttentionWithOptions(headed("query", l.query, l.queryBias), headed("key", l.key, l.keyBias), headed("value", l.value, l.valueBias),
			tensor.AttentionOptions{Scale: float32(1 / math.Sqrt(float64(width)))})
		stream = residual(stream, linear(b.Reshape(attended, hidden, count), name("output"), l.output, l.outputBias, hidden, hidden), name("scale1"), l.scale1)
		normed = b.AffineLayerNorm(stream, weight(name("norm2.weight"), l.norm2Weight, hidden), weight(name("norm2.bias"), l.norm2Bias, hidden), epsilon)
		inner := uint64(l.intermediate)
		var activated *tensor.Tensor
		if l.up != nil {
			activated = b.GELUErf(linear(normed, name("up"), l.up, l.upBias, hidden, inner))
		} else {
			// The gated feed-forward's fused input holds the gate, then the value.
			fused := linear(normed, name("gate_up"), l.gateUp, l.gateUpBias, hidden, tensor.PairedExtent*inner)
			half := func(offset uint64) *tensor.Tensor {
				return b.Reshape(b.GroupSlice(fused, offset, inner, tensor.SingletonExtent, tensor.PairedExtent*inner), inner, count)
			}
			activated = b.SwiGLU(half(0), half(inner))
		}
		stream = residual(stream, linear(activated, name("down"), l.down, l.downBias, inner, hidden), name("scale2"), l.scale2)
	}
	card.tokens = b.AffineLayerNorm(stream, weight("final.weight", e.finalWeight, hidden), weight("final.bias", e.finalBias, hidden), epsilon)
	if err := b.Err(); err != nil {
		return fail(fmt.Errorf("vitencoder card: %w", err))
	}
	if card.compiled, err = executor.Compile(card.tokens); err != nil {
		return fail(fmt.Errorf("vitencoder card: %w", err))
	}
	card.inputs = card.compiled.NewDeviceInputs()
	for node, values := range weights {
		pointer, err := card.allocations.Upload(ctx, driver.Bytes(values))
		if err == nil {
			err = card.inputs.Set(node, pointer)
		}
		if err != nil {
			return fail(err)
		}
	}
	return card, nil
}

// Tokens decodes one PNG or JPEG image and returns every token after the final norm, as the encoder's Tokens does.
func (c *Card) Tokens(ctx context.Context, image []byte) ([]float32, error) {
	pixels, err := c.encoder.imagePixels(image)
	if err != nil {
		return nil, err
	}
	embedded, err := c.encoder.embed(pixels)
	if err != nil {
		return nil, err
	}
	out, err := c.executor.ExecuteCompiled(ctx, c.compiled, map[*tensor.Tensor]reference.Value{c.embedded: {Shape: c.embedded.Shape, Data: embedded}}, c.inputs)
	if err != nil {
		return nil, fmt.Errorf("vitencoder card: %w", err)
	}
	return finite(out[c.tokens].Data)
}

// Embed is the picture's class token after the final norm, the pooled image embedding the encoder's Embed gives.
func (c *Card) Embed(ctx context.Context, image []byte) ([]float32, error) {
	tokens, err := c.Tokens(ctx, image)
	if err != nil {
		return nil, err
	}
	hidden := c.encoder.geometry.Hidden
	return tokens[:hidden:hidden], nil
}

// Shape is the encoder's: the side of its square input, the side of its patch grid and the width of each token.
func (c *Card) Shape() (input, grid, hidden int) { return c.encoder.Shape() }

// Close releases the card's weights, its executor and its worker.
func (c *Card) Close() error {
	var closing error
	if c.executor != nil {
		closing = c.executor.Close()
	}
	return errors.Join(closing, c.allocations.Close(context.Background()), c.worker.Close())
}
