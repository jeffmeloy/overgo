// Package tensorgrad derives a backward pass from a forward one. Given a
// graph of tensor operations it builds the graph of the gradients of a
// scalar by some of the graph's inputs, out of the operations that exist: a
// rule for each operation says what a gradient of its output gives each of
// its inputs. Nothing here executes; the graphs it builds run wherever a
// tensor program runs. The package is training's: nothing that serves a
// model imports it.
package tensorgrad

import (
	"fmt"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// rule gives, for the gradient of a node's output, the gradient it sends to
// each of the node's inputs, in their order. A nil gradient sends nothing: an
// input the operation does not differentiate by, as an index is.
type rule func(b *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error)

// rules are the operations a backward graph can be built through.
var rules = map[tensor.Op]rule{
	tensor.OpAdd:         same,
	tensor.OpMultiply:    product,
	tensor.OpScale:       scaled,
	tensor.OpMulMat:      projection,
	tensor.OpTranspose2D: transposed,
	tensor.OpReshape:     reshaped,
	tensor.OpSiLU:        gated,
	tensor.OpRMSNorm:     normalized,
	tensor.OpSoftmax:     shared,
	tensor.OpRoPENeoX:    rotated,
	tensor.OpRoPENormal:  rotated,
}

// Backward builds, in b, the gradients of a scalar by the tensors wanted,
// given seed, the scalar's gradient by output: the scalar is whatever seed
// says it is, the sum of output weighed by seed when seed is fed as a
// constant. The gradients come back in wanted's order, each of its tensor's
// shape. Only what lies between a wanted tensor and the output is
// differentiated; an operation on that path with no rule is refused by name,
// and so is a wanted tensor the output does not depend on. The constants are
// inputs the gradients read that the forward graph does not have, each with
// what it is always fed.
func Backward(b *tensor.Builder, output, seed *tensor.Tensor, wanted ...*tensor.Tensor) ([]*tensor.Tensor, []Constant, error) {
	if b == nil || output == nil || seed == nil || !seed.Shape.Equal(output.Shape) {
		return nil, nil, fmt.Errorf("tensorgrad: a backward graph starts from an output and a gradient of its shape")
	}
	var constants []Constant
	masks := map[uint64]*tensor.Tensor{}
	// mask is the causal mask of a sequence of so many tokens, one a length.
	mask := func(tokens uint64) *tensor.Tensor {
		if masks[tokens] == nil {
			constant := causalMask(b, tokens)
			masks[tokens], constants = constant.Tensor, append(constants, constant)
		}
		return masks[tokens]
	}
	var order []*tensor.Tensor
	seen := map[*tensor.Tensor]bool{}
	var visit func(node *tensor.Tensor)
	visit = func(node *tensor.Tensor) {
		if seen[node] {
			return
		}
		seen[node] = true
		for _, input := range node.Inputs {
			visit(input)
		}
		order = append(order, node)
	}
	visit(output)
	reaches := map[*tensor.Tensor]bool{}
	for _, node := range order {
		for _, target := range wanted {
			reaches[node] = reaches[node] || node == target
		}
		for _, input := range node.Inputs {
			reaches[node] = reaches[node] || reaches[input]
		}
	}
	gradients := map[*tensor.Tensor]*tensor.Tensor{output: seed}
	for index := len(order) - 1; index >= 0; index-- {
		node := order[index]
		gradient := gradients[node]
		if gradient == nil || len(node.Inputs) == 0 || !reaches[node] {
			continue
		}
		derive, known := rules[node.Op]
		if node.Op == tensor.OpAttention {
			derive, known = func(b *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
				return attended(b, node, gradient, mask)
			}, true
		}
		if !known {
			return nil, nil, fmt.Errorf("tensorgrad: operation %s has no derivative rule", node.Op)
		}
		sent, err := derive(b, node, gradient)
		if err != nil {
			return nil, nil, fmt.Errorf("tensorgrad: %s: %w", node.Op, err)
		}
		for at, input := range node.Inputs {
			if sent[at] == nil || !reaches[input] {
				continue
			}
			if held := gradients[input]; held != nil {
				gradients[input] = b.Add(held, sent[at])
			} else {
				gradients[input] = sent[at]
			}
		}
	}
	found := make([]*tensor.Tensor, len(wanted))
	for index, target := range wanted {
		if found[index] = gradients[target]; found[index] == nil {
			return nil, nil, fmt.Errorf("tensorgrad: the output does not depend on wanted tensor %d (%q)", index, target.Name)
		}
	}
	return found, constants, b.Err()
}

// Constant is an input of a backward graph that is no value of the forward
// graph, and what it is fed at every execution.
type Constant struct {
	Tensor *tensor.Tensor
	Data   []float32
}

// masked is what a causal mask adds to the score of a key after its query:
// far enough below any score that the key's share is nothing.
const masked = -1e30

// causalMask is the mask a causal attention over tokens tokens adds to its
// scores, a row a query and an entry a key: nothing for a key at or before
// its query, and masked for one after.
func causalMask(b *tensor.Builder, tokens uint64) Constant {
	shape, err := tensor.NewShape(tokens, tokens)
	if err != nil {
		return Constant{}
	}
	data := make([]float32, tokens*tokens)
	for query := range tokens {
		for key := query + 1; key < tokens; key++ {
			data[query*tokens+key] = masked
		}
	}
	return Constant{Tensor: b.Input(fmt.Sprintf("tensorgrad.causal_mask.%d", tokens), dtype.F32, shape), Data: data}
}

// attended is the rule of plain causal attention over one sequence, grouped
// or not: queries, keys and values of [width, heads, tokens], a query head
// attending the key and value head of its group. It is built a query head at
// a time from what the forward operation computes: the scores, a product of
// the head's keys and queries, scaled and masked; their softmax, the shares;
// the shares' gradient, a product of the values and the output's gradient,
// taken through the softmax's own rule; and from that the three gradients,
// each a product, a group's keys and values summing over its query heads. An
// attention with a window, a bias, a cap or a cache is refused.
func attended(b *tensor.Builder, node, gradient *tensor.Tensor, mask func(tokens uint64) *tensor.Tensor) ([]*tensor.Tensor, error) {
	attributes, held := node.Attrs.(tensor.AttentionAttributes)
	if !held || len(node.Inputs) != attentionInputs {
		return nil, fmt.Errorf("an attention with a bias, sinks or a mask of its own")
	}
	query, key, value := node.Inputs[0], node.Inputs[1], node.Inputs[attentionInputs-1]
	if !attributes.Causal || attributes.Softcap != 0 || attributes.MaxALiBiBias != 0 || attributes.Window != 0 || attributes.QueryStart != 0 ||
		attributes.KeyValueTokens != 0 || query.Shape.Rank != attentionInputs || query.Shape.Dims[attentionInputs-1] != key.Shape.Dims[attentionInputs-1] {
		return nil, fmt.Errorf("an attention that is not plain causal attention over one whole sequence")
	}
	width, heads, tokens := query.Shape.Dims[0], query.Shape.Dims[1], query.Shape.Dims[attentionInputs-1]
	valueWidth, groups := value.Shape.Dims[0], key.Shape.Dims[1]
	// head is one head's rows, [width, tokens], of a tensor laid a token, then a head, then a width.
	head := func(whole *tensor.Tensor, index, width uint64) *tensor.Tensor {
		flat := b.Reshape(whole, width*whole.Shape.Dims[1], tokens)
		return b.Reshape(b.GroupSlice(flat, index*width, width, 1, width), width, tokens)
	}
	joined := func(parts []*tensor.Tensor, width uint64) *tensor.Tensor {
		whole := parts[0]
		for _, part := range parts[1:] {
			whole = b.Concat(whole, part, 0)
		}
		return b.Reshape(whole, width, uint64(len(parts)), tokens)
	}
	product := func(left, right *tensor.Tensor) *tensor.Tensor {
		return b.MulMatWithCompute(left, right, tensor.MulMatComputeExact)
	}
	queries, keys, values := make([]*tensor.Tensor, heads), make([]*tensor.Tensor, groups), make([]*tensor.Tensor, groups)
	for index := range heads {
		group := index / (heads / groups)
		ownQuery, ownKey, ownValue, ownGradient := head(query, index, width), head(key, group, width), head(value, group, valueWidth), head(gradient, index, valueWidth)
		shares := b.Softmax(b.Add(b.Scale(product(ownKey, ownQuery), attributes.Scale), mask(tokens)))
		through, err := shared(b, shares, product(ownValue, ownGradient))
		if err != nil {
			return nil, err
		}
		scores := b.Scale(through[0], attributes.Scale)
		queries[index] = product(b.Transpose2D(ownKey), scores)
		toKey, toValue := product(b.Transpose2D(ownQuery), b.Transpose2D(scores)), product(b.Transpose2D(ownGradient), b.Transpose2D(shares))
		if keys[group] != nil {
			toKey, toValue = b.Add(keys[group], toKey), b.Add(values[group], toValue)
		}
		keys[group], values[group] = toKey, toValue
	}
	return []*tensor.Tensor{joined(queries, width), joined(keys, width), joined(values, valueWidth)}, nil
}

// attentionInputs is how many inputs a plain attention has, its queries, keys
// and values, and the rank of each: a width, a head and a token.
const attentionInputs = 3

// same is the rule of a sum: each term receives the gradient whole. A term
// that was broadcast, as a projection's bias is over its rows, receives
// nothing: it would have to gather the gradient, and what is trained through
// here is the other term.
func same(_ *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
	sent := make([]*tensor.Tensor, len(node.Inputs))
	for index, input := range node.Inputs {
		if input.Shape.Equal(node.Shape) {
			sent[index] = gradient
		}
	}
	return sent, nil
}

// product is the rule of an elementwise product: each factor receives the
// gradient times the other. A factor that was broadcast, as a normalization's
// weight is over the rows it scales, receives nothing: it would have to
// gather the gradient, and what is trained through here is the other factor.
func product(b *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
	sent := make([]*tensor.Tensor, len(node.Inputs))
	for index, input := range node.Inputs {
		if input.Shape.Equal(node.Shape) {
			sent[index] = b.Multiply(gradient, node.Inputs[len(node.Inputs)-1-index])
		}
	}
	return sent, nil
}

// scaled is the rule of a multiple: the gradient times the same factor.
func scaled(b *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
	attributes, held := node.Attrs.(tensor.ScaleAttributes)
	if !held {
		return nil, fmt.Errorf("a scale without its factor")
	}
	return []*tensor.Tensor{b.Scale(gradient, attributes.Value)}, nil
}

// projection is the rule of a matrix product, which takes a weight of
// [in, out] and rows of [in, rows] to [out, rows]: the rows receive the
// gradient through the weight's transpose, and the weight the product of the
// rows' and the gradient's transposes.
func projection(b *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
	weight, rows := node.Inputs[0], node.Inputs[1]
	return []*tensor.Tensor{
		b.MulMatWithCompute(b.Transpose2D(rows), b.Transpose2D(gradient), tensor.MulMatComputeExact),
		b.MulMatWithCompute(b.Transpose2D(weight), gradient, tensor.MulMatComputeExact),
	}, nil
}

// transposed is the rule of a transpose: the gradient's transpose.
func transposed(b *tensor.Builder, _, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
	return []*tensor.Tensor{b.Transpose2D(gradient)}, nil
}

// reshaped is the rule of a reshape: the gradient in the input's shape.
func reshaped(b *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
	return []*tensor.Tensor{b.Reshape(gradient, node.Inputs[0].Shape.Slice()...)}, nil
}

// gated is the rule of SiLU, x times its sigmoid s: the gradient times
// s + y - y s, which is the derivative written with the output y so that it
// needs no constant.
func gated(b *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
	sigmoid := b.Sigmoid(node.Inputs[0])
	slope := b.Add(b.Add(sigmoid, node), b.Scale(b.Multiply(node, sigmoid), -1))
	return []*tensor.Tensor{b.Multiply(gradient, slope)}, nil
}

// rotated is the rule of rotary position encoding, which turns each pair of
// channels by an angle its position gives: a rotation's transpose is the
// rotation back, so the gradient is turned by the same operation reversed.
// An encoding whose angles are stretched for a longer context, or scaled by
// a tensor of factors, is refused: its reverse is not built here.
func rotated(b *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
	attributes, held := node.Attrs.(tensor.RoPEAttributes)
	if !held || len(node.Inputs) != 1 || attributes.OriginalContext != 0 || attributes.ExtFactor != 0 {
		return nil, fmt.Errorf("a rotary encoding with stretched or scaled angles")
	}
	layout := tensor.RoPELayoutNormal
	if node.Op == tensor.OpRoPENeoX {
		layout = tensor.RoPELayoutNeoX
	}
	return []*tensor.Tensor{b.RoPEWithOptions(gradient, tensor.RoPEOptions{
		Layout: layout, Positions: attributes.Positions, RotaryDimensions: attributes.RotaryDimensions,
		FrequencyBase: attributes.FrequencyBase, FrequencyScale: attributes.FrequencyScale, Reverse: true,
	})}, nil
}

// shared is the rule of a softmax over a row, y: a row's gradient g becomes
// y (g - sum(g y)), what is left of g once the share every entry of the row
// gives up to the others is taken from it.
func shared(b *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
	taken := b.Scale(b.SumRows(b.Multiply(gradient, node)), -1)
	return []*tensor.Tensor{b.Multiply(node, b.Add(gradient, taken))}, nil
}

// normalized is the rule of RMS normalization, y = x r with r one over the
// root of a row's mean square and its epsilon: a row's gradient g becomes
// r (g - y mean(g y)). r is taken from the operation's own output, as the
// row sum of y y over the row sum of y x, which is r whatever the epsilon.
func normalized(b *tensor.Builder, node, gradient *tensor.Tensor) ([]*tensor.Tensor, error) {
	input := node.Inputs[0]
	width := input.Shape.Dims[0]
	if width == 0 {
		return nil, fmt.Errorf("a normalization of no width")
	}
	rate := b.Divide(b.SumRows(b.Multiply(node, node)), b.SumRows(b.Multiply(node, input)))
	mean := b.Scale(b.SumRows(b.Multiply(gradient, node)), -1/float32(width))
	return []*tensor.Tensor{b.Multiply(b.Add(gradient, b.Multiply(node, mean)), rate)}, nil
}
