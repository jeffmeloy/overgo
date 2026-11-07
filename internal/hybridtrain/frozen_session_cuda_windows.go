//go:build windows

package hybridtrain

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
)

// FrozenSession walks a hybrid language model on the card with every weight frozen and rows put in place of some
// tokens' embeddings, and gives the mean next-token loss over the trained tokens and its gradient with respect to
// those rows: the frozen text model an image composite's bridge trains through. The layers and the head stay on the
// card in F32; the final norm of the rows that predict a trained token is taken on the host.
type FrozenSession struct {
	model  *FrozenLanguageModel
	ops    *devicemath.ResidentOpsSession
	layers []frozenLayer
	length int
	// input and outputGradient hold a sequence's rows; scored, logits and targets the rows that reach the head.
	input, outputGradient, head, scored, logits, targets, losses driver.DevicePtr
}

// frozenLayer is a decoder layer on the card that walks back to its input alone.
type frozenLayer interface {
	Forward(driver.DevicePtr) (driver.DevicePtr, error)
	Backward(driver.DevicePtr) (driver.DevicePtr, error)
	Close() error
}

// float32Bytes is the bytes of one value a buffer on the card holds.
const float32Bytes = 4

// OpenFrozenSession puts the model's layers and head on the card for sequences of length tokens.
func OpenFrozenSession(ctx context.Context, model *FrozenLanguageModel, worker *device.Worker, length int) (*FrozenSession, error) {
	if model == nil || length != model.Geometry.Tokens {
		return nil, errors.New("hybridtrain frozen session: a model read for sequences of the session's length")
	}
	ops, err := devicemath.NewResidentOpsSession(worker)
	if err != nil {
		return nil, err
	}
	session := &FrozenSession{model: model, ops: ops, length: length}
	fail := func(cause error) (*FrozenSession, error) { return nil, errors.Join(cause, session.Close()) }
	geometry := model.Geometry
	for index, kind := range geometry.Types {
		dims := geometry.layerDims(kind)
		var layer frozenLayer
		if kind == LinearAttention {
			layer, err = ops.OpenFrozenGatedDeltaLayer(model.Layers[index], dims)
		} else {
			layer, err = ops.OpenFrozenGatedAttentionLayer(model.Layers[index], dims)
		}
		if err != nil {
			return fail(fmt.Errorf("hybridtrain frozen session: layer %d: %w", index, err))
		}
		session.layers = append(session.layers, layer)
	}
	hidden, vocab := geometry.Hidden, model.Vocab
	if session.head, err = ops.Upload(ctx, driver.Bytes(model.Head)); err != nil {
		return fail(err)
	}
	for _, buffer := range []struct {
		pointer *driver.DevicePtr
		count   int
	}{
		{&session.input, length * hidden}, {&session.outputGradient, length * hidden}, {&session.scored, length * hidden},
		{&session.logits, length * vocab}, {&session.targets, length}, {&session.losses, length},
	} {
		if *buffer.pointer, err = ops.Allocate(ctx, uint64(buffer.count*float32Bytes)); err != nil {
			return fail(err)
		}
	}
	return session, nil
}

// Run is the mean loss over the trained tokens, a row predicting the token after it, with rows in place of the
// embeddings of the tokens at positions, and the loss's gradient with respect to those rows.
func (s *FrozenSession) Run(ctx context.Context, tokens []int, trained []bool, positions []int, rows []float32) (float64, []float32, error) {
	hidden, length, eps := s.model.Geometry.Hidden, s.length, s.model.Geometry.Eps
	if len(tokens) != length || len(trained) != length || len(rows) != len(positions)*hidden {
		return 0, nil, fmt.Errorf("hybridtrain frozen session: %d tokens and %d flags for %d, and %d values for %d rows", len(tokens), len(trained), length, len(rows), len(positions))
	}
	ids := make([]uint32, length)
	for at, token := range tokens {
		if token < 0 || token >= s.model.Vocab {
			return 0, nil, fmt.Errorf("hybridtrain frozen session: token %d outside the vocabulary of %d", token, s.model.Vocab)
		}
		ids[at] = uint32(token)
	}
	for _, position := range positions {
		if position < 0 || position >= length {
			return 0, nil, errors.New("hybridtrain frozen session: a placed row stands outside the tokens")
		}
	}
	// Row r predicts token r+1, so a trained token's row is the one before it.
	var predictors []int
	var targets []uint32
	for at := 1; at < length; at++ {
		if trained[at] {
			predictors, targets = append(predictors, at-1), append(targets, ids[at])
		}
	}
	if len(predictors) == 0 {
		return 0, nil, errors.New("hybridtrain frozen session: no token after the first is trained")
	}
	x, err := s.model.TokenRows(ctx, ids)
	if err != nil {
		return 0, nil, err
	}
	hostmath.ScatterRows(x, rows, positions, hidden)
	if err := s.ops.Run(func(ops *devicemath.ResidentOps) error { return ops.Write(s.input, driver.Bytes(x)) }); err != nil {
		return 0, nil, err
	}
	output := s.input
	for _, layer := range s.layers {
		if output, err = layer.Forward(output); err != nil {
			return 0, nil, err
		}
	}
	out := make([]float32, length*hidden)
	if err := s.ops.Run(func(ops *devicemath.ResidentOps) error { return ops.DownloadF32(output, out) }); err != nil {
		return 0, nil, err
	}
	count, vocab := len(predictors), s.model.Vocab
	reached := make([]float32, count*hidden)
	hostmath.GatherRows(reached, out, predictors, hidden)
	normed := make([]float32, count*hidden)
	hostmath.RMSNormInto(normed, reached, s.model.FinalNorm, count, hidden, eps)
	losses, dNormed := make([]float32, count), make([]float32, count*hidden)
	err = s.ops.Run(func(ops *devicemath.ResidentOps) error {
		if err := errors.Join(ops.Write(s.scored, driver.Bytes(normed)), ops.Write(s.targets, driver.Bytes(targets))); err != nil {
			return err
		}
		if err := ops.GEMM(false, true, count, hidden, vocab, s.scored, s.head, s.logits); err != nil {
			return err
		}
		// The cross-entropy leaves each row's gradient in its logits, scaled to the mean over the rows.
		if err := ops.SoftmaxCrossEntropy(s.logits, s.targets, s.losses, count, vocab); err != nil {
			return err
		}
		if err := ops.GEMM(false, false, count, vocab, hidden, s.logits, s.head, s.scored); err != nil {
			return err
		}
		return errors.Join(ops.DownloadF32(s.losses, losses), ops.DownloadF32(s.scored, dNormed))
	})
	if err != nil {
		return 0, nil, err
	}
	var loss float64
	for _, value := range losses {
		loss += float64(value)
	}
	dReached := make([]float32, count*hidden)
	hostmath.RMSNormBackward(dReached, make([]float32, hidden), reached, s.model.FinalNorm, dNormed, count, hidden, eps, false)
	dOut := make([]float32, length*hidden)
	hostmath.ScatterRows(dOut, dReached, predictors, hidden)
	if err := s.ops.Run(func(ops *devicemath.ResidentOps) error { return ops.Write(s.outputGradient, driver.Bytes(dOut)) }); err != nil {
		return 0, nil, err
	}
	gradient := s.outputGradient
	for index := len(s.layers) - 1; index >= 0; index-- {
		if gradient, err = s.layers[index].Backward(gradient); err != nil {
			return 0, nil, err
		}
	}
	dX := make([]float32, length*hidden)
	if err := s.ops.Run(func(ops *devicemath.ResidentOps) error { return ops.DownloadF32(gradient, dX) }); err != nil {
		return 0, nil, err
	}
	rowGradient := make([]float32, len(rows))
	hostmath.GatherRows(rowGradient, dX, positions, hidden)
	return loss, rowGradient, nil
}

// Close releases the layers and the session's buffers.
func (s *FrozenSession) Close() error {
	if s == nil {
		return nil
	}
	var errs []error
	for _, layer := range s.layers {
		errs = append(errs, layer.Close())
	}
	return errors.Join(append(errs, s.ops.Close())...)
}
