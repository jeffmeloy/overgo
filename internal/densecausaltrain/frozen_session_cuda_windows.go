//go:build windows

package densecausaltrain

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/densecausal"
	"overgo/internal/devicemath"
	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
)

// FrozenSession runs a dense model on the device with every weight frozen and rows put in place of some tokens'
// embeddings, and gives the loss and its gradient with respect to those rows: a frozen text model's forward and
// backward for an adapter that feeds its input, as an image composite's bridge does. No step is taken, so the weights
// stay what the model holds.
type FrozenSession struct {
	worker    *device.Worker
	session   *devicemath.CausalTrainingSession
	vectors   []devicemath.LayerTrainVectors
	allocated []driver.DevicePtr
}

// OpenFrozenSession puts the model's weights on the device for sequences of length tokens.
func OpenFrozenSession(m *densecausal.Model, worker *device.Worker, length int) (*FrozenSession, error) {
	d := m.Dims
	if ok, reason := densecausal.DeviceTrainingSupported(d); !ok {
		return nil, fmt.Errorf("densecausal frozen session: %s", reason)
	}
	pack, err := m.TrainingPack()
	if err != nil {
		return nil, err
	}
	weights, _ := pack.Slabs()
	layout, err := newResidentLayout(d, pack.Plan())
	if err != nil {
		return nil, err
	}
	frozen := &FrozenSession{worker: worker, vectors: layout.vectorWeights(d, weights)}
	resident := func(count int, initial []float32) (driver.DevicePtr, error) {
		pointer, err := devicemath.AllocResidentF32(worker, count, initial)
		if err == nil {
			frozen.allocated = append(frozen.allocated, pointer)
		}
		return pointer, err
	}
	upload := func(name string) (driver.DevicePtr, error) {
		group := layout.group(name)
		return resident(group.End-group.Start, weights[group.Start:group.End])
	}
	fail := func(cause error) (*FrozenSession, error) { return nil, errors.Join(cause, frozen.Close()) }
	dW, err := resident(layout.layerEnd-layout.layerStart, weights[layout.layerStart:layout.layerEnd])
	if err != nil {
		return fail(err)
	}
	// The layer gradients a run writes are its scratch: no step reads them.
	dG, err := resident(layout.layerEnd-layout.layerStart, nil)
	if err != nil {
		return fail(err)
	}
	embeddingName, finalNormName, headName := m.LexicalTensorNames()
	embedding, err := upload(embeddingName)
	if err != nil {
		return fail(err)
	}
	head := embedding
	if headName != embeddingName {
		if head, err = upload(headName); err != nil {
			return fail(err)
		}
	}
	finalNorm, err := upload(finalNormName)
	if err != nil {
		return fail(err)
	}
	finalNormGrad, err := resident(d.Hidden, nil)
	if err != nil {
		return fail(err)
	}
	invFreq := dtype.Float64SliceToFloat32(hostmath.RopeInvFreq(d.RopeTheta, d.HeadDim))
	frozen.session, err = devicemath.NewCausalTrainingSession(
		worker, dW, dG, layout.offsets, invFreq,
		devicemath.CausalLexical{Embedding: embedding, Head: head, FinalNorm: finalNorm, FinalNormGrad: finalNormGrad},
		length, d.Vocab, d.Hidden, d.Heads, d.KVHeads, d.HeadDim, d.Intermediate, d.RMSEps,
	)
	if err != nil {
		return fail(err)
	}
	return frozen, nil
}

// Run is the mean loss over the trained tokens, a row predicting the token after it, with rows in place of the
// embeddings of the tokens at positions, and the loss's gradient with respect to those rows.
func (s *FrozenSession) Run(tokens []int, trained []bool, positions []int, rows []float32) (float64, []float32, error) {
	if len(trained) != len(tokens) || !slices.Contains(trained[min(1, len(trained)):], true) {
		return 0, nil, errors.New("densecausal frozen session: a trained flag for each token, and a token after the first trained")
	}
	// Row r predicts token r+1, so the first token is never predicted.
	predicted := trained[1:]
	var count float32
	for _, flag := range predicted {
		if flag {
			count++
		}
	}
	scales := make([]float32, len(predicted))
	for row, flag := range predicted {
		if flag {
			scales[row] = 1 / count
		}
	}
	return s.session.RunPlaced(tokens, scales, s.vectors, positions, rows)
}

// Close releases the session and the weights it put on the device.
func (s *FrozenSession) Close() error {
	if s == nil {
		return nil
	}
	return errors.Join(s.session.Close(), devicemath.FreeResident(s.worker, s.allocated...))
}
