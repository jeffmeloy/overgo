package executor

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
)

// The batched pass of the drafter's prompt K/V (Prefill::draft_kv, E-9): the rows a batch takes, a multiple of 64 as
// Strata's B, whose buffers take about 36 MB of the card beside the drafter's own (8,192 prompt rows in 12.3 ms, 9.7 ms
// at 1,024 a batch in 142 MB, 153 ms six rows a launch); and the fewest rows a prefill takes it for, Strata's smallest
// batch, the six-row pass (E-4) taking the rest, a window's rows among them.
const (
	strataDraftBatchRows    = 256
	strataDraftBatchMinimum = 64
)

// strataDraftBatch is the batched pass's buffers, carved from one allocation made at its first use: a batch's rows of
// the token embedding, its fc product, the normed hidden rows (then the gate of the hyper-connection's mix), the residual
// streams, their row scales and BF16 image, the low-rank product and its BF16 SiLU, the mixed input, and its K and V.
type strataDraftBatch struct {
	arena                                          driver.DevicePtr
	emb, e2, hn, residual, rs, xn, lo, lo16, mixed driver.DevicePtr
	k, v                                           driver.DevicePtr
}

// carve lays the buffers out from b's base for rows rows.
func (s *strataDraftBatch) carve(b *strataRoundBump, rows uint64) {
	n, hcn := uint64(strataGREmbedding), uint64(strataGRStreams*strataGREmbedding)
	kv := uint64(tensor.QSAKeyValueHeads * tensor.QSAHeadWidth)
	s.emb, s.e2 = b.take(rows*n*strataRoundWordBytes), b.take(rows*n*strataRoundWordBytes)
	s.hn, s.residual = b.take(rows*hcn*strataRoundWordBytes), b.take(rows*hcn*strataRoundWordBytes)
	s.rs, s.xn = b.take(rows*strataGRStreams*strataRoundWordBytes), b.take(rows*hcn*bf16ScalarBytes)
	s.lo, s.lo16 = b.take(rows*strataGRRank*strataRoundWordBytes), b.take(rows*strataGRRank*bf16ScalarBytes)
	s.mixed = b.take(rows * n * strataRoundWordBytes)
	s.k, s.v = b.take(rows*kv*strataRoundWordBytes), b.take(rows*kv*strataRoundWordBytes)
}

// draftBatched reports whether the drafter's prompt K/V takes the batched pass: its fc, K and V projections in Q8_0,
// which the staged products multiply, and its attention hyper-connection's down and up projections in BF16.
func (d *StrataDrafter) draftBatched() bool {
	s := d.spec
	for _, weight := range []StrataRoundWeight{s.FCEmbedding, s.FCHidden, s.Key, s.Value} {
		if weight.Type != strataGGMLQ8_0 || weight.Pointer == 0 {
			return false
		}
	}
	return s.Attention.Norm != 0 && s.Attention.Down != 0 && s.Attention.Up != 0
}

// prefillStrataDrafterBatched is Prefill::draft_kv on the executor's stream: the n prompt cells from cell0 that the
// window can still reach, a batch of rows at a time, their token embeddings and final residual rows (rows) through the
// fc projections, the attention hyper-connection's read and the K and V projections as batched products, the K normed
// and rotated, and both appended to the drafter's K/V as its own pass appends them. The records
// (strataDrafterPrefillRecords) are the tokens, each row's step record and each K row's position, uploaded at once. The
// products round otherwise than the six-row pass's integer dot products, so the drafts may differ, the target's tokens
// never.
func (e *Executor) prefillStrataDrafterBatched(state *device.State, d *StrataDrafter, rows driver.DevicePtr, n, cell0, firstNeeded int64) error {
	s := d.spec
	nn, hc, hcn := int64(strataGREmbedding), int64(strataGRStreams), int64(strataGRStreams*strataGREmbedding)
	nkv, hd := int64(tensor.QSAKeyValueHeads), int64(tensor.QSAHeadWidth)
	kv := nkv * hd
	batch := int64(strataDraftBatchRows)
	// the staging holds a batch's widest activation, every stream's hidden rows, and a whole fc matrix's rows
	staging := uint64(batch*hc*nn)*bf16ScalarBytes + uint64(nn*nn)*bf16ScalarBytes
	resources, err := e.ensureResources(state, true, 0, 0, staging)
	if err != nil {
		return err
	}
	if d.batch.arena == 0 {
		var count strataRoundBump
		d.batch.carve(&count, uint64(batch))
		if d.batch.arena, err = d.resources.allocate(state, count.used); err != nil {
			return fmt.Errorf("mtp prefill: the batched pass's buffers do not fit: %w", err)
		}
		d.batch.carve(&strataRoundBump{base: d.batch.arena}, uint64(batch))
	}
	functions, blas, b := resources.functions, resources.blas, &d.batch
	tok := d.prefillRecords
	step := at(tok, n)
	pos := at(step, n*strataDrafterStepWords)
	geometry := strataKVGeometry{kvHeads: tensor.QSAKeyValueHeads, headDim: tensor.QSAHeadWidth, pageSize: tensor.QSABlockCells}
	project := func(weights StrataRoundWeight, input, output driver.DevicePtr, outputs, inputRows int64) error {
		return launchStagedQuantMatMul(state, functions, blas, kernelDequantRowsQ80F16, weights.Pointer, input, output,
			uint32(nn), uint32(outputs), uint32(inputRows))
	}
	// y[count, outputs] = x[count, inner] . w^T for a BF16 matrix w of outputs rows
	bf16 := func(weights, input, output driver.DevicePtr, count, outputs, inner int64) error {
		return blas.library.GEMMEx(blas.handle, cublas.OperationTranspose, cublas.OperationNone,
			int32(outputs), int32(count), int32(inner), gemmProductScale, weights, cublas.DataBF16, int32(inner),
			input, cublas.DataBF16, int32(inner), gemmAccumulatorScale, output, cublas.DataF32, int32(outputs),
			cublas.ComputeF32, cublas.GemmDefault)
	}
	eps := float32(strataRoundEpsilon)
	for b0 := max(0, firstNeeded-cell0); b0 < n; b0 += batch {
		nb := min(batch, n-b0)
		// the input branches: the next token's embedding, and this cell's final residual rows
		launches := []func() error{
			func() error {
				return launchStrataEmbedRowsQ80(state, functions, s.Embedding, s.EmbeddingRow, at(tok, b0), int32(nb), nn, b.emb)
			},
			func() error {
				return launchStrataNativeQSANorm(state, functions, b.emb, s.EmbeddingNorm, b.emb, int32(nn), int32(nb), eps)
			},
			func() error { return project(s.FCEmbedding, b.emb, b.e2, nn, nb) },
			func() error {
				return launchStrataNativeQSANorm(state, functions, at(rows, b0*hcn), s.HiddenNorm, b.hn, int32(hcn), int32(nb), eps)
			},
			// every stream through fc_hidden
			func() error { return project(s.FCHidden, b.hn, b.residual, nn, nb*hc) },
			func() error {
				return launchStrataAddStreamsBroadcast(state, functions, b.residual, b.e2, b.residual, nn, int32(hc), int32(nb))
			},
			// the attention hyper-connection's read, its mixed input only: this pass writes nothing back
			func() error {
				return launchGridABI(state, functions[kernelStrataPromptGrNormRs], driver.Dim3{X: uint32(nb * hc), Y: 1, Z: 1},
					driver.Dim3{X: strataGRThreads, Y: 1, Z: 1}, &b.residual, &s.Attention.Norm, &eps, &b.rs, &b.xn)
			},
			func() error { return bf16(s.Attention.Down, b.xn, b.lo, nb, strataGRRank, hcn) },
			func() error {
				count := nb * strataGRRank
				return launch1DABI(state, functions[kernelStrataPromptGrSilu], uint32(count), &b.lo, &b.lo16, &count)
			},
			func() error { return bf16(s.Attention.Up, b.lo16, b.hn, nb, hcn, strataGRRank) },
			func() error {
				return launch1DABI(state, functions[kernelStrataPromptGrMixR], uint32(nb*nn), &b.residual, &b.rs,
					&s.Attention.Norm, &b.hn, &b.mixed, &nb)
			},
			// K and V into the drafter's cache, as its own pass appends them
			func() error { return project(s.Key, b.mixed, b.k, kv, nb) },
			func() error { return project(s.Value, b.mixed, b.v, kv, nb) },
			func() error {
				return launchStrataNativeQSANorm(state, functions, b.k, s.KeyNorm, b.k, int32(hd), int32(nb*nkv), eps)
			},
			func() error {
				return launchStrataNativeRopeApply(state, functions, b.k, b.k, int32(nb*nkv), int32(hd),
					strataNativeNormRopeRotation, d.rope, at(pos, b0*nkv), 0)
			},
			func() error {
				return launchStrataKVAppendSteps(state, functions, d.pools, d.host, d.table, at(step, b0*strataDrafterStepWords),
					b.k, b.v, geometry, int32(nb), strataDrafterStepWords, int32(kv))
			},
		}
		for _, launch := range launches {
			if err := launch(); err != nil {
				return fmt.Errorf("mtp prefill: the batched pass: %w", err)
			}
		}
	}
	return state.Driver.StreamSynchronize(state.Stream)
}

// prefillStrataDrafterBatch is the batched pass (E-9) over the cells from firstNeeded on: the drafter's own stream idle
// before its records and its K/V are written (mtp.idle), the records with each K row's position, a key head a row.
func (e *Executor) prefillStrataDrafterBatch(ctx context.Context, d *StrataDrafter, rows driver.DevicePtr, next []int32, cell0,
	firstNeeded int64) error {
	records := strataDrafterPrefillRecords(next, cell0, tensor.QSAKeyValueHeads)
	return e.worker.Do(ctx, func(state *device.State) error {
		if err := state.Driver.StreamSynchronize(d.cs); err != nil {
			return err
		}
		if err := d.prefillRecordsUpload(state, records, state.Stream); err != nil {
			return err
		}
		return e.prefillStrataDrafterBatched(state, d, rows, int64(len(next)), cell0, firstNeeded)
	})
}

// strataDrafterPrefillRecords is a prefill's records for its rows from cell0, next the token after each: the tokens, then
// each row's step record, then each row's position for heads heads.
func strataDrafterPrefillRecords(next []int32, cell0, heads int64) []int32 {
	n := int64(len(next))
	records := make([]int32, n*(1+strataDrafterStepWords+heads))
	tok, step, pos := records[:n], records[n:n+strataDrafterStepWords*n], records[n+strataDrafterStepWords*n:]
	for i := range n {
		cell := cell0 + i
		tok[i] = next[i]
		record := step[i*strataDrafterStepWords:]
		record[0], record[1] = int32(cell), int32(cell+1)
		record[strataRoundStepBlocks], record[strataRoundStepWidth] = int32((cell+1)/tensor.QSABlockCells), int32(cell+1)
		for h := range heads {
			pos[i*heads+h] = int32(cell)
		}
	}
	return records
}

// prefillRecordsUpload uploads a prefill's records on stream into the drafter's record buffer (pf_dev_), grown to fit, as
// Strata's records go up at once.
func (d *StrataDrafter) prefillRecordsUpload(state *device.State, records []int32, stream driver.Stream) error {
	words := int64(len(records))
	if d.prefillCapacity < words {
		if d.prefillRecords != 0 {
			if err := state.Driver.MemFree(d.prefillRecords); err != nil {
				return err
			}
			d.prefillRecords, d.prefillCapacity = 0, 0
		}
		var err error
		if d.prefillRecords, err = d.resources.allocate(state, uint64(words)*strataRoundWordBytes); err != nil {
			return fmt.Errorf("mtp prefill: the input records do not fit: %w", err)
		}
		d.prefillCapacity = words
	}
	if err := state.Driver.MemcpyHtoDAsync(d.prefillRecords, int32Words(records), stream); err != nil {
		return errors.Join(errors.New("mtp prefill: the input records upload failed"), err)
	}
	return nil
}
