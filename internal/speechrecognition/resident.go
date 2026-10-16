package speechrecognition

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"overgo/internal/binaryschema"
	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// residentEncoder owns immutable feed-forward weights and one current shape per block.
// Calls serialize through the existing device worker and release only after completion.
type residentEncoder struct {
	mu      sync.Mutex
	session *graphruntime.ResidentSession
	blocks  map[*feedForward]*residentBlock
	budget  uint64
	static  uint64
}

type residentBlock struct {
	name          string
	weights       *feedForward
	program       *graphruntime.ResidentProgram
	input, output *tensor.Tensor
	rows          int
}

// BindDevice retains feed-forward execution before the encoder accepts requests.
func (e *Encoder) BindDevice(ctx context.Context, ordinal int) error {
	if e == nil || e.resident != nil || ctx == nil {
		return errors.New("encoder: device binding requires an unbound encoder and context")
	}
	d := &residentEncoder{blocks: make(map[*feedForward]*residentBlock), budget: e.memoryBytes}
	for index := range e.blocks {
		b := &e.blocks[index]
		for side, f := range []*feedForward{&b.first, &b.second} {
			for _, values := range [][]float32{f.norm.weight, f.norm.bias, f.in.weight, f.in.bias, f.out.weight, f.out.bias} {
				d.static += uint64(len(values)) * binaryschema.Uint32Bytes
			}
			d.blocks[f] = &residentBlock{name: fmt.Sprintf("encoder.%d.ff.%d", index, side), weights: f}
		}
	}
	if d.static >= d.budget {
		return errors.New("encoder: resident weights exceed declared byte budget")
	}
	var err error
	d.session, err = graphruntime.NewResidentSession(ordinal)
	if err != nil {
		return err
	}
	for _, b := range d.blocks {
		if err := d.compile(ctx, e, b, 1); err != nil {
			return errors.Join(err, d.session.Close(context.WithoutCancel(ctx)))
		}
	}
	e.resident = d
	return nil
}

func (d *residentEncoder) compile(ctx context.Context, e *Encoder, block *residentBlock, rows int) error {
	f := block.weights
	// The sum bounds graph-owned inputs and intermediates before allocation.
	builder := tensor.NewBuilder()
	input := builder.Input("frames", dtype.F32, tensor.MustShape(uint64(f.in.in), uint64(rows)))
	bindings := make(map[*tensor.Tensor][]float32)
	bind := func(name string, values []float32, shape ...uint64) *tensor.Tensor {
		node := builder.Input(block.name+name, dtype.F32, tensor.MustShape(shape...))
		bindings[node] = values
		return node
	}
	affine := func(x *tensor.Tensor, name string, p linear) *tensor.Tensor {
		y := builder.MulMat(bind(name+".weight", p.weight, uint64(p.in), uint64(p.out)), x)
		if len(p.bias) != 0 {
			y = builder.Add(y, bind(name+".bias", p.bias, uint64(p.out)))
		}
		return y
	}
	y := builder.AffineLayerNorm(input, bind(".norm.weight", f.norm.weight, uint64(f.in.in)), bind(".norm.bias", f.norm.bias, uint64(f.in.in)), float32(e.lnEpsilon))
	y = affine(y, ".in", f.in)
	if e.activation == "silu" {
		y = builder.SiLU(y)
	} else {
		y = builder.GELUErf(y)
	}
	y = affine(y, ".out", f.out)
	y = builder.Add(input, builder.Scale(y, e.ffScale))
	graph, err := tensor.CompileProgram(y)
	if err != nil {
		return err
	}
	remaining := d.budget - d.static
	for _, node := range graph.Order() {
		if _, weight := bindings[node]; weight {
			continue
		}
		bytes, err := node.Shape.Bytes(node.Type)
		if err != nil || bytes > remaining {
			return errors.New("encoder: resident activation graph exceeds declared byte budget")
		}
		remaining -= bytes
	}
	program, err := d.session.Compile(ctx, block.name, nil, y)
	if err != nil {
		return err
	}
	for node, values := range bindings {
		if err := d.session.BindF32(ctx, program, "encoder", node, values); err != nil {
			return errors.Join(err, d.session.Release(context.WithoutCancel(ctx), program))
		}
	}
	// Bind the replacement before releasing its predecessor so weights stay resident.
	if err := d.session.Release(context.WithoutCancel(ctx), block.program); err != nil {
		return errors.Join(err, d.session.Release(context.WithoutCancel(ctx), program))
	}
	block.program, block.input, block.output, block.rows = program, input, y, rows
	return nil
}

func (d *residentEncoder) forward(ctx context.Context, e *Encoder, f *feedForward, x []float32, rows int) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.session == nil {
		return errors.New("encoder: resident execution is closed")
	}
	block := d.blocks[f]
	if block == nil {
		return errors.New("encoder: feed-forward weights differ from resident binding")
	}
	if block.rows != rows {
		if err := d.compile(ctx, e, block, rows); err != nil {
			return err
		}
	}
	result, err := d.session.Execute(ctx, block.program, map[*tensor.Tensor]reference.Value{
		block.input: {Shape: block.input.Shape, Data: x},
	})
	if err == nil {
		copy(x, result[block.output].Data)
	}
	return err
}

// Close releases resident execution after the encoder finishes all requests.
func (e *Encoder) Close(ctx context.Context) error {
	if e == nil || e.resident == nil {
		return nil
	}
	d := e.resident
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.session == nil {
		return nil
	}
	err := d.session.Close(ctx)
	d.session, d.blocks = nil, nil
	return err
}

// BindDevice retains feed-forward execution on the explicitly selected device.
// Host callers retain the original execution until this binding succeeds.
func (r *Recognizer) BindDevice(ctx context.Context, ordinal int) error {
	if r == nil {
		return errors.New("recognizer: device binding requires a loaded model")
	}
	return r.encoder.BindDevice(ctx, ordinal)
}

// Close releases resident execution after its owner finishes all requests.
func (r *Recognizer) Close(ctx context.Context) error {
	if r == nil {
		return nil
	}
	return r.encoder.Close(ctx)
}

// BindDevice retains the acoustic encoder's feed-forward execution.
func (t *Transducer) BindDevice(ctx context.Context, ordinal int) error {
	if t == nil {
		return errors.New("transducer: device binding requires a loaded model")
	}
	return t.encoder.BindDevice(ctx, ordinal)
}

// Close releases resident execution after all transducer requests finish.
func (t *Transducer) Close(ctx context.Context) error {
	if t == nil {
		return nil
	}
	return t.encoder.Close(ctx)
}

// PeakDeviceBytes reports the resident session's observed allocation high-water mark.
func (r *Recognizer) PeakDeviceBytes() (uint64, error) {
	if r == nil || r.encoder == nil || r.encoder.resident == nil {
		return 0, nil
	}
	d := r.encoder.resident
	d.mu.Lock()
	defer d.mu.Unlock()
	stats, err := d.session.Stats(context.Background())
	return stats.Device.PeakBytes, err
}
