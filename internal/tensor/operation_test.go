package tensor

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

func TestOperationCatalogIsDenseAndUnique(t *testing.T) {
	descriptors := Operations()
	if len(descriptors) != int(OpTopKPartials)+1 {
		t.Fatalf("descriptor count = %d", len(descriptors))
	}
	names := make(map[string]struct{}, len(descriptors))
	for index, descriptor := range descriptors {
		if descriptor.Op != Op(index) || descriptor.Name == "" {
			t.Fatalf("descriptor %d = %+v", index, descriptor)
		}
		if _, exists := names[descriptor.Name]; exists {
			t.Fatalf("duplicate operation name %q", descriptor.Name)
		}
		names[descriptor.Name] = struct{}{}
		if descriptor.Op != OpInput && descriptor.Backends != allExecutionBackends {
			t.Fatalf("operation %s backend coverage = %d", descriptor.Name, descriptor.Backends)
		}
		if descriptor.Op.String() != descriptor.Name {
			t.Fatalf("operation %d string = %q", descriptor.Op, descriptor.Op.String())
		}
	}
	if _, ok := DescribeOperation(Op(65535)); ok {
		t.Fatal("unknown operation described")
	}
}

func TestOperationStorageContracts(t *testing.T) {
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(8))
	reshape := builder.Reshape(input, 2, 4)
	slice := builder.FlatSlice(reshape, 2, 3)
	for node, wantOffset := range map[*Tensor]uint64{reshape: 0, slice: 2} {
		view, aliases, err := ResolveStorageView(node)
		if err != nil || !aliases || view.Input != 0 || view.ElementOffset != wantOffset {
			t.Fatalf("%s storage view = (%+v, %t, %v)", node.Op, view, aliases, err)
		}
	}
	view, _, _ := ResolveStorageView(slice)
	if offset, err := view.ByteOffset(dtype.F32); err != nil || offset != 8 {
		t.Fatalf("F32 byte offset = (%d, %v)", offset, err)
	}
	if _, err := view.ByteOffset(dtype.Q8_0); err == nil {
		t.Fatal("unaligned quantized view offset accepted")
	}
	malformed := &Tensor{ID: 99, Op: OpFlatSlice, Inputs: []*Tensor{input}, Shape: MustShape(1)}
	if _, err := Topological(malformed); err == nil {
		t.Fatal("malformed storage view passed graph validation")
	}
}

func TestOutputTargetAliasContract(t *testing.T) {
	builder := NewBuilder()
	left := builder.Input("left", dtype.F32, MustShape(3))
	right := builder.Input("right", dtype.F32, MustShape(2))
	joined := builder.Concat(left, right, 0)
	contract, err := CompileOutputTargetContract(joined)
	if err != nil {
		t.Fatal(err)
	}
	if contract.Bytes != 5*4 || contract.Alignment != 4 || contract.Alias == nil ||
		contract.Alias.Input != 0 || contract.Alias.InitializedBytes != 3*4 ||
		contract.Alias.WriteOffsetBytes != 3*4 || contract.Alias.WriteBytes != 2*4 {
		t.Fatalf("concat target contract = %+v", contract)
	}
	scaled := builder.Scale(left, 2)
	contract, err = CompileOutputTargetContract(scaled)
	if err != nil || contract.Alias != nil || contract.Bytes != 3*4 {
		t.Fatalf("scale target contract = (%+v, %v)", contract, err)
	}
}
