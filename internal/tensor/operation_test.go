package tensor

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

func TestOperationCatalogIsDenseAndUnique(t *testing.T) {
	names := make(map[string]struct{}, len(operationDescriptors))
	for index, descriptor := range operationDescriptors {
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

func TestOperationAttributesRejectMismatches(t *testing.T) {
	const (
		fixtureWidth    = 4
		fixtureTensorID = 1
		fixtureEpsilon  = 1e-5
	)
	builder := NewBuilder()
	input := builder.Input("input", dtype.F32, MustShape(fixtureWidth))
	if output := builder.unary(OpScale, input, RMSNormAttributes{Epsilon: fixtureEpsilon}); output != nil {
		t.Fatal("mismatched builder attributes produced a tensor")
	}
	if builder.Err() == nil {
		t.Fatal("mismatched builder attributes were not reported")
	}
	malformed := &Tensor{
		ID:     fixtureTensorID,
		Type:   dtype.F32,
		Shape:  MustShape(fixtureWidth),
		Op:     OpScale,
		Inputs: []*Tensor{input},
		Attrs:  RMSNormAttributes{Epsilon: fixtureEpsilon},
	}
	if _, err := Topological(malformed); err == nil {
		t.Fatal("mismatched graph attributes passed validation")
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
	capacityBuilder := NewBuilder()
	capacityBuilder.SetCacheAppendPlan(CacheAppendPlan{
		ActiveTokens: 2, SourceCapacityTokens: 2, CapacityTokens: 4,
	})
	cache := capacityBuilder.AppendCache(
		capacityBuilder.Input("cache", dtype.F32, MustShape(2, 1, 2)),
		capacityBuilder.Input("append", dtype.F32, MustShape(2, 1, 1)),
		2,
	)
	contract, err = CompileOutputTargetContract(cache)
	if err != nil {
		t.Fatal(err)
	}
	if contract.Bytes != 8*4 || contract.Alias == nil ||
		contract.Alias.InitializedBytes != 4*4 || contract.Alias.WriteOffsetBytes != 4*4 ||
		contract.Alias.WriteBytes != 2*4 {
		t.Fatalf("cache append target contract = %+v", contract)
	}
	scaled := builder.Scale(left, 2)
	contract, err = CompileOutputTargetContract(scaled)
	if err != nil || contract.Alias != nil || contract.Bytes != 3*4 {
		t.Fatalf("scale target contract = (%+v, %v)", contract, err)
	}
}
