package tensor

import "testing"

func TestOperationCatalogIsDenseAndUnique(t *testing.T) {
	descriptors := Operations()
	if len(descriptors) != int(OpSAMAttention)+1 {
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
