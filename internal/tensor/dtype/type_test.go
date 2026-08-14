package dtype

import "testing"

func TestIsQuantizedUsesPhysicalTraits(t *testing.T) {
	tests := []struct {
		name   string
		typeID Type
		want   bool
	}{
		{name: "block", typeID: Q4K, want: true},
		{name: "ternary block", typeID: TQ1_0, want: true},
		{name: "float", typeID: F32},
		{name: "native fp8", typeID: F8E4M3},
		{name: "unknown", typeID: Count},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.typeID.IsQuantized(); got != test.want {
				t.Fatalf("IsQuantized() = %t, want %t", got, test.want)
			}
		})
	}
}
