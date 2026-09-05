package pytorchzip

import "testing"

func TestCheckpointNamespaceMetadata(t *testing.T) {
	for _, tc := range []struct {
		name, module, class string
		argument            bool
		valid               bool
	}{
		{name: "saved configuration", module: "argparse", class: "Namespace", valid: true},
		{name: "unknown class", module: "example", class: "Constructor"},
		{name: "constructor arguments", module: "argparse", class: "Namespace", argument: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var w pickleWriter
			w.raw([]byte{0x80, 2, '}', '('})
			w.str("args")
			w.global(tc.module, tc.class)
			if tc.argument {
				w.int32v(1)
				w.op(0x85)
			} else {
				w.op(')')
			}
			w.op(0x81)
			w.raw([]byte{'}', '('})
			w.str("input_dimension")
			w.int32v(80)
			w.raw([]byte{'u', 'b'})
			w.tensorEntry("weight", "FloatStorage", "0", 6, 0, []int64{2, 3}, []int64{3, 1})
			w.raw([]byte{'u', '.'})
			catalog, err := ParseCatalog(w.buf.Bytes())
			if !tc.valid {
				if err == nil {
					t.Fatal("accepted unsupported object construction")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(catalog.Tensors) != 1 || catalog.Tensors[0].Name != "weight" || catalog.Tensors[0].Numel != 6 || catalog.Scalars["input_dimension"] != int64(80) {
				t.Fatalf("metadata after Namespace: %+v", catalog)
			}
		})
	}
	for _, truncated := range [][]byte{{0x80, 2, 0x81}, {0x80, 2, ')', 0x81}} {
		if _, err := ParseCatalog(truncated); err == nil {
			t.Fatal("accepted truncated NEWOBJ stack")
		}
	}
}
