package quant

import "testing"

func TestBlockCodecLayoutsMatchTypeTraits(t *testing.T) {
	layouts := []blockCodecLayout{
		iq2XXSBlockLayout, iq2XSBlockLayout, iq2SBlockLayout,
		iq3XXSBlockLayout, iq3SBlockLayout, iq1SBlockLayout, iq1MBlockLayout,
		iq4NLBlockLayout, iq4XSBlockLayout, tq1BlockLayout, tq2BlockLayout,
		mxfp4BlockLayout, nvfp4BlockLayout, q1BlockLayout, q2BlockLayout,
		q4BlockLayout, q5BlockLayout, q8BlockLayout, q8KBlockLayout,
		q2KCodec.block, q3KCodec.block, q4KCodec.block, q5KCodec.block, q6KCodec.block,
	}
	for _, layout := range layouts {
		traits, ok := layout.dataType.Traits()
		if !ok {
			t.Fatalf("%s has no traits", layout.dataType)
		}
		if layout.elements != int(traits.BlockSize) || layout.size != int(traits.TypeSize) {
			t.Errorf("%s layout = (%d, %d), traits = (%d, %d)",
				layout.dataType, layout.elements, layout.size, traits.BlockSize, traits.TypeSize)
		}
	}
}

func TestQuantCodecFieldsFillTypedStorage(t *testing.T) {
	tests := []struct {
		name   string
		layout blockCodecLayout
		end    int
	}{
		{"iq2_xxs", iq2XXSBlockLayout, iqAuxiliaryStart},
		{"iq2_xs", iq2XSBlockLayout, iq2XSScaleStart + iq2XSScaleBytes},
		{"iq2_s", iq2SBlockLayout, iq2SScaleStart + iq2SScaleBytes},
		{"iq3_xxs", iq3XXSBlockLayout, iq3XXSScaleSignStart + iq3XXSScaleSignBytes},
		{"iq3_s", iq3SBlockLayout, iq3SScaleStart + iq3SScaleBytes},
		{"tq1_0", tq1BlockLayout, tq1ScaleStart + iqScaleBytes},
		{"tq2_0", tq2BlockLayout, tq2ScaleStart + iqScaleBytes},
		{"q1_0", q1BlockLayout, iqScaleBytes + q1PackedBytes},
		{"q2_K", q2KCodec.block, q2KCodec.minimum.end()},
		{"q3_K", q3KCodec.block, q3KCodec.delta.end()},
		{"q4_K", q4KCodec.block, q4KCodec.packed.end()},
		{"q5_K", q5KCodec.block, q5KCodec.packed.end()},
		{"q6_K", q6KCodec.block, q6KCodec.delta.end()},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.end != test.layout.size {
				t.Errorf("last field ends at %d, typed storage size is %d", test.end, test.layout.size)
			}
		})
	}
}
