package latentvideo

import (
	"path/filepath"
	"reflect"
	"testing"
)

// TestRealTokenizerParity: real umt5-format tokenizer ids vs the reference
// repo's committed expectations (adaptive TestFixedUnigramTokenizerRealGated).
func TestRealTokenizerParity(t *testing.T) {
	modelDir := wanModelDir(t)
	tok, err := loadFixedUnigramTokenizer(filepath.Join(modelDir, "google", "umt5-xxl"), 16)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		text string
		ids  []int
		mask []int
	}{
		{
			text: "a red fox licking a vanilla ice cream cone",
			ids:  []int{289, 4062, 273, 56209, 2048, 41797, 289, 362, 2614, 273, 3116, 27355, 379, 292, 1, 0},
			mask: []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
		},
		{
			text: "  hello   world  ",
			ids:  []int{154424, 3914, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			mask: []int{1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			text: "色调，过曝",
			ids:  []int{273, 1838, 8483, 275, 4915, 85615, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			mask: []int{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
	}
	for _, tc := range cases {
		ids, mask, err := tok.EncodeWithMask(tc.text)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ids, tc.ids) || !reflect.DeepEqual(mask, tc.mask) {
			t.Fatalf("%q ids=%v mask=%v", tc.text, ids, mask)
		}
	}
}
