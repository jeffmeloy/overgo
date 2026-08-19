package tokenizer

import (
	"reflect"
	"testing"

	"overgo/internal/gguf"
)

// dnaTestVocab declares a k=2 sequence extension over a tiny gpt2 base:
// base IDs 0-2 are h, i, x; the extension occupies IDs 3..21 as three
// declared specials followed by the sixteen 2-mers in alphabet product
// order. Everything the encoder needs comes from this metadata.
func dnaTestVocab(t *testing.T, autoTags bool) *Vocab {
	t.Helper()
	file := &gguf.File{Metadata: []gguf.Metadata{
		scalar("tokenizer.ggml.model", gguf.ValueTypeString, "gpt2"),
		scalar("tokenizer.ggml.pre", gguf.ValueTypeString, "default"),
		array("tokenizer.ggml.tokens", gguf.ValueTypeString, []string{
			"h", "i", "x",
			"[P0]", "[P1]", "[P2]", "[P3]", "[P4]", "[P5]", "[P6]", "[P7]", "[P8]", "[P9]",
			"[P10]", "[P11]", "[P12]", "[P13]", "[P14]", "[P15]", "[P16]", "[P17]", "[P18]",
		}),
		array("tokenizer.ggml.token_type", gguf.ValueTypeInt32, []int32{
			1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
		}),
		array("tokenizer.ggml.merges", gguf.ValueTypeString, []string{}),
		scalar(MetadataDNAK, gguf.ValueTypeUint32, uint32(2)),
		scalar(MetadataDNAStartID, gguf.ValueTypeUint32, uint32(3)),
		scalar(MetadataDNAVocabulary, gguf.ValueTypeUint32, uint32(19)),
		array(MetadataDNASpecialTokens, gguf.ValueTypeString, []string{"<dna>", "</dna>", "<oov>"}),
		scalar(MetadataDNAAutoTags, gguf.ValueTypeBool, autoTags),
	}}
	vocab, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	return vocab
}

// TestCarbonDNAEncoding pins encode parity with the upstream reference
// tokenizer the Carbon family ships: declared tags route to the extension,
// sequences chunk into k-mers stepped by k, off-alphabet windows become the
// declared out-of-vocabulary token, a trailing remainder right-pads with the
// first alphabet symbol, lowercase input uppercases, untagged text takes the
// base path, unclosed and unopened tags follow the reference scanner, and
// auto-tag deployments wrap whole. IDs mirror the decoder's k-mer math, so
// encode-decode is a round trip. The production code is metadata-driven --
// the Carbon name appears only in this test's provenance.
func TestCarbonDNAEncoding(t *testing.T) {
	vocab := dnaTestVocab(t, false)
	const (
		begin = TokenID(3)
		end   = TokenID(4)
		oov   = TokenID(5)
		kmer0 = TokenID(6) // AA; product order over ATCG, most significant first
	)
	kmer := func(first, second int) TokenID { return kmer0 + TokenID(first*4+second) }
	acgt := []TokenID{kmer(0, 2), kmer(3, 1)} // AC, GT with digits A=0 T=1 C=2 G=3

	for name, testcase := range map[string]struct {
		text string
		want []TokenID
	}{
		"tagged sequence": {"<dna>ACGT</dna>", append(append([]TokenID{begin}, acgt...), end)},
		"lowercase":       {"<dna>acgt</dna>", append(append([]TokenID{begin}, acgt...), end)},
		"remainder pads with first symbol": {"<dna>ACG</dna>",
			[]TokenID{begin, kmer(0, 2), kmer(3, 0), end}}, // G + A pad = GA
		"off-alphabet window": {"<dna>ACNT</dna>", []TokenID{begin, kmer(0, 2), oov, end}},
		"mixed base and sequence": {"hi<dna>AC</dna>x",
			[]TokenID{0, 1, begin, kmer(0, 2), end, 2}},
		"unclosed begin":      {"<dna>AC", []TokenID{begin, kmer(0, 2)}},
		"unopened end":        {"AC</dna>", []TokenID{kmer(0, 2), end}},
		"bare tags":           {"<dna></dna>", []TokenID{begin, end}},
		"untagged stays base": {"hix", []TokenID{0, 1, 2}},
	} {
		ids, err := vocab.Encode(testcase.text, EncodeOptions{})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !reflect.DeepEqual(ids, testcase.want) {
			t.Fatalf("%s: Encode(%q) = %v, want %v", name, testcase.text, ids, testcase.want)
		}
	}

	ids, err := vocab.Encode("<dna>ACGT</dna>", EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := vocab.Decode(ids, true)
	if err != nil || decoded != "<dna>ACGT</dna>" {
		t.Fatalf("round trip = (%q, %v)", decoded, err)
	}

	auto := dnaTestVocab(t, true)
	ids, err = auto.Encode("ACGT", EncodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if want := append(append([]TokenID{begin}, acgt...), end); !reflect.DeepEqual(ids, want) {
		t.Fatalf("auto-tag Encode = %v, want %v", ids, want)
	}
	tagged, err := auto.Encode("<dna>ACGT</dna>", EncodeOptions{})
	if err != nil || !reflect.DeepEqual(tagged, ids) {
		t.Fatalf("auto-tag double wrap: (%v, %v)", tagged, err)
	}
}
