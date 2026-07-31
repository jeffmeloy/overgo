package sampling

import "testing"

func FuzzGBNFCompileNeverPanics(f *testing.F) {
	f.Add(`root ::= "a"`)
	f.Add(`root ::= [a-z]{1,4}`)
	f.Add(`root ::= <[0]> | <[1]>`)
	f.Fuzz(func(t *testing.T, source string) {
		if len(source) > maxGBNFSourceBytes+1 {
			t.Skip()
		}
		_, _ = NewGBNFGrammar(source, "root", [][]byte{[]byte("a"), []byte("b"), nil}, []int{2})
	})
}

func FuzzJSONSchemaToGrammarNeverPanics(f *testing.F) {
	f.Add([]byte(`{"type":"boolean"}`))
	f.Add([]byte(`{"type":"string","pattern":"^(?:foo|bar){1,3}$"}`))
	f.Add([]byte(`{"$ref":"#","anyOf":[{"type":"null"}]}`))
	f.Fuzz(func(t *testing.T, input []byte) {
		if len(input) > maxJSONSchemaBytes+1 {
			t.Skip()
		}
		grammar, err := JSONSchemaToGrammar(input)
		if err != nil {
			return
		}
		_, _ = NewGBNFGrammar(
			grammar,
			"root",
			[][]byte{[]byte("a"), nil},
			[]int{1},
		)
	})
}

func FuzzSamplerLoadStateNeverPanics(f *testing.F) {
	sampler, err := New(Config{Temperature: 0.8, TopK: 4, Seed: 17})
	if err != nil {
		f.Fatal(err)
	}
	valid, err := sampler.SaveState()
	if err != nil {
		f.Fatal(err)
	}
	f.Add(valid)
	f.Add([]byte(samplerStateMagic))
	f.Add([]byte("not-state"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 2<<20 {
			t.Skip()
		}
		target, newErr := New(Config{Temperature: 0.8, TopK: 4, Seed: 17})
		if newErr != nil {
			t.Fatal(newErr)
		}
		_ = target.LoadState(data)
	})
}
