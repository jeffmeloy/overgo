package evaluation

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/dlclark/regexp2/v2"
	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestIFEvalLanguageAcceptance(t *testing.T) {
	p, err := compiledIFEvalLanguage()
	if err != nil {
		t.Fatal(err)
	}
	parse := func(text string) artifact.ID {
		t.Helper()
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	var selection struct {
		Oracle   artifact.ID
		Native   artifact.ID `json:"native_scores"`
		Profile  string      `json:"compiled_profile_sha256"`
		Boundary int         `json:"boundary_cases"`
		Views    int         `json:"retained_views"`
		Distinct int         `json:"distinct_retained"`
	}
	readRetainedEvidence(t, store, parse("evidence:sha256:7af0c46db65d189c70cde72baef789b7c5a894b8112bd5baa551be8e1ac0d4a4"), &selection)
	if selection.Native.DigestHex() != "78a1241f82ce179d229248564ccffb66f291d922eba864d1a261fc94d386c839" || selection.Profile != fmt.Sprintf("%x", sha256.Sum256(ifevalLanguageProfile)) || selection.Boundary != 24 || selection.Views != 760 || selection.Distinct != 341 {
		t.Fatal("native language selection changed")
	}
	type observation struct {
		Cleaned       string `json:"cleaned_sha256"`
		NGrams        int    `json:"ngrams"`
		NGramHash     string `json:"ngram_sha256"`
		Detected      string
		Probabilities []float64
		Exception     int `json:"exception_code"`
		Upper, Lower  bool
	}
	var oracle struct {
		Boundary []struct {
			Response string
			observation
		}
		Retained []struct {
			Name string
			View int
			Hash string `json:"response_sha256"`
		}
		Observations map[string]observation
		RNG          []struct {
			Size           int
			Gauss, Uniform float64
			Choice         int
			Bits           uint32
		}
		WordCount int    `json:"rng_word_count"`
		WordHash  string `json:"rng_words_sha256"`
		Checks    []struct {
			Boundary      int
			Instruction   string
			Kwargs        map[string]json.RawMessage
			Strict, Loose []bool
		}
	}
	readRetainedEvidence(t, store, selection.Oracle, &oracle)
	if len(oracle.Boundary) != selection.Boundary || len(oracle.Retained) != selection.Views || len(oracle.Observations) != selection.Distinct || len(oracle.Checks) != 72 || len(oracle.RNG) != 35 || oracle.WordCount != 1024 {
		t.Fatal("native language denominator changed")
	}
	r := p.random()
	h := sha256.New()
	var word [4]byte
	for range oracle.WordCount {
		binary.LittleEndian.PutUint32(word[:], r.next())
		h.Write(word[:])
	}
	if hex.EncodeToString(h.Sum(nil)) != oracle.WordHash {
		t.Fatal("native RNG words differ across state cycles")
	}
	r = p.random()
	maxNormalError := 0.0
	for index, want := range oracle.RNG {
		g, c, u, b := r.gauss(), r.choice(want.Size), r.uniform(), r.next()
		if c != want.Choice || u != want.Uniform || b != want.Bits {
			t.Fatalf("native RNG operation order differs at %d", index)
		}
		maxNormalError = max(maxNormalError, math.Abs(g-want.Gauss))
	}
	ngramHash := func(grams []string) string {
		h := sha256.New()
		var size [8]byte
		for _, gram := range grams {
			binary.LittleEndian.PutUint64(size[:], uint64(len(gram)))
			h.Write(size[:])
			h.Write([]byte(gram))
		}
		return hex.EncodeToString(h.Sum(nil))
	}
	// Labels and instruction verdicts are exact. Probability comparisons also
	// reject changes beyond a conservative operation-count roundoff budget.
	epsilon := math.Nextafter(1, 2) - 1
	operationBudget := float64(len(p.input.Languages) * p.input.Trials * (p.input.Iterations + 1))
	roundoffBudget := operationBudget * epsilon / (1 - operationBudget*epsilon)
	maximumError := 0.0
	compare := func(name, raw string, want observation) {
		t.Helper()
		clean, grams, err := p.features(raw)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256([]byte(clean))) != want.Cleaned || len(grams) != want.NGrams || ngramHash(grams) != want.NGramHash {
			t.Fatalf("%s normalization or ngrams differ", name)
		}
		if ifevalUniformCase(raw, nltkUpper) != want.Upper || ifevalUniformCase(raw, nltkLower) != want.Lower {
			t.Fatalf("%s native case predicate differs", name)
		}
		label, features, err := p.detect(raw)
		if err != nil {
			t.Fatal(err)
		}
		if !features {
			if want.Exception == 0 {
				t.Fatalf("%s missing native no-feature disposition", name)
			}
			return
		}
		if want.Exception != 0 || label != want.Detected {
			t.Fatalf("%s detected=%s/%s", name, label, want.Detected)
		}
		prob, err := p.probabilities(grams)
		if err != nil {
			t.Fatal(err)
		}
		if len(prob) != len(want.Probabilities) {
			t.Fatal("language probability denominator differs")
		}
		for index, value := range prob {
			delta := math.Abs(value - want.Probabilities[index])
			maximumError = max(maximumError, delta)
			if math.IsNaN(value) || delta > roundoffBudget {
				t.Fatalf("%s probability %d delta=%g exceeds roundoff budget=%g", name, index, delta, roundoffBudget)
			}
		}
	}
	for index, c := range oracle.Boundary {
		compare(fmt.Sprintf("boundary/%d", index), c.Response, c.observation)
	}
	for index, c := range oracle.Checks {
		if c.Boundary < 0 || c.Boundary >= len(oracle.Boundary) {
			t.Fatal("invalid native check binding")
		}
		rules, mapped := ifevalRuleFor(c.Instruction, c.Kwargs)
		if !mapped || len(rules) != 1 {
			t.Fatal("native language instruction missing or duplicated")
		}
		rule, err := compileInstructionRule(rules[0])
		if err != nil {
			t.Fatal(err)
		}
		strict, loose, err := evaluateInstructionViews(oracle.Boundary[c.Boundary].Response, []compiledInstructionRule{rule})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(strict, c.Strict) || !slices.Equal(loose, c.Loose) {
			t.Errorf("boundary check %d %s strict=%v/%v loose=%v/%v", index, c.Instruction, strict, c.Strict, loose, c.Loose)
		}
	}
	var native struct{ Selection, Profile artifact.ID }
	readRetainedEvidence(t, store, selection.Native, &native)
	responses := retainedIFEvalResponses(t, store, native.Selection, native.Profile)
	seen, viewKeys := map[string]bool{}, map[string]bool{}
	for _, row := range oracle.Retained {
		key := fmt.Sprintf("%s/%d", row.Name, row.View)
		raw, found := responses[row.Name]
		views := looseInstructionViews(raw)
		if !found || row.View < 0 || row.View >= len(views) || viewKeys[key] {
			t.Fatal("missing or duplicate native view")
		}
		viewKeys[key] = true
		raw = views[row.View]
		if fmt.Sprintf("%x", sha256.Sum256([]byte(raw))) != row.Hash {
			t.Fatal("native language view identity differs")
		}
		want, present := oracle.Observations[row.Hash]
		if !present {
			t.Fatal("native language observation missing")
		}
		if !seen[row.Hash] {
			compare(key, raw, want)
			seen[row.Hash] = true
		}
	}
	if len(seen) != selection.Distinct {
		t.Fatal("unconsumed native language observations")
	}
	t.Run("changed profiles and unresolved languages are refused", func(t *testing.T) {
		if _, err := loadIFEvalLanguage(bytes.Replace(ifevalLanguageProfile, []byte("1.0.9"), []byte("1.0.8"), 1)); err == nil {
			t.Fatal("changed language profile accepted")
		}
		for _, values := range [][]string{nil, {""}, {"not-a-language"}, {"en", "fr"}} {
			if _, err := compileInstructionRule(InstructionRule{Name: "language", Kind: ifevalLanguageRule, Values: values}); err == nil {
				t.Fatal("unresolved language accepted")
			}
		}
	})
	t.Run("execution errors are not native no-feature outcomes", func(t *testing.T) {
		broken := *p
		broken.url, err = regexp2.Compile(`(?:^){100}`, regexp2.OptionMaxBacktrackingStackSize(32))
		if err != nil {
			t.Fatal(err)
		}
		rule, err := compileInstructionRule(InstructionRule{Name: "language", Kind: ifevalLanguageRule, Values: []string{"en"}})
		if err != nil {
			t.Fatal(err)
		}
		rule.language = &broken
		strict, loose, err := evaluateInstructionViews("HELLO", []compiledInstructionRule{rule})
		if !errors.Is(err, regexp2.ErrBacktrackingStackLimit) || strict != nil || loose != nil {
			t.Fatal("execution error became a native verdict")
		}
		rule.language = p
		strict, loose, err = evaluateInstructionViews("This is a complete English sentence.", []compiledInstructionRule{rule})
		if err != nil || !slices.Equal(strict, []bool{true}) || !slices.Equal(loose, []bool{true}) {
			t.Fatal("language evaluation failed to recover")
		}
	})
	t.Logf("native language parity: boundaries=%d checks=%d views=%d unique=%d max_probability_error=%g max_gaussian_error=%g", selection.Boundary, len(oracle.Checks), selection.Views, selection.Distinct, maximumError, maxNormalError)
}
