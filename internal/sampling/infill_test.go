package sampling

import (
	"math"
	"slices"
	"testing"
)

func TestInfillKeepsOnlyEOGWhenMassIsHigh(t *testing.T) {
	vocabulary := &InfillVocabulary{
		Pieces: []string{"a", "b", "c", "</s>"},
		EOG:    []bool{false, false, false, true},
		EOT:    3,
		EOS:    3,
	}
	candidates := infillCandidates([]float64{0.25, 0.25, 0.25, 0.25})
	got, err := new(Sampler).applyInfill(candidates, vocabulary)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].id != 3 {
		t.Fatalf("infill candidates = %+v, want EOG only", got)
	}
}

func TestInfillCombinesCommonPrefixes(t *testing.T) {
	vocabulary := &InfillVocabulary{
		Pieces: []string{"a", "ab", "x", "</s>"},
		EOG:    []bool{false, false, false, true},
		EOT:    3,
		EOS:    3,
	}
	candidates := infillCandidates([]float64{0.15, 0.25, 0.599, 0.001})
	got, err := new(Sampler).applyInfill(candidates, vocabulary)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]int, len(got))
	for index, item := range got {
		ids[index] = item.id
	}
	if !slices.Equal(ids, []int{2, 1, 3}) {
		t.Fatalf("infill IDs = %v, want [2 1 3]", ids)
	}
}

func TestInfillFallsBackToEOT(t *testing.T) {
	pieces := make([]string, 10)
	eog := make([]bool, 10)
	probabilities := make([]float64, 10)
	for index := 0; index < 9; index++ {
		pieces[index] = string(rune('a'+index)) + "_"
		probabilities[index] = 0.11
	}
	pieces[9] = "</s>"
	eog[9] = true
	probabilities[9] = 0.01
	got, err := new(Sampler).applyInfill(
		infillCandidates(probabilities),
		&InfillVocabulary{
			Pieces: pieces,
			EOG:    eog,
			EOT:    9,
			EOS:    -1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].id != 9 {
		t.Fatalf("infill fallback = %+v, want EOT", got)
	}
}

func TestInfillStageRequiresVocabulary(t *testing.T) {
	if _, err := New(Config{
		Temperature: 1,
		Samplers:    []SamplerStage{SamplerInfill},
	}); err == nil {
		t.Fatal("infill sampler without vocabulary was accepted")
	}
	order, err := ParseSamplerOrder("top_k;infill")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(order, []SamplerStage{SamplerTopK, SamplerInfill}) {
		t.Fatalf("order = %v", order)
	}
}

func infillCandidates(probabilities []float64) []candidate {
	result := make([]candidate, len(probabilities))
	for index, probability := range probabilities {
		result[index] = candidate{
			id:          index,
			scaledLogit: math.Log(probability),
		}
	}
	return result
}
