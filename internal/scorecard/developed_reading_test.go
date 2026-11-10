package scorecard

import (
	"strings"
	"testing"
)

// TestAReadingSaysHowManyOfItsCasesAreDevelopment holds both the scorecard's and README's words for a held-out quality
// some of whose cases were asked in choosing its protocol (SQA-361): each says how many, and a reading with none says
// nothing of it.
func TestAReadingSaysHowManyOfItsCasesAreDevelopment(t *testing.T) {
	t.Parallel()
	cell := Cell{Value: 0.55, Cases: 1209, Developed: 312, Label: "accuracy, store/mmlu-pro (chat-template)"}
	if text := cell.QualityText(); !strings.Contains(text, "312 of them development evidence, asked in choosing how the model is asked") {
		t.Errorf("the scorecard says %q", text)
	}
	if brief := cell.Brief("MMLU-Pro"); brief != "MMLU-Pro 0.55 on 1,209 cases, 312 of them development evidence" {
		t.Errorf("README says %q", brief)
	}
	cell.Developed = 0
	if strings.Contains(cell.QualityText()+cell.Brief("MMLU-Pro"), "development") {
		t.Errorf("a reading none of whose cases is development evidence says %q and %q", cell.QualityText(), cell.Brief("MMLU-Pro"))
	}
}
