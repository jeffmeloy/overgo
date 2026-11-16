package evaluation

import (
	"fmt"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// alone names each case its own medium, a suite whose every question asks of a medium of its own.
func alone(cases map[string]bool) map[string]string {
	media := make(map[string]string, len(cases))
	for name := range cases {
		media[name] = name
	}
	return media
}

// TestPresetAdmissionReadsItsNullFromItsGroups holds a preset's admission to the media its questions ask about
// (preset-evidence/statistic, SQA-342): AdmitPreset read 524 questions of 80 clips as 524 independent pairs. One sign
// pattern, eight questions the preset loses among 32, reads two ways: asked of eight clips of their own, the loss's
// tail is 1/256 and the preset is refused; asked four of each of two clips, the two clips' signs alone are the
// evidence, a tail of 1/4, and it is admitted. A case of no medium the suite names is refused.
func TestPresetAdmissionReadsItsNullFromItsGroups(t *testing.T) {
	t.Parallel()
	person := map[string]float64{"width": 1024, "height": 1024, "steps": 30}
	reference := PresetArm{Settings: person, Format: "png"}
	candidate := PresetArm{Preset: "apg", Settings: person, Format: "png"}
	referenceRight, presetRight, clips := map[string]bool{}, map[string]bool{}, map[string]string{}
	for at := range 32 {
		name := fmt.Sprintf("question-%02d", at)
		referenceRight[name], presetRight[name] = true, at >= 8
		clips[name] = fmt.Sprintf("clip-%d", at/4)
	}
	questions := testcheck.Must(AdmitPreset(person, reference, candidate, referenceRight, presetRight, alone(referenceRight)))(t)
	grouped := testcheck.Must(AdmitPreset(person, reference, candidate, referenceRight, presetRight, clips))(t)
	if questions.Admitted || questions.Media != 32 || !strings.HasSuffix(questions.Says, "0 wins, 8 losses and 24 ties over 32 media") {
		t.Errorf("eight losses of questions each of its own clip read %+v; want refused over 32 media", questions)
	}
	if !grouped.Admitted || grouped.Media != 8 || !strings.HasSuffix(grouped.Says, "0 wins, 8 losses and 24 ties over 8 media") {
		t.Errorf("eight losses of two clips' questions read %+v; want admitted over 8 media, the two clips' signs alone", grouped)
	}
	delete(clips, "question-00")
	if _, err := AdmitPreset(person, reference, candidate, referenceRight, presetRight, clips); err == nil || !strings.Contains(err.Error(), "question-00") {
		t.Errorf("a case of no medium was admitted as evidence: %v", err)
	}
}
