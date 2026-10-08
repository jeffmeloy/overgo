//go:build modeltest

package routedlm

import (
	"encoding/json"
	"os"
	"slices"
	"testing"

	"overgo/internal/hfbpe"
	"overgo/internal/testutil"
)

// promptIDsScript writes the fixture this test reads, from the model's own
// code. It is kept where Python is kept, on the machine that ran it and not
// in git: the fixture is the data.
const promptIDsScript = "scripts/python_golden_references/sensenova_prompt_ids.py"

// TestSenseNovaPromptIDsMatchUpstream holds the first stretch of the path an
// image request takes, from the prompt's text to the token ids the model
// reads, to the model's own code: its conversation template, its system
// message and its tokenizer. The prompts are those of the drawn-text measure,
// every brief of its list, and others that try a tokenizer: capitals, digits,
// punctuation, quotation marks, German and Chinese, and runs of spaces. The
// ids of the unconditional query, which every guided request also reads, are
// held too. The forward pass from given ids is compared with that code by
// another test; without this one a fault before the ids would read as the
// model's score, as a tokenizer's did for another model.
func TestSenseNovaPromptIDsMatchUpstream(t *testing.T) {
	raw, err := os.ReadFile(testutil.FixturePath(t, "sensenova/prompt_ids.json"))
	if err != nil {
		t.Fatalf("the upstream prompt ids are absent (%v); write them with %s", err, promptIDsScript)
	}
	var fixture struct {
		Unconditional []int `json:"unconditional_ids"`
		Cases         []struct {
			Prompt      string `json:"prompt"`
			Conditional []int  `json:"conditional_ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 || len(fixture.Unconditional) == 0 {
		t.Fatalf("the fixture holds %d prompts and %d unconditional ids", len(fixture.Cases), len(fixture.Unconditional))
	}
	tokenizer, err := hfbpe.LoadSplit(testutil.ModelArtifactDir(t, "SenseNova-U1-8B-MoT-Infographic-V3"))
	if err != nil {
		t.Skipf("UNAVAILABLE: SenseNova tokenizer unavailable: %v; parity NOT verified", err)
	}
	template := SenseNovaPromptTemplate()
	rendered := func(prompt string, role PromptRole) []int {
		t.Helper()
		query, err := RenderEditPrompt(tokenizer, prompt, role, template, PromptSource{})
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]int, len(query.IDs))
		for index, id := range query.IDs {
			ids[index] = int(id)
		}
		return ids
	}
	if got := rendered("", PromptRoleUnconditional); !slices.Equal(got, fixture.Unconditional) {
		t.Fatalf("the unconditional query is read as %v, and the model's own code reads %v", got, fixture.Unconditional)
	}
	for _, reference := range fixture.Cases {
		if got := rendered(reference.Prompt, PromptRolePrompt); !slices.Equal(got, reference.Conditional) {
			at := 0
			for at < min(len(got), len(reference.Conditional)) && got[at] == reference.Conditional[at] {
				at++
			}
			t.Errorf("%q is read as %d ids and the model's own code reads %d; they part at %d: %v against %v",
				reference.Prompt, len(got), len(reference.Conditional), at, got[at:min(at+8, len(got))], reference.Conditional[at:min(at+8, len(reference.Conditional))])
		}
	}
	t.Logf("%d prompts and the unconditional query: a served request reads the ids the model's own code reads", len(fixture.Cases))
}
