package latentimage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// promptStringsGolden is ComfyUI's own nodes run on fixed texts (scripts/python_golden_references/
// comfy_prompt_strings_golden.py): mode 2's chain on texts that take every branch, each rule alone on texts at its
// edges, and mode 3's layout node on fixed boxes.
type promptStringsGolden struct {
	Cases []struct {
		Inputs struct {
			Source, Exclusions, First, Retry, Negative string
		} `json:"inputs"`
		Outputs struct {
			PositiveRequest string `json:"positive_request"`
			LongEnough      bool   `json:"long_enough"`
			RetryRequest    string `json:"retry_request"`
			Positive        string `json:"positive"`
			NegativeRequest string `json:"negative_request"`
			Photo           bool   `json:"photo"`
			Extracted       string `json:"extracted"`
			Deduplicated    string `json:"deduplicated"`
			Guarded         string `json:"guarded"`
			Trimmed         string `json:"trimmed"`
			None            bool   `json:"none"`
			Negative        string `json:"negative"`
		} `json:"outputs"`
	} `json:"cases"`
	Rules map[string][]struct {
		Text   string          `json:"text"`
		Result json.RawMessage `json:"result"`
	} `json:"rules"`
	Layouts []struct {
		Prompt     string      `json:"prompt"`
		Elements   []LayoutBox `json:"elements"`
		Width      int         `json:"width"`
		Height     int         `json:"height"`
		Characters string      `json:"characters"`
		Layout     string      `json:"layout"`
	} `json:"layouts"`
}

func TestPromptModeStringsEqualComfyUIsNodes(t *testing.T) {
	raw := testcheck.Must(os.ReadFile(filepath.Join("testdata", "moss_prompt_strings", "golden.json")))(t)
	var golden promptStringsGolden
	testcheck.NoError(t, json.Unmarshal(raw, &golden))
	enhancer := testcheck.Must(mossEnhancer())(t)
	if len(golden.Cases) == 0 || len(golden.Rules) == 0 || len(golden.Layouts) == 0 {
		t.Fatalf("the golden holds %d cases, %d rules and %d layouts, none may be missing", len(golden.Cases), len(golden.Rules), len(golden.Layouts))
	}
	for index, c := range golden.Cases {
		in, want := c.Inputs, c.Outputs
		got := map[string][2]any{
			"positive_request": {enhancerFormat(enhancer.PositiveRequest, in.Source, in.Exclusions, ""), want.PositiveRequest},
			"long_enough":      {hasEnhancerWords(in.First), want.LongEnough},
			"retry_request":    {enhancerFormat(enhancer.RetryRequest, in.Source, in.Exclusions, in.First), want.RetryRequest},
			"positive":         {enhancedPositive(in.Source, in.First, in.Retry), want.Positive},
			"negative_request": {enhancerFormat(enhancer.NegativeRequest, in.Source, want.Positive, in.Exclusions), want.NegativeRequest},
			"photo":            {photoSource(in.Source), want.Photo},
			"extracted":        {firstPhrases(in.Negative, enhancerPhrases), want.Extracted},
			"deduplicated":     {withoutDuplicates(want.Extracted), want.Deduplicated},
			"guarded":          {map[bool]func(string) string{true: guardPhoto, false: guardProtected}[want.Photo](want.Deduplicated), want.Guarded},
			"trimmed":          {trimSeparators(want.Guarded), want.Trimmed},
			"none":             {isNoNegative(want.Trimmed), want.None},
			"negative":         {enhancedNegative(in.Negative, in.Source, in.Exclusions, enhancer.NegativeBackup), want.Negative},
		}
		for name, pair := range got {
			if pair[0] != pair[1] {
				t.Errorf("case %d's %s = %q, ComfyUI's %q", index, name, pair[0], pair[1])
			}
		}
	}
	rules := map[string]func(string) any{
		"451": func(s string) any { return hasEnhancerWords(s) },
		"469": func(s string) any { return isNoNegative(s) },
		"472": func(s string) any { return firstPhrases(s, enhancerPhrases) },
		"473": func(s string) any { return withoutDuplicates(s) },
		"474": func(s string) any { return photoSource(s) },
		"477": func(s string) any { return guardPhoto(s) },
		"478": func(s string) any { return guardProtected(s) },
		"480": func(s string) any { return trimSeparators(s) },
	}
	for node, probes := range golden.Rules {
		rule, ok := rules[node]
		if !ok {
			t.Errorf("the golden probes node %s, which no rule here ports", node)
			continue
		}
		for _, probe := range probes {
			var want any
			testcheck.NoError(t, json.Unmarshal(probe.Result, &want))
			if got := rule(probe.Text); got != want {
				t.Errorf("node %s on %q = %#v, ComfyUI's %#v", node, probe.Text, got, want)
			}
		}
	}
	for index, layout := range golden.Layouts {
		if got := layoutPrompt(layout.Prompt, LayoutBoxes{Boxes: layout.Elements, Characters: layout.Characters}, layout.Width, layout.Height); got != layout.Layout {
			t.Errorf("layout %d = %s, ComfyUI's %s", index, got, layout.Layout)
		}
	}
}
