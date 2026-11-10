//go:build windows && modeltest

package main

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// parenReadings are each model's chat-template MMLU-Pro readings of the store's 1209-case share, recorded on
// 2026-11-10: the one the newest scorecard showed, after "The answer is", and the one read again at 93d2f520f after
// "The answer is (" (answer-evidence/reread-cells). Qwen3.8-27B's second reading ran under the recipe it was activated
// under since.
var parenReadings = []struct{ model, earlier, paren string }{
	{"Qwen2.5-0.5B", "evaluation:sha256:d521b0f92ad1f44c6fa90ea0c71f6704cf9624bb7c1fd286757c01449a6597a0", "evaluation:sha256:7720d8efbef1004e235f9cee04f0f2a928df7565c01102a47feafb68db989a6b"},
	{"MiniCPM5-1B", "evaluation:sha256:a8a2768801bbc5ab48614a21cd0c4c509f3a43f6d6fe76f1320d5877d37d7014", "evaluation:sha256:30cdf5167709d688bfc7ac1be7ee3fb12984e6e2cd21d29e98c2f8d3d4f74441"},
	{"Qwen3.5-4B", "evaluation:sha256:419060e3788608faa487410e3611b70767f5c946c7db3874f758883fe09af1a3", "evaluation:sha256:81716f67198ed248d9ff458ee8f59b566470389e268ba631cecd696ad7326027"},
	{"gemma-4-E4B-it", "evaluation:sha256:111974607e0d64ec8434d27bfba9d06bcf0b63c2c69fed15cf7272224076068f", "evaluation:sha256:99694231eb045b7a42b81f377778afb510c734008add4b2848034f53a3e922aa"},
	{"Qwen3.5-9B", "evaluation:sha256:0d743f977e48fb3ec5c87d2a20e376add2ec074d6146e14b87f836fa8f6bee2d", "evaluation:sha256:66d63caa209ed2de6cf57a350028e9102e332a61f0d0687bc8a0fba1f5910322"},
	{"gemma-4-12B-it-fp8-native", "evaluation:sha256:e9c00eb93f4ebc46810e9af0b0b586b01d9516908cfb16ad61622f823499c3ae", "evaluation:sha256:9f597ffa5f24c52bf37758bc8854a16b8c6dfb523263cf512994ce45b9ffc1b8"},
	{"gemma-4-12B-it-fp8", "evaluation:sha256:1b82602653cbc9a6e529e0dabd79eef5ed2a8531fa22822b4863565b1f23a423", "evaluation:sha256:532e9ab11c476e74257307e0c17fea79f4534c41c40952e2c0f28aac4f5daa33"},
	{"Qwen3.8 Flash Next", "evaluation:sha256:3c2e093c1d9651bbbc23986736f6109887fe40d971e6d349e4ae7f649f8b11ba", "evaluation:sha256:b830af809e6bcbeb96be7800f8452021918c76782638307f7dee703f773e16d0"},
	{"Bonsai-27B-Q1_0", "evaluation:sha256:62a7b2de78ff74ff015ad3d52e0caba8b461354075f747a323eecee36ef777d7", "evaluation:sha256:c37f1c5a652e2949c5a461b6099d0571d0f5d09f04d0d7f7a0efa2343ec503f0"},
	{"Qwen3.8-27B", "evaluation:sha256:255d673bbee113a35276a92083a578e6dba297d13d43be5f0ec866fb631daf51", "evaluation:sha256:fd80e029f3510e5103477b09ab9b10165d6c05869722ecb7558501c278ea2cc7"},
}

// TestTheParenOpenerLosesMoreCasesThanItGains sets each model's two readings of parenReadings side by side, case by
// case, and holds the finding that withdrew "The answer is (" (answer-evidence/earlier-opener): summed over the ten
// models it answered fewer cases right than "The answer is" did on the same cases (38 fewer on 2026-11-10, six models
// of ten fewer), so the chat-template protocol opens with "The answer is". The bar, a sum over the models below
// zero, was set after the readings: the re-ask that chose the opener measured only the answers the earlier opener had
// lost, and could not see the answers it had right.
func TestTheParenOpenerLosesMoreCasesThanItGains(t *testing.T) {
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	store := testcheck.Must(overgodb.OpenReadOnly(roots.Store))(t)
	t.Cleanup(func() { testcheck.NoError(t, store.Close()) })
	right := func(record string) map[string]bool {
		t.Helper()
		id := testcheck.Must(artifact.ParseID(record))(t)
		for {
			evaluated := testcheck.Must(runrecord.RequireEvaluation(ctx, store, id))(t)
			bound := testcheck.Must(runrecord.RequireRun(ctx, store, evaluated.Run))(t)
			if len(bound.Outputs) != 1 {
				t.Fatalf("the run of %s has %d outputs, want its report", id, len(bound.Outputs))
			}
			if receipt, err := evaluation.RequireReplayReceipt(ctx, store, bound.Outputs[0]); err == nil && receipt.Carried.Valid() {
				id = receipt.Carried
				continue
			}
			content, found, err := artifact.ReadContent(ctx, store, bound.Outputs[0])
			if err != nil || !found {
				t.Fatalf("the report of %s cannot be read: %v", id, err)
			}
			var report struct {
				Observations []struct {
					Name     string `json:"name"`
					Selected int    `json:"selected"`
					Answer   int    `json:"answer"`
				} `json:"observations"`
			}
			testcheck.NoError(t, json.Unmarshal(content.Data, &report))
			answered := make(map[string]bool, len(report.Observations))
			for _, observation := range report.Observations {
				answered[observation.Name] = observation.Selected == observation.Answer
			}
			return answered
		}
	}
	net, fewer := 0, 0
	for _, pair := range parenReadings {
		earlier, paren := right(pair.earlier), right(pair.paren)
		if len(earlier) != len(paren) {
			t.Fatalf("%s's readings answer %d and %d cases, want the same share", pair.model, len(earlier), len(paren))
		}
		lost, gained := 0, 0
		for name, wasRight := range earlier {
			nowRight, asked := paren[name]
			switch {
			case !asked:
				t.Fatalf("%s's paren reading holds no case %s", pair.model, name)
			case wasRight && !nowRight:
				lost++
			case nowRight && !wasRight:
				gained++
			}
		}
		if net += gained - lost; gained < lost {
			fewer++
		}
		t.Logf("%s: right only after \"The answer is\" %d, only after \"The answer is (\" %d", pair.model, lost, gained)
	}
	t.Logf("summed over %d models the paren opener answered %+d cases right, %d models fewer", len(parenReadings), net, fewer)
	if net >= 0 {
		t.Errorf("summed over the models the paren opener answered %+d cases right, want fewer than the earlier opener", net)
	}
	if words, _ := evaluation.ChoiceProtocol(evaluation.PromptingChatTemplate.Label()); !strings.Contains(words, `"The answer is"`) {
		t.Errorf("the chat-template protocol is said as %q, want it to open with \"The answer is\"", words)
	}
}
