package inference

import (
	"testing"

	"overgo/internal/model"
)

func TestExactAttentionCaptureRejectsModifiedScorePolicies(t *testing.T) {
	base := model.LayerPlan{HasKV: true, AttentionGraph: model.AttentionGraphPlan{Causal: true}}
	if !base.ExactAttentionReplay() {
		t.Fatal("plain causal attention rejected")
	}
	for name, graph := range map[string]model.AttentionGraphPlan{
		"noncausal": {Causal: false},
		"sinks":     {Causal: true, UseSinks: true},
		"window":    {Causal: true, Window: 128},
		"softcap":   {Causal: true, Softcap: 30},
		"alibi":     {Causal: true, MaxALiBiBias: 8},
	} {
		t.Run(name, func(t *testing.T) {
			if (model.LayerPlan{HasKV: true, AttentionGraph: graph}).ExactAttentionReplay() {
				t.Fatal("modified score policy accepted")
			}
		})
	}
}
