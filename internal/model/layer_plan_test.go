package model

import "testing"

func TestPlanLayerDerivesExecutionPolicy(t *testing.T) {
	tests := []struct {
		name      string
		spec      Spec
		recurrent bool
		attention AttentionPolicy
		extent    CacheExtent
	}{
		{
			name:      "llama",
			spec:      Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 1}},
			attention: AttentionStandard, extent: CacheExtentToken,
		},
		{
			name:      "dsa",
			spec:      Spec{CommonSpec: CommonSpec{Architecture: "deepseek32", BlockCount: 1}},
			attention: AttentionDSA, extent: CacheExtentToken,
		},
		{
			name: "qwen-gdn",
			spec: Spec{
				CommonSpec:    CommonSpec{Architecture: "qwen35", BlockCount: 1},
				RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true}},
			},
			attention: AttentionQwenGDN, extent: CacheExtentFixed,
		},
		{
			name:      "lfm2",
			spec:      Spec{CommonSpec: CommonSpec{Architecture: "lfm2", BlockCount: 1}},
			recurrent: true, attention: AttentionLFM2, extent: CacheExtentFixed,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan := test.spec.PlanLayer(0, test.recurrent)
			if plan.Attention != test.attention || plan.CacheExtent != test.extent {
				t.Fatalf("plan = %+v", plan)
			}
		})
	}
}
