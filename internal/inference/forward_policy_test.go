package inference

import (
	"context"
	"strings"
	"testing"

	"llamacpp2go/internal/model"
)

func TestForwardUsesCompiledPolicy(t *testing.T) {
	for _, test := range []struct {
		architecture string
		message      string
	}{
		{"dflash", "requires feature fusion"},
		{"eagle3", "requires NewEagle3Session"},
		{"gemma4-assistant", "requires NewGemma4AssistantSession"},
		{"t5", "requires NewT5Session"},
	} {
		t.Run(test.architecture, func(t *testing.T) {
			profile, ok := model.LookupArchitecture(test.architecture)
			if !ok {
				t.Fatal("architecture is not registered")
			}
			runner := &Runner{preparedModel: preparedModel{
				spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: test.architecture}},
				plan: model.ModelPlan{Profile: profile},
			}}
			_, err := runner.Forward(context.Background(), nil)
			if err == nil || !strings.Contains(err.Error(), test.message) {
				t.Fatalf("Forward error = %v", err)
			}
		})
	}
}
