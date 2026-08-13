package inference

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/model"
)

func TestForwardUsesCompiledPolicy(t *testing.T) {
	for _, test := range []struct {
		architecture string
		message      string
	}{
		{"dflash", "requires a paired-feature session"},
		{"eagle3", "requires a feature-draft session"},
		{"gemma4-assistant", "requires a paired projection session"},
		{"t5", "requires an encoder-decoder session"},
	} {
		t.Run(test.architecture, func(t *testing.T) {
			spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: test.architecture, BlockCount: 1}}
			runner := &Runner{preparedModel: preparedModel{
				spec: spec, program: fixtureProgram(spec, model.Weights{}),
			}}
			_, err := runner.Forward(context.Background(), nil)
			if err == nil || !strings.Contains(err.Error(), test.message) {
				t.Fatalf("Forward error = %v", err)
			}
		})
	}
}
