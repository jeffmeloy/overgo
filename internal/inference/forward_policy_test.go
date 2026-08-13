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
		{"dflash", "requires feature fusion"},
		{"eagle3", "requires NewEagle3Session"},
		{"gemma4-assistant", "requires a paired projection session"},
		{"t5", "requires NewT5Session"},
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
