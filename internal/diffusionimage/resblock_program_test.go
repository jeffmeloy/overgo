package diffusionimage

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestResBlockProgramMatchesTinyReference(t *testing.T) {
	fixture := loadGolden[uditTinyFixture](t, "udit_tiny")
	model := compileTiny(t, fixture)
	input, height, width := conv2dValidStride(
		fixture.X, model.patch.weight, model.patch.bias,
		fixture.B, model.Cfg.InChannels, model.Cfg.BaseChannels,
		fixture.H, fixture.W, model.Cfg.PatchSize, model.Cfg.PatchSize,
	)
	block, ok := model.encoders[0].blocks[0].(*resBlock)
	if !ok {
		t.Fatal("first tiny encoder block is not residual")
	}
	program := testcheck.Must(compileResBlockProgram(block, model.Cfg, model.Cfg.BaseChannels, height, width))(t)
	got := testcheck.Must(program.execute(t.Context(), nil, input))(t)
	want := block.forward(model, input, 1, model.Cfg.BaseChannels, height, width)
	requireWithin(t, "residual block graph", got, want, tolTight)
}
