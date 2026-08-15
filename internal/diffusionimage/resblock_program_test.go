package diffusionimage

import (
	"context"
	"testing"
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
	program, err := compileResBlockProgram(block, model.Cfg, model.Cfg.BaseChannels, height, width)
	if err != nil {
		t.Fatal(err)
	}
	got, err := program.execute(context.Background(), nil, input)
	if err != nil {
		t.Fatal(err)
	}
	want := block.forward(model, input, 1, model.Cfg.BaseChannels, height, width)
	requireWithin(t, "residual block graph", got, want, tolTight)
}
