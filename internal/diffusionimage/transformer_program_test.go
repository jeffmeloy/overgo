package diffusionimage

import (
	"testing"
)

func TestTransformerBlockProgramMatchesTorchFixture(t *testing.T) {
	fixture := loadGolden[transformerBlockFixture](t, "transformerblock")
	program, err := compileTransformerBlockProgram(fixture.block(), fixture.model().Cfg, fixture.C, fixture.H*fixture.W)
	if err != nil {
		t.Fatal(err)
	}
	got, err := program.execute(t.Context(), nil, fixture.X)
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "transformer graph", got, fixture.Y, tolLoose)
}
