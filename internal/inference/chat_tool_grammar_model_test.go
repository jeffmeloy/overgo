//go:build modeltest

package inference

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modeltest"
	"overgo/internal/testcheck"
)

func TestNativeToolGrammarsCompileForLocalTemplateFamilies(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	for _, fixture := range []struct {
		name  string
		model string
	}{
		{name: "qwen-json", model: qwen3Fixture},
		{name: "qwen35-hermes", model: qwen35Fixture},
		{name: "bonsai-hermes", model: bonsaiFixture},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			runner, err := openFixtureRunner(modeltest.Directory(t, fixture.model), 0)
			testcheck.NoError(t, err)
			defer runner.Close()
			source, root, patterns, err := runner.ChatToolGrammar(
				[]ChatTool{toolWeatherDefinition()},
				false,
				true,
				true,
			)
			testcheck.NoError(t, err)
			if _, err := runner.CompileLazyGBNF(
				source,
				root,
				patterns,
				nil,
			); err != nil {
				t.Fatal(err)
			}
			source, root, patterns, err = runner.ChatToolGrammar(
				[]ChatTool{toolWeatherDefinition()},
				true,
				true,
				true,
			)
			testcheck.NoError(t, err)
			if len(patterns) != 0 {
				t.Fatalf("required grammar triggers = %q", patterns)
			}
			if _, err := runner.CompileGBNF(source, root); err != nil {
				t.Fatal(err)
			}
		})
	}
}
