//go:build modeltest

package evaluation

import "testing"

func TestQwenNineIFEvalScoresAcceptance(t *testing.T) {
	t.Parallel()
	checkFullNativeIFEvalScores(t,
		"evidence:sha256:4e5eb501233489e0e4c54289971d5a67f729a5cf1c1153ed5d31e3eb2bd9c00e",
		"evaluation/qwen9-ifeval-cc267157", "", "")
}
