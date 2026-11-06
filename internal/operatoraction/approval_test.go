package operatoraction_test

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestDecisionBindsExactAction(t *testing.T) {
	t.Parallel()
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "operation")
	recipe := testutil.ArtifactID(t, artifact.KindRecipe, "recipe")
	action := operatoraction.Action{Code: "publish", Summary: "Publish output", Argv: []string{"overgo", "publish", "--recipe", recipe.String()}}
	request := testcheck.Must(operatoraction.NewApprovalRequest(operation, recipe, action, artifact.ID{}))(t)
	if !request.Binds(action) {
		t.Fatal("exact action was not bound")
	}
	substituted := action
	substituted.Argv = append([]string(nil), action.Argv...)
	substituted.Argv[len(substituted.Argv)-1] = operation.String()
	if request.Binds(substituted) {
		t.Fatal("substituted action was bound")
	}
}
