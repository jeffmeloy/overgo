package server

import (
	"strings"
	"testing"
)

// webuiSourceClass is why a test reads the served UI source instead of
// driving the page: a ratchet (a count ceiling, a census or a one-owner
// rule), a ban (a pattern the source must not hold), a contract (how the
// server serves the page), or pending the named plan row whose browser leg
// replaces it.
type webuiSourceClass string

const (
	sourceRatchet  webuiSourceClass = "ratchet"
	sourceBan      webuiSourceClass = "ban"
	sourceContract webuiSourceClass = "contract"
	pendingPrefix                   = "pending:"
)

// pendingLeg names the plan row whose browser leg replaces a needle test.
func pendingLeg(row string) webuiSourceClass {
	return webuiSourceClass(pendingPrefix + row)
}

// webuiPendingCeiling bounds the needle tests that still stand in for a
// browser leg; each leg row that lands lowers it.
const webuiPendingCeiling = 4

// webuiSourceReaders declares every server test that reads the served UI
// source; internal/gate's census holds the declaration to the tests.
var webuiSourceReaders = map[string]webuiSourceClass{
	"TestWebUIComposerBudget":               sourceRatchet,
	"TestWebUIReviewRatchet":                sourceRatchet,
	"TestArtifactContentURLHasOneOwner":     sourceRatchet,
	"TestWebUIClientHelpersHaveOneOwner":    sourceRatchet,
	"TestWebUIRouteTable":                   sourceRatchet,
	"TestAgentRoutesHaveOnePrefix":          sourceRatchet,
	"TestWebUIClientDedup":                  sourceRatchet,
	"TestWebUIStyleInvariants":              sourceRatchet,
	"TestGUITrainingAndExportShareServices": sourceRatchet,

	"TestWebUIOneShell":                         sourceContract,
	"TestWebUIServesEmbeddedAssets":             sourceContract,
	"TestWebUIContentSecurityPolicy":            sourceContract,
	"TestWebUIAssetsPublicWhenAPIKeyConfigured": sourceContract,
	"TestWebUIUnknownPathReturns404":            sourceContract,
	"TestWebUIDevelopmentDirectory":             sourceContract,
	"TestIdleShellAnswersWhileNothingServes":    sourceContract,
	"TestGUIMobileDeviceEvidence":               sourceContract,
	"TestGUIMobileEvidenceContract":             sourceContract,

	"TestFrontPageConversations":                                  sourceBan,
	"TestWorkspaceCapabilityDocument":                             sourceBan,
	"TestArtifactGalleryProjectedPageIsBoundedAndStreamsPayloads": sourceBan,
	"TestGenerationWorkspaceUsesRecipeCapabilities":               sourceBan,
	"TestGlobalOperationShell":                                    sourceBan,
	"TestRuntimeActivitySessionLedgerGUI":                         sourceBan,
	"TestWebUITabsLoadThroughOneResource":                         sourceBan,
	"TestWebUIWorkspaceRoutes":                                    sourceBan,
	"TestWebUIRuntimeMonitor":                                     sourceBan,
	"TestAgentGUIUsesProjectedQueriesAndSSE":                      sourceBan,
	"TestWebUIChatUsesServerContextAndTiming":                     sourceBan,
	"TestWebUIChatMarkdown":                                       sourceBan,
	"TestRLWorkspaceUsesGenericWorkflowEndpoints":                 sourceBan,
	"TestPeerWorkspaceUsesGlobalOperations":                       sourceBan,

	"TestCompositionGUIWorkflow":                      pendingLeg("gui-legs-compositions"),
	"TestActiveRecipeInspectorUsesCompiledOrder":      sourceBan,
	"TestEvaluationWorkbenchUsesDeclaredCapabilities": sourceBan,
	"TestRLWorkspaceRendersMeasuredEvidence":          sourceBan,

	"TestAgentWorkspaceNoHiddenReasoning": sourceBan,

	"TestFrontPageModelSwitch":            sourceRatchet,
	"TestFrontPageAgent":                  pendingLeg("gui-legs-agent-thread"),
	"TestFrontPageGenerationDeclarations": sourceBan,
	"TestFrontPageLibrary":                pendingLeg("gui-legs-library"),
	"TestFrontPageRemoteTurns":            pendingLeg("gui-legs-remote"),
}

// TestWebUITestsDriveBehaviourNotSource holds the tests that read the
// served UI source to a declared reason: each is a ratchet, a ban, a serving
// contract, or pending the plan row whose browser leg replaces it, and the
// pending needles never grow past their ceiling. internal/gate's
// TestServerTestsDeclareEveryWebUISourceRead holds the declaration to the
// tests themselves and each pending row to the open plan.
func TestWebUITestsDriveBehaviourNotSource(t *testing.T) {
	t.Parallel()
	pending := 0
	for test, class := range webuiSourceReaders {
		switch row, isPending := strings.CutPrefix(string(class), pendingPrefix); {
		case isPending:
			pending++
			if row == "" {
				t.Errorf("%s is pending no plan row", test)
			}
		case class != sourceRatchet && class != sourceBan && class != sourceContract:
			t.Errorf("%s reads the served source for an undeclared reason %q", test, class)
		}
	}
	if pending > webuiPendingCeiling {
		t.Errorf("%d tests stand in for a browser leg, above the ceiling %d; drive the page instead", pending, webuiPendingCeiling)
	}
}
