package plan

import "testing"

func testCompletionAuthority(t *testing.T, document Plan, completed ...string) CompletionAuthority {
	return testCompletionAuthorityAt(t, document, "test-repository", "test-revision", completed...)
}

func testCompletionAuthorityAt(
	t *testing.T,
	document Plan,
	repository, revision string,
	completed ...string,
) CompletionAuthority {
	t.Helper()
	digest, err := completionPlanDigest(document)
	if err != nil {
		t.Fatal(err)
	}
	authority := CompletionAuthority{
		resolved:            true,
		planDigest:          digest,
		repository:          repository,
		revision:            revision,
		completedReferences: make(map[string]completionEvidence, len(completed)),
		retiredItems:        make(map[string]completionEvidence),
	}
	for _, reference := range completed {
		authority.completedReferences[reference] = completionEvidence{}
	}
	return authority
}
