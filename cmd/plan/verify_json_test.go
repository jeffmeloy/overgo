package main

import (
	"fmt"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testevidence"
)

func TestVerifyGoTestJSONTargets(t *testing.T) {
	t.Parallel()
	out := fmt.Sprintf(
		"{\"Action\":\"pass\",\"Package\":\"overgo/x\",\"Test\":\"TestAcceptance\"}\n"+
			"{\"Action\":\"output\",\"Package\":\"overgo/x\",\"Test\":\"TestHardware\",\"Output\":%q}\n"+
			"{\"Action\":\"skip\",\"Package\":\"overgo/x\",\"Test\":\"TestHardware\"}\n"+
			"{\"Action\":\"pass\",\"Package\":\"overgo/x\"}\n",
		"UNAVAILABLE: GPU fixture\n",
	)
	testcheck.NoError(t, testevidence.VerifyGoTestEvidence("go test ./x -run '^TestAcceptance$'", out))
	if err := testevidence.VerifyGoTestEvidence("go test ./x -run '^TestHardware$'", out); err == nil {
		t.Fatal("targeted skipped hardware test passed")
	}
	if err := testevidence.VerifyGoTestEvidence("go test ./x", out); err == nil {
		t.Fatal("broad acceptance credited a skipped test")
	}
}
