package main

import (
	"fmt"
	"testing"

	"overgo/internal/testevidence"
)

func TestVerifyGoTestJSONTargets(t *testing.T) {
	out := fmt.Sprintf(
		"{\"Action\":\"pass\",\"Package\":\"overgo/x\",\"Test\":\"TestAcceptance\"}\n"+
			"{\"Action\":\"output\",\"Package\":\"overgo/x\",\"Test\":\"TestHardware\",\"Output\":%q}\n"+
			"{\"Action\":\"skip\",\"Package\":\"overgo/x\",\"Test\":\"TestHardware\"}\n"+
			"{\"Action\":\"pass\",\"Package\":\"overgo/x\"}\n",
		"UNAVAILABLE: GPU fixture\n",
	)
	if err := testevidence.VerifyGoTestTarget("go test ./x -run '^TestAcceptance$'", out); err != nil {
		t.Fatal(err)
	}
	if err := testevidence.VerifyGoTestTarget("go test ./x -run '^TestHardware$'", out); err == nil {
		t.Fatal("targeted skipped hardware test passed")
	}
	if err := testevidence.VerifyGoTestTarget("go test ./x", out); err == nil {
		t.Fatal("unnamed acceptance target passed")
	}
}
