package main

import "testing"

// TestOnlyTheCellsThatReadTimesTakeTheQuietMachine holds which cells a cycle runs with no
// landing beside them: the serving and capability measurements and the proofs that train,
// whose first output, throughput and load a build or a test beside them slows, and not the
// benchmarks, the guards or the lanes, whose results a busy machine does not change, so a
// landing that reaches no device lane is not delayed by a quality cell.
func TestOnlyTheCellsThatReadTimesTakeTheQuietMachine(t *testing.T) {
	t.Parallel()
	for validation, timed := range map[string]bool{
		servingValidation: true, trainingProof: true, projectionProof: true,
		"image-proof": true, "video-proof": true, "speech-proof": true, "vqa-proof": true, "specialized-proof": true,
		guardValidation: false, "mmlu-pro": false, "bbh": false, "ifeval": false, "mmlu": false,
	} {
		if got := timedValidation(validation); got != timed {
			t.Errorf("%s: timed = %t, want %t", validation, got, timed)
		}
	}
}
