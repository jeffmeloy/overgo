package loop

import (
	"crypto/sha256"
	"fmt"
	"strings"
)

// StrategyEnvironment names the environment variable carrying the
// declared strategy identity from the driver to the worker's gate
// runs, so every attempt record states which strategy produced it.
const StrategyEnvironment = "OVERGO_STRATEGY"

// strategyDigestLength bounds the derived identity: 12 hex characters
// of the worker-command digest distinguish strategies without carrying
// the command line itself into every record.
const strategyDigestLength = 12

// StrategyIdentity derives the strategy a worker configuration
// declares: the explicit name when the config states one, otherwise a
// stable digest of the exact worker command line. Two configurations
// share an identity exactly when they run the same command.
func StrategyIdentity(worker []string, declared string) string {
	if name := strings.TrimSpace(declared); name != "" {
		return name
	}
	if len(worker) == 0 {
		return ""
	}
	digest := sha256.Sum256([]byte(strings.Join(worker, "\x00")))
	return "worker-" + fmt.Sprintf("%x", digest)[:strategyDigestLength]
}
