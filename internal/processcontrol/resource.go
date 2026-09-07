package processcontrol

import (
	"encoding/json"
	"errors"
	"maps"
	"os"
	"slices"
	"strings"
	"sync"
)

var resourceMu sync.Mutex
var claimedResources = map[string]bool{}

const inheritedResourcesEnvironment = "OVERGO_PROCESS_RESOURCES"

// ClaimResource reserves a physical resource for this process and its supervised
// children until they exit. Names must identify hardware, independent of stores
// and worktrees. A child may reenter its inherited reservation. Contention fails
// immediately; heartbeat age grants no authority. Unsupported platforms refuse.
func ClaimResource(name string) error {
	if name == "" || strings.TrimSpace(name) != name || strings.ContainsRune(name, 0) {
		return errors.New("processcontrol: invalid physical resource name")
	}
	resourceMu.Lock()
	defer resourceMu.Unlock()
	if err := claimResource(name); err != nil {
		return err
	}
	claimedResources[name] = true
	return nil
}

// inheritedResources reacquires handles before launching further descendants.
// Names carry identity only; the OS still proves membership or exclusivity.
func inheritedResources() error {
	value := os.Getenv(inheritedResourcesEnvironment)
	if value == "" {
		return nil
	}
	var names []string
	if err := json.Unmarshal([]byte(value), &names); err != nil {
		return err
	}
	for _, name := range names {
		if err := ClaimResource(name); err != nil {
			return err
		}
	}
	return nil
}

// resourceEnvironment runs under resourceMu; a caller cannot drop reservations
// by replacing its command environment. Empty identity never grants a claim.
func resourceEnvironment(environment []string) []string {
	if len(claimedResources) == 0 {
		return environment
	}
	names := slices.Sorted(maps.Keys(claimedResources))
	payload, _ := json.Marshal(names)
	return append(slices.Clone(environment), inheritedResourcesEnvironment+"="+string(payload))
}
