// Package protection verifies the repository-owned part of automation
// protection. Host branch rules and parent-harness hook activation remain
// external facts and are reported without being inferred from configuration.
package protection

// HarnessConfigDirectory prefixes every harness-owned configuration path.
const HarnessConfigDirectory = ".claude/"
