package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
)

// This identity freezes the criteria and retained requests before implementation
// changes. Whitespace does not alter identity; changing a judge or case does.
const imageVideoProtocolSHA256 = "9a270a1847acc8b478fb2b7595fb254f9d89d315bf0dfc23e6a0d97a9ceb3eb2"

func checkImageVideoProtocolIdentity(data []byte) error {
	return checkMediaProtocolIdentity(data, imageVideoProtocolSHA256)
}

func checkMediaProtocolIdentity(data []byte, digest string) error {
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	canonical, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if fmt.Sprintf("%x", sha256.Sum256(canonical)) != digest {
		return fmt.Errorf("media protocol: frozen inputs, references, or criteria changed: measured %x", sha256.Sum256(canonical))
	}
	return nil
}
