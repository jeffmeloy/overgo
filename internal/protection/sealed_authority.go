package protection

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	SealedWriteVersion   uint16 = 1
	SealedWriteMediaType        = "application/vnd.overgo.sealed-authority-write+json"
	SealedWriteSchema           = "overgo/sealed-authority-write/v1"
	sealedWritePath             = "/v1/sealed"
	sealedNameBytes             = 512
)

type SealedKind string

const (
	SealedGolden          SealedKind = "golden"
	SealedEvaluator       SealedKind = "evaluator"
	SealedPromotionPolicy SealedKind = "promotion-policy"
	SealedChampionAlias   SealedKind = "champion-alias"
)

type ExternalProvenance struct {
	Tool    string      `json:"tool"`
	Version string      `json:"version"`
	Output  artifact.ID `json:"output"`
}

type SealedWriteSpec struct {
	Kind       SealedKind          `json:"kind"`
	Name       string              `json:"name"`
	Value      artifact.ID         `json:"value"`
	Previous   *artifact.ID        `json:"previous,omitempty"`
	Provenance *ExternalProvenance `json:"provenance,omitempty"`
}

type SealedWrite struct {
	Version   uint16 `json:"version"`
	Principal string `json:"principal"`
	SealedWriteSpec
	Signature []byte      `json:"signature"`
	ID        artifact.ID `json:"-"`
}

var sealedWriteContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: SealedWriteMediaType, Schema: SealedWriteSchema,
}

func SignSealedWrite(private ed25519.PrivateKey, spec SealedWriteSpec) (SealedWrite, error) {
	if len(private) != ed25519.PrivateKeySize {
		return SealedWrite{}, errors.New("sealed authority: invalid private key")
	}
	public := private.Public().(ed25519.PublicKey)
	write := SealedWrite{Version: SealedWriteVersion, Principal: sealedPrincipal(public), SealedWriteSpec: spec}
	payload, err := write.signingBytes()
	if err != nil {
		return SealedWrite{}, err
	}
	write.Signature = ed25519.Sign(private, payload)
	content, err := write.Content()
	if err != nil {
		return SealedWrite{}, err
	}
	write.ID = content.Descriptor.ID
	return write, nil
}

func (w SealedWrite) Content() (artifact.Content, error) {
	if err := w.validate(); err != nil {
		return artifact.Content{}, err
	}
	data, err := json.Marshal(struct {
		Version   uint16 `json:"version"`
		Principal string `json:"principal"`
		SealedWriteSpec
		Signature []byte `json:"signature"`
	}{w.Version, w.Principal, w.SealedWriteSpec, w.Signature})
	if err != nil {
		return artifact.Content{}, err
	}
	return sealedWriteContract.ContentBytes(data)
}

func (w SealedWrite) signingBytes() ([]byte, error) {
	if err := w.validateSpec(); err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Version   uint16 `json:"version"`
		Principal string `json:"principal"`
		SealedWriteSpec
	}{w.Version, w.Principal, w.SealedWriteSpec})
}

func (w SealedWrite) validate() error {
	if len(w.Signature) != ed25519.SignatureSize {
		return errors.New("sealed authority: invalid signature")
	}
	return w.validateSpec()
}

func (w SealedWrite) validateSpec() error {
	if w.Version != SealedWriteVersion || w.Principal == "" || !w.Value.Valid() || len(w.Name) > sealedNameBytes ||
		strings.TrimSpace(w.Name) == "" || w.Name != strings.TrimSpace(w.Name) || strings.ContainsAny(w.Name, "\x00\r\n") {
		return errors.New("sealed authority: invalid write")
	}
	switch w.Kind {
	case SealedGolden, SealedEvaluator, SealedPromotionPolicy, SealedChampionAlias:
	default:
		return errors.New("sealed authority: invalid protected kind")
	}
	if w.Previous != nil && w.Previous.Kind() != artifact.KindEvidence {
		return errors.New("sealed authority: previous binding differs")
	}
	if w.Kind == SealedGolden {
		if w.Provenance == nil || strings.TrimSpace(w.Provenance.Tool) == "" || strings.TrimSpace(w.Provenance.Version) == "" ||
			w.Provenance.Tool != strings.TrimSpace(w.Provenance.Tool) || w.Provenance.Version != strings.TrimSpace(w.Provenance.Version) ||
			strings.ContainsAny(w.Provenance.Tool, "\x00\r\n") || strings.ContainsAny(w.Provenance.Version, "\x00\r\n") ||
			!w.Provenance.Output.Valid() {
			return errors.New("sealed authority: golden requires external tool/version/output provenance")
		}
	}
	return nil
}

type SealedAuthority struct {
	public          ed25519.PublicKey
	principal       string
	store           artifact.Repository
	maxRequestBytes int64
}

type SealedAuthorityConfig struct {
	MaxRequestBytes int64
}

func NewSealedAuthority(public ed25519.PublicKey, store artifact.Repository, config SealedAuthorityConfig) (*SealedAuthority, error) {
	if len(public) != ed25519.PublicKeySize || store == nil || config.MaxRequestBytes <= 0 || config.MaxRequestBytes == math.MaxInt64 {
		return nil, errors.New("sealed authority: public key, store, and request bound required")
	}
	key := append(ed25519.PublicKey(nil), public...)
	return &SealedAuthority{public: key, principal: sealedPrincipal(key), store: store, maxRequestBytes: config.MaxRequestBytes}, nil
}

func (s *SealedAuthority) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost || request.URL.Path != sealedWritePath {
		http.Error(response, "not found", http.StatusNotFound)
		return
	}
	data, err := io.ReadAll(io.LimitReader(request.Body, s.maxRequestBytes+1))
	if err != nil || int64(len(data)) > s.maxRequestBytes {
		http.Error(response, "invalid request", http.StatusBadRequest)
		return
	}
	var write SealedWrite
	if err := strictjson.DecodeBytes(data, &write); err != nil || write.Principal != s.principal {
		http.Error(response, "authority refused", http.StatusForbidden)
		return
	}
	payload, err := write.signingBytes()
	if err != nil || !ed25519.Verify(s.public, payload, write.Signature) {
		http.Error(response, "authority refused", http.StatusForbidden)
		return
	}
	content, err := write.Content()
	if err != nil {
		http.Error(response, "invalid write", http.StatusBadRequest)
		return
	}
	alias := "sealed/" + string(write.Kind) + "/" + write.Name
	parents := []artifact.ID{write.Value}
	if write.Provenance != nil {
		parents = append(parents, write.Provenance.Output)
	}
	descriptors := make([]artifact.Descriptor, 0, len(parents))
	seen := map[artifact.ID]bool{}
	for _, parent := range parents {
		if !seen[parent] {
			descriptors = append(descriptors, artifact.Descriptor{ID: parent})
			seen[parent] = true
		}
	}
	batch := artifact.Batch{
		Key: "sealed-authority/" + content.Descriptor.ID.String(), Artifacts: descriptors,
		Contents: []artifact.Content{content}, Lineage: artifact.DependencyLineage(content.Descriptor.ID, parents...),
		Aliases: []artifact.AliasBinding{{Name: alias, Target: content.Descriptor.ID, Previous: write.Previous}},
	}
	err = batch.Validate()
	if err == nil {
		_, err = artifact.CommitBatch(request.Context(), s.store, batch)
	}
	if err != nil {
		http.Error(response, "write conflict", http.StatusConflict)
		return
	}
	response.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(response).Encode(struct {
		ID artifact.ID `json:"id"`
	}{content.Descriptor.ID})
}

func PublishSealed(ctx context.Context, client *http.Client, endpoint string, write SealedWrite) (artifact.ID, error) {
	if ctx == nil || client == nil {
		return artifact.ID{}, errors.New("sealed authority: client absent")
	}
	data, err := json.Marshal(write)
	if err != nil {
		return artifact.ID{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(endpoint, "/")+sealedWritePath, strings.NewReader(string(data)))
	if err != nil {
		return artifact.ID{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return artifact.ID{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return artifact.ID{}, fmt.Errorf("sealed authority: write refused with status %d", response.StatusCode)
	}
	var result struct {
		ID artifact.ID `json:"id"`
	}
	if err := strictjson.Decode(response.Body, &result); err != nil || result.ID.Kind() != artifact.KindEvidence {
		return artifact.ID{}, errors.Join(err, errors.New("sealed authority: invalid response"))
	}
	return result.ID, nil
}

func sealedPrincipal(public ed25519.PublicKey) string {
	digest := sha256.Sum256(public)
	return "service:ed25519:" + hex.EncodeToString(digest[:])
}
