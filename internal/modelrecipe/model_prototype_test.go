package modelrecipe

import (
	"bytes"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/testutil"
)

func TestModelPrototypeIdentityAndValidation(t *testing.T) {
	names := model.SupportedArchitectures()
	if len(names) < 2 {
		t.Fatal("model prototype test requires two compiled architectures")
	}
	profile := registeredProfileDocument(t, names[0])
	prototype, err := NewModelPrototype(profile)
	if err != nil {
		t.Fatal(err)
	}
	if prototype.ID().Kind() != artifact.KindRecipe || prototype.Architecture() != names[0] ||
		prototype.Profile() != profile.ID || prototype.ValidateIdentity() != nil {
		t.Fatal("model prototype does not bind the exact compiled profile")
	}

	second, err := NewModelPrototype(profile)
	if err != nil {
		t.Fatal(err)
	}
	if second.ID() != prototype.ID() {
		t.Fatal("identical compiled profile produced unstable prototype identity")
	}
	content, err := prototype.Content()
	if err != nil {
		t.Fatal(err)
	}
	if content.Descriptor.MediaType != ModelPrototypeMediaType || content.Descriptor.Schema != ModelPrototypeSchema {
		t.Fatalf("unexpected prototype contract: %+v", content.Descriptor)
	}
	for _, forbidden := range [][]byte{[]byte("status"), []byte("runtime"), []byte("weights"), []byte("authority"), []byte("activation")} {
		if bytes.Contains(content.Data, forbidden) {
			t.Fatalf("prototype copied lifecycle or runtime field %q", forbidden)
		}
	}
	parsedDocument, err := modelPrototypeCodec.Parse(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	parsed := ModelPrototype{document: parsedDocument}
	if parsed.ID() != prototype.ID() || parsed.ValidateIdentity() != nil {
		t.Fatal("model prototype did not round-trip with exact identity")
	}
	lineage := prototype.Lineage()
	if len(lineage) != 1 || lineage[0].Child != prototype.ID() || lineage[0].Parent != profile.ID ||
		lineage[0].Relation != artifact.RelationDependsOn {
		t.Fatalf("unexpected model prototype lineage: %+v", lineage)
	}

	otherProfile := registeredProfileDocument(t, names[1])
	invalidDocuments := map[string]modelPrototypeDocument{
		"unknown architecture": {
			Version: artifact.InitialDocumentVersion, Architecture: "not-compiled", Profile: profile.ID,
		},
		"stale profile": {
			Version: artifact.InitialDocumentVersion, Architecture: names[0], Profile: otherProfile.ID,
		},
		"wrong profile kind": {
			Version: artifact.InitialDocumentVersion, Architecture: names[0],
			Profile: testutil.ArtifactID(t, artifact.KindRecipe, "wrong profile kind"),
		},
	}
	for name, document := range invalidDocuments {
		t.Run(name, func(t *testing.T) {
			data, err := json.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := modelPrototypeCodec.Parse(data); err == nil {
				t.Fatal("invalid model prototype accepted")
			}
		})
	}

	unknownField := append(bytes.TrimSuffix(content.Data, []byte("}")), []byte(",\"status\":\"active\"}")...)
	if _, err := modelPrototypeCodec.Parse(unknownField); err == nil {
		t.Fatal("prototype accepted lifecycle authority outside its schema")
	}

	tamperedProfile := profile
	tamperedProfile.Architecture = names[1]
	if _, err := NewModelPrototype(tamperedProfile); err == nil {
		t.Fatal("prototype accepted a forged profile document")
	}
	forged := prototype
	forged.document.ID = testutil.ArtifactID(t, artifact.KindRecipe, "forged prototype")
	if forged.ValidateIdentity() == nil {
		t.Fatal("forged prototype identity accepted")
	}
}

func registeredProfileDocument(t testing.TB, name string) ProfileDocument {
	t.Helper()
	profile, found := model.LookupArchitecture(name)
	if !found {
		t.Fatalf("compiled architecture %q absent", name)
	}
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	return document
}
