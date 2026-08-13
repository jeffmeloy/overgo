package artifact

import "testing"

func TestOwnedContentBytesRetainsCallerBuffer(t *testing.T) {
	contract := DocumentContract{Kind: KindOutput, MediaType: "image/png", Schema: "test.encoded-image.v1"}
	data := []byte{1, 2, 3, 4}
	content, err := contract.OwnedContentBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if &content.Data[0] != &data[0] {
		t.Fatal("owned content cloned caller buffer")
	}
	if err := content.Validate(); err != nil {
		t.Fatal(err)
	}
}
