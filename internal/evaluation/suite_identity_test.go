package evaluation

import (
	"fmt"
	"testing"
)

// pinnedSuite is a suite document of one kind and what CompileSuite bound for it.
//   - descriptor = the compiled suite's kind, source and case count
//   - identities = the plan, its comparison, the dataset, the split, the case profile and the scorer, in that order
type pinnedSuite struct {
	document   string
	descriptor string
	identities [6]string
}

// TestSuiteKindsBindTheirPinnedIdentities holds each suite kind's compile and bind to the identities it bound for a fixture document.
//   - each document compiles through CompileSuite under ListingAuthorities, so the kind's compiler and its case of compileSuite are both held
//   - multiple choice and grouped choice are left out because TestEvaluationProtocolCapturesAllPolicyInputs and TestMMLUProIsAGroupedChoiceWithItsIdentitiesUnchanged pin them
//   - a speech round trip counted one way is a second fixture because its dataset is the texts with their normalization
//   - a changed identity is refused because stored plans, campaigns and answers name the identities
func TestSuiteKindsBindTheirPinnedIdentities(t *testing.T) {
	t.Parallel()
	for label, pinned := range map[string]pinnedSuite{
		"exact-generation": {document: `{"schema":"fixture/v1","source":"fixture","cases":[
			{"name":"one","prompt":"Say ok.","max_tokens":4,"text":"ok","prompt_tokens":3,"generated_tokens":2},
			{"name":"two","prompt":"Say no.","max_tokens":4,"text":"no","prompt_tokens":3,"generated_tokens":2}]}`,
			descriptor: "exact-generation fixture 2", identities: [6]string{
				"profile:sha256:6eab96bda384f05b2abece71bbd85a9e3c36799ead0e9ad75d4c788a76a9d437", "profile:sha256:a5b6dce461570ae8552284f0af0f10e2903c38dafe329efd549e04b77ecd76e9",
				"dataset:sha256:c9d7a3cf1c41a8935715862befc60df3aec4f52a42c1ce7034029dcf2ee75a8b", "dataset-shard:sha256:5f3b3ff6e750c00b2e92bef4f974f87d13a429adcce50dd858a24b68772add99",
				"profile:sha256:bce4b9e8c9e3e81bad3f23ef79cd36be908cab27580a9ad9565e33859316a6bb", "profile:sha256:a121b63af7e6e9f22d461cc2f972f7b6c7fd688f8853518a5113782cf741d7ad",
			}},
		"sequence-scoring": {document: `{"kind":"sequence-scoring","schema":"fixture/v1","source":"fixture","cases":[
			{"name":"seq-1","group":"mrna","text":"ACGTACGT"},{"name":"seq-2","group":"mrna","text":"TTGGCCAA"}]}`,
			descriptor: "sequence-scoring fixture 2", identities: [6]string{
				"profile:sha256:61c89426d0e07181bb5dabcf79ee15d3a72ef87b53e9ef4dfd5f6991e7541ab9", "profile:sha256:e43791ca9bcb2c8b9728acb46079da7867d546122875fc1d678d9788e1f6bc82",
				"dataset:sha256:08baeb3910eea55f898065253a8de310c413d58065870fde463e1783e9296c1f", "dataset-shard:sha256:4cc7497f9f93a1f8a5272aa8182458c15b6f15ad64af0b80dfd42d9a86f97459",
				"profile:sha256:ab9f0efab0ecbe03530628febc1a9eca98d3fb6ce0bf527569ca692e263167e6", "profile:sha256:615408bb9e2ddae041170e9898d78c3cc7502440d6601d443ced7e6344b31cc1",
			}},
		"image-retrieval": {document: `{"kind":"image-retrieval","schema":"fixture/v1","source":"fixture","cases":[
			{"name":"a1","group":"a","image":"a1.png"},{"name":"a2","group":"a","image":"a2.png"},{"name":"solo","group":"solo","image":"solo.png"}]}`,
			descriptor: "image-retrieval fixture 3", identities: [6]string{
				"profile:sha256:bdd92ad97a02efb78412dced6c0f7b11baa707421a28593b2116569ad60649e1", "profile:sha256:bec192be364453af9b4178db35986faa03589d60fbec2d18f847f13dae2b2fd3",
				"dataset:sha256:fef02c231a93e96aad3eaef5618234fd67bada702c5a29c4b68fe041857905aa", "dataset-shard:sha256:4212689cb3a00f1a3c9b9995e3d172da2675ae426ce752e3f06080c0abf548ad",
				"profile:sha256:7eb41069df4565969de04e14f97a2c363b03ed1c8ed3ffaadcbf9c30c093ceae", "profile:sha256:870fc85587801234a35d5017617d13dcfd0ea0499199744b2e6314784dd4d5e8",
			}},
		"reconstruction": {document: `{"kind":"reconstruction","schema":"fixture/v1","source":"fixture","cut":{"frames":5,"fps":16,"width":8,"height":8},"cases":[
			{"name":"one level","clip":"one.mp4","sha256":"abababababababababababababababababababababababababababababababab"},
			{"name":"ten levels","clip":"ten.mp4","sha256":"cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd"}]}`,
			descriptor: "reconstruction fixture 2", identities: [6]string{
				"profile:sha256:41ffd5548252f6f077418d0763181ad5872ee5675873f6879caf90955c6f8001", "profile:sha256:d53f314feddcff2560314cf45322fa6a6a1b7788798c8c686c65b98a0c9514af",
				"dataset:sha256:4eff7d48fcb5617a53c5154f6bb6213e1dafec31d9ebe557d54db1c806516db0", "dataset-shard:sha256:b62248b469c54c1a1e0502bfa556f13b974963f3387f9bfabfe043b52cd4c4dc",
				"profile:sha256:8b1b9ce30eb92e0257691426602e0d6c41c2e9dce6afc7f09899d5de2ae2585d", "profile:sha256:10befa96741382985d2d15d4025aada87baf8fd1ee661a0c635f797fbb4da384",
			}},
		"speech-round-trip": {document: `{"kind":"speech-round-trip","schema":"fixture/v1","source":"fixture","voice":"alba","seed":7,"judge":"recognizer-a","cases":[
			{"name":"stew","text":"He hoped there would be stew."},{"name":"mat","text":"The cat sat on the mat."}]}`,
			descriptor: "speech-round-trip fixture 2", identities: [6]string{
				"profile:sha256:7b38a7fcac5b7250eba9e51b8daaccafead322b8405e14079935c545d9e7121e", "profile:sha256:47dbdea69116aa6653ca40e1d232c92919ac74f5c1502bd5fd202ff508decdbb",
				"dataset:sha256:3b8f8544e28a2b3361bda825a80093d4301c929da45c4c1beb0962df293d854c", "dataset-shard:sha256:bfe8d58441f15ca98721585c85c29b2df5ffe4a23b49e32e2d86151167b43591",
				"profile:sha256:68c5f52145010bb30c551ccbb8d74ebc79fcb3e99b882b9f72f475969d62a861", "profile:sha256:7e15a7ca03d8ca16c02a512ea8ce4fac4965069e9026f55f6d343eb62a69d139",
			}},
		"speech-round-trip counted one way": {document: `{"kind":"speech-round-trip","schema":"fixture/v1","source":"fixture","voice":"alba","seed":7,"judge":"recognizer-a",
			"normalization":["whisper-english"],"cases":[{"name":"numbers","text":"She counted ten boats."}]}`,
			descriptor: "speech-round-trip fixture 1", identities: [6]string{
				"profile:sha256:203267616c45ad43297eb2ab5c334cab71692c18690dee52ccc2b3c9c041ab2a", "profile:sha256:c4693db76606f67a05cb57ddbf990e22bd0627dee7d267a27a827b09acac65ab",
				"dataset:sha256:c42afb0f1df6df7aa49050876898ef2ce6a44f39f5fa683efc243c67717564bb", "dataset-shard:sha256:4299f5ba5e1c6c881a47b0d1aa1d8373d1ed077ce5771a36663bb1d4264eb9af",
				"profile:sha256:5b2345eb43fd13c83b1134feaf80d0fb238a19492ca81b8b9eece30bc55df8fd", "profile:sha256:7e15a7ca03d8ca16c02a512ea8ce4fac4965069e9026f55f6d343eb62a69d139",
			}},
		"description-matching": {document: `{"kind":"description-matching","schema":"fixture/v1","source":"fixture","cases":[
			{"name":"clip","prompt":"Which describes the clip?","max_tokens":4,"candidates":["a cat sleeps","a dog runs"],"answer":1}]}`,
			descriptor: "description-matching fixture 1", identities: [6]string{
				"profile:sha256:f5ca6de0ffaec08887f132df65f04022e4df4c6275096a2853a5fda1f1c5f934", "profile:sha256:5d91a2af23b648c13205632ffd54a4a896c5f77f23e64723daa255db041a4858",
				"dataset:sha256:53c5633517bc3443e52a2e63a2ea14199f48400c632898ae7c1162d347c6b0e0", "dataset-shard:sha256:411ca0deea5c0024b8145389ebbc373dc3b483f31fac0d0ac4012085b2af3ecf",
				"profile:sha256:ef7dbfe96666ee8eb27b1970883573638753132e52f99d2c0177bd5d55926bfa", "profile:sha256:2f37fbf590c64d42942b4252c222a8ac88be238a799b2b2be12a578f9ca111b0",
			}},
		"series-forecast": {document: `{"kind":"series-forecast","schema":"fixture/v1","source":"fixture","cases":[
			{"name":"rising","context":[1,2,3],"target":[4,5]},{"name":"flat","context":[7,7],"target":[7,9,7]}]}`,
			descriptor: "series-forecast fixture 2", identities: [6]string{
				"profile:sha256:6e344c6ba4a56aa0c6143056c6f361760cbfba7d0e9bd0b90e548bd7c7997005", "profile:sha256:207c00eff689b7cb67ead742547a85d3bf95543e78bbd60f03c63664ecebf98b",
				"dataset:sha256:afc4880574dfcc0bc8c254514416392701f3491d99585e3c1ae6de6eafe2382c", "dataset-shard:sha256:b4dea0f665b4259f7376d53aa33fc7a9c270c0beae67f2d03d1d64a03dbd41d7",
				"profile:sha256:d4485813870300c3cb7f2436b64dd80bc816f3ac4d4679114f2a1f7bf57d00c8", "profile:sha256:f452c3d624157c5a71966f184db4950de999f9f0bfa22eba3731d30c0e0c2c18",
			}},
		"carried-continuation": {document: `{"kind":"carried-continuation","schema":"fixture/v1","source":"fixture","chunks":2,"opening":1,"cases":[
			{"name":"river","text":"The river ran past the mill and on to the sea."}]}`,
			descriptor: "carried-continuation fixture 1", identities: [6]string{
				"profile:sha256:47d60e76cae68f51c695583130f05d9a763312fbd0ceaf86cd7d96ca6541c855", "profile:sha256:f597611eb992731806ec0b3c8bb59c807dabd7da01dc09d7005fbd60ca243276",
				"dataset:sha256:c736ba53cb4b6ff49b593c9aadd77b7b1bdcfc4a1df2364d01d397902a51494e", "dataset-shard:sha256:a62f147e3c3e2714c89ecc01c197c182fc09a9918bf8460a358a722c1ad0c538",
				"profile:sha256:4c566bb7d213a7da02243e5144c648c92855405c152871962b9d66ec2549973b", "profile:sha256:37ba80e6a713753843e2e81e58fd89c4fb58b75ae1507ea33faeda93a45ee817",
			}},
		"function-call": {document: `{"kind":"function-call","schema":"fixture/v1","source":"fixture","cases":[
			{"name":"weather","prompt":"What is the weather in Oslo?","max_tokens":16,"function":"get_weather","arguments":{"city":["Oslo"],"unit":["celsius","c"]}}]}`,
			descriptor: "function-call fixture 1", identities: [6]string{
				"profile:sha256:b2a4dfb6ee98968dfcc7d4f7a0805031eb1d2df1ac4228c84198834f1a6eaf42", "profile:sha256:474aca7e17f9120afa6043e2681c5ba957b620e47e36da77e982d3e41c5701e4",
				"dataset:sha256:f4b21ed04ccc0d71e7c931348a7a0f2c7eb28f96e6032ca913d69948815ffca5", "dataset-shard:sha256:5eec4d35ffe77a397bb0665c7035d7d54ddfce23e6874451b0e15b11b74ece5c",
				"profile:sha256:748ad14f47691a3803e3eafa5a04e4acecce511b70faa24e0a11c29634eda2d5", "profile:sha256:6164d2ca4e0a26e054a40a160067cfc2abc77f5ef9cb5035cbdbc9a2a5be3d51",
			}},
		"generated-answer": {document: `{"kind":"generated-answer","schema":"fixture/v1","source":"fixture","transforms":["trim-space","lowercase"],"cases":[
			{"name":"capital","prompt":"The capital of France is","max_tokens":4,"answers":["paris","Paris."]}]}`,
			descriptor: "generated-answer fixture 1", identities: [6]string{
				"profile:sha256:07df0f5e1fe67e4534b8d287c95e0c41a2af670f28b8102b80fecc08572c9461", "profile:sha256:c5df975dbe69b153924abe9bc09e138e9a40783e024e9f5a7742d4f57dba1925",
				"dataset:sha256:2453b04c693971fe006d92664c83ee246d8c9e9d82de11550c1ebdf41f51b202", "dataset-shard:sha256:f3646198c5d4aecf5b6883b30cb05216c60a2d388ed33d187cc5a09c2a5e8d9c",
				"profile:sha256:3e8b73d65cd2bda89f4051ddc9a7b292ff7da38756ae18dbae0753ed8ca19fab", "profile:sha256:7d2f851f8b66bd781ced1dba9e5b0792008b17c8008db2d7bdb7fe9856de4e40",
			}},
		"probability-mass-choice": {document: `{"kind":"probability-mass-choice","schema":"fixture/v1","source":"fixture","normalization":"sum","cases":[
			{"name":"breathing","prompt":"Q: Can people breathe under water?\nA:","candidates":[" No."," Correct."," Yes."],"positive":[true,true,false]}]}`,
			descriptor: "probability-mass-choice fixture 1", identities: [6]string{
				"profile:sha256:3343d402184d0916474a5ada6295211368778d2cd9b1738c6aef4a8d57979f02", "profile:sha256:c1acf0caa65d2d8acc4808040d511f2bfe6304322840e11881bd7b59d6006994",
				"dataset:sha256:f7b9fdbf533758d1647100e289592c1e5912f5b31e704e4379b393fcd5249eae", "dataset-shard:sha256:aca864f2482c7cc1cf91a14329e2d9f5093cf6019c9dd5d0a4cf6536f5e9c543",
				"profile:sha256:2c35c87fe7ee34089ed803aebc66f240c56c35081cf383558a41ee7e50597e3e", "profile:sha256:beaf3257c22d1bdf6edec59d21354b5b7743e1da26bd9cbc1a5c4dc25a683551",
			}},
		"structured-generated": {document: `{"kind":"structured-generated","schema":"lm-eval/hendrycks-math/v1.0","source":"fixture",
			"extractor":"dollar-span-or-full","equivalence":"latex-surface","cases":[
			{"name":"fraction","group":"algebra","prompt":"Problem: Divide one by two.\nAnswer:","max_tokens":8,"answers":["\\frac{1}{2}"]}]}`,
			descriptor: "structured-generated fixture 1", identities: [6]string{
				"profile:sha256:4865c569ce5c16919bb87762f4501f5cc372aab3b4b8bb18dbb1d817f793400c", "profile:sha256:5b2e132d60580b53d34c8e8216642a92838c21666dde05658824b824855c2f00",
				"dataset:sha256:9ed150bc828bd7ddb6b55c87498f2af894f49efcf3d27f1ecaec2762b6d4ef93", "dataset-shard:sha256:e2966c7298d505b63d30d9b13f7a929e361b052ec1b886d1f59f370c5e4d9c74",
				"profile:sha256:6012ec33b9d44b8339c67c452427566c827ac7ad89fa310954e1f7600734437f", "profile:sha256:6e534c6c6d1fabec1e269af2a374182705c37f6fdcfe97c6c43a18dc0534cc15",
			}},
		"instruction-rules": {document: `{"kind":"instruction-rules","schema":"lm-eval/ifeval/v4.0","source":"fixture","cases":[
			{"name":"uppercase","prompt":"Reply in uppercase and include HELLO.","max_tokens":8,"rules":[
			{"name":"uppercase","kind":"uppercase"},{"name":"keyword","kind":"contains-all","values":["HELLO"]}]}]}`,
			descriptor: "instruction-rules fixture 1", identities: [6]string{
				"profile:sha256:e79442fe5c84c50cdaad9a8fd688d3eda0a08f1a762e6c09a3e4866dbbcc01af", "profile:sha256:22bffc36c8f7bc1409914419729703f835aa112b0a17ba6a94329a1ddf5e5ce8",
				"dataset:sha256:23806e3a677a0e5aedf4ebe526b12305dda319c903899109b72ece0100a046d7", "dataset-shard:sha256:52b471ae5297ece61b6d6bf28eb0727941d127907734055acfab9c590987ac2f",
				"profile:sha256:c76f484a387fe3de4a3e77ccdfc6791c3ff96c17df721bf288125bbddc6cd393", "profile:sha256:99b95a37f9115fbeef6eecfc08e030a9be8962714d7d9d8cfe1277215220230d",
			}},
	} {
		suite, err := CompileSuite([]byte(pinned.document), ListingAuthorities())
		if err != nil {
			t.Errorf("the %s fixture does not compile: %v", label, err)
			continue
		}
		plan := suite.Plan()
		comparison, err := plan.comparisonIdentity()
		if err != nil {
			t.Fatal(err)
		}
		bound := [6]string{}
		for index, id := range [6]fmt.Stringer{plan.Identity(), comparison, plan.body.Dataset, plan.body.Split, plan.body.CaseProfile, plan.body.Scorer} {
			bound[index] = id.String()
		}
		descriptor := suite.Descriptor()
		if got := fmt.Sprintf("%s %s %d", descriptor.Kind, descriptor.Source, descriptor.Cases); got != pinned.descriptor || bound != pinned.identities {
			t.Errorf("the %s fixture compiles to %q and binds\n%q\nwant %q and\n%q", label, got, bound, pinned.descriptor, pinned.identities)
		}
	}
}
