package gpt2_test

import (
	. "github.com/go-skynet/go-ggml-transformers.cpp"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("gpt2 binding", func() {
	Context("Declaration", func() {
		It("fails with no model", func() {
			model, err := FalconBackendInitializer.Defaults("not-existing")
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})
	})
})
