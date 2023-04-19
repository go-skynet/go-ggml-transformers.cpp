package gpt2_test

import (
	. "github.com/go-skynet/go-gpt4all-j.cpp"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("gpt2 binding", func() {
	Context("Declaration", func() {
		It("fails with no model", func() {
			model, err := New("not-existing", 1024)
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})
	})
})
