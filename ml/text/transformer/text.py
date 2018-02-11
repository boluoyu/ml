from ml.text.transformer.base import Transformer


class TextTransformer(Transformer):
    name = 'text_transformer'

    def transform(self, text):
        return self.preproccess(text)

    def preproccess(self, text):
        return text
