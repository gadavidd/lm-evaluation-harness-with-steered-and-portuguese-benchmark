import re
import textdistance
from unidecode import unidecode
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("custom")
class CustomFilter(Filter):
    """
    Custom filter that applies a custom, user-defined function to the model responses.
    """

    def __init__(self, **kwargs) -> None:
        self.filter_fn = kwargs.pop("filter_fn")

        super().__init__(**kwargs)

    def apply(self, resps, docs):
        return self.filter_fn(resps, docs)

@register_filter("find_choices")
class ChoicesFilter(Filter):
    def __init__(self, choices=None, fallback="[invalid]", regex_patterns=None):
        if choices is None:
            choices = ["A", "B", "C", "D", "E"]
        self.choices = set(choices)
        self.fallback = fallback
        self.regex_patterns = [re.compile(p) for p in (regex_patterns or [])]

    def _extract_choice(self, text: str) -> str:
        if not isinstance(text, str):
            return self.fallback

        text = text.strip()

        if text in self.choices:
            return text

        for regex in self.regex_patterns:
            match = regex.search(text)
            if match:
                value = match.group(1).strip()
                if value in self.choices:
                    return value

        return self.fallback

    def apply(self, resps, docs):
        output = []
        for inst in resps:
            current = []
            for resp in inst:
                if isinstance(resp, tuple):
                    resp = resp[0]
                current.append(self._extract_choice(resp))
            output.append(current)
        return output

    def process_resp(self, text):
        text = text.strip()

        if text in self.choices:
            return text

        for regex in self.regex_patterns:
            match = re.search(regex, text)
            if match:
                return match.group(1)

        return self.fallback

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.process_resp(resp) for resp in inst]

        return [filter_set(resp) for resp in resps]

@register_filter("find_similar_label")
class SimilarLabelFilter(Filter):
    def __init__(
        self,
        labels,
        fallback="[invalid]"
    ) -> None:
        self.labels = labels
        self.fallback = fallback

    def process_resp(self, prediction):
        norm_label = [unidecode(s.strip().lower()) for s in self.labels]
        prediction = unidecode(prediction.strip().lower())

        if prediction in norm_label:
            return self.labels[norm_label.index(prediction)]

        if prediction == "":
            return self.fallback

        count_matches = 0
        last_match = self.fallback
        for label in norm_label:
            if label in prediction:
                count_matches += 1
                last_match = label
        if count_matches == 1:
            return self.labels[norm_label.index(last_match)]

        get_text_until = [".", ",", ";", ":", "(", ")", "[", "]", "\n"]
        for split_char in get_text_until:
            if split_char in prediction:
                prediction = prediction[:prediction.find(split_char)]

        max_length = max(len(s) for s in norm_label)
        prediction = prediction[:max_length]

        similarities = [
            textdistance.levenshtein.normalized_similarity(prediction, label)
            for label in norm_label
        ]

        if max(similarities) < 0.5:
            prediction = self.fallback
        else:
            prediction = self.labels[similarities.index(max(similarities))]

        return prediction

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.process_resp(resp) for resp in inst]

        return [filter_set(resp) for resp in resps]